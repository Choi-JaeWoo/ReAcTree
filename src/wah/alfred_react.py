import torch
import guidance

import os
import pdb
import datetime
from PIL import Image
from src.alfred.utils import dotdict, load_task_json
import src.alfred.utils as utils
from react import React

class AlfredReact(React):
    def run(self, task_d, args_dict, log):
        ################################################
        log.info(task_d)

        task_name = task_d['task']
        repeat_idx = task_d['repeat_idx']
        traj_data = load_task_json(task_d)
        instruction_text = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        model_args = dotdict(args_dict)

        task_type = task_name.split('/')[0]
        trial_num = task_name.split('/')[1]
        
        try:
            # setup scene
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']
            # scene_room = traj_data['scene']['floor_plan']

            scene_name = 'FloorPlan%d' % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
            if self.cfg.llm_agent.working_memory:
                self.env.reset_working_memory()

            # initialize
            self.env.step(dict(traj_data['scene']['init_action']))
            self.env.set_task(traj_data, model_args, reward_type='dense')
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error: ", repr(e))

        init_obs = self.env.init_reset(traj_data)
        init_obs_text = init_obs['text']
        nl_inst_info = {'nl_inst': instruction_text, 'task_type': traj_data['task_type'],'message': None}
        self.llm_agent.reset(nl_inst_info, init_obs_text)
        self.cur_step_id, self.cur_decision_id = 1, 1
        log.info(f'Your task is to: {instruction_text}')
        log.info(init_obs_text)

        skill_set = self.llm_agent.update_skill_set(init_obs)

        self.env.vis_log = [{'action': 'init', 'images': Image.fromarray(self.env.last_event.frame)}]

        try_count = 0
        max_try   = 5
        while True:
            if self.cur_decision_id > self.max_decisions:
                terminate_info = {'terminate': 'max_decision', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied(), 'nl_inst': instruction_text}
                log.info('Max steps')
                return terminate_info
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
            except Exception as error_message:
                ### TODO: except case
                terminate_info = {'terminate': 'plan_next_step_error', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied(), 'nl_inst': instruction_text}
                log.info(f"Plan Next Step Error: {error_message}")
                return terminate_info

            if next_step_class == 'Think':
                self.cur_decision_id += 1
                log.info('OK.')
                pass
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied(), 'nl_inst': instruction_text}
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied(), 'nl_inst': instruction_text}
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    log.info(obs_text)
                    self.llm_agent.add_obs(obs_text)
                    self.cur_step_id += 1
                    self.cur_decision_id += 1
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
                else:
                    obs = self.env.llm_skill_interact(next_step)
                    obs_text = obs['message']
                    self.llm_agent.add_obs(obs['message'])
                    self.cur_step_id += 1
                    self.cur_decision_id +=1
                    log.info(obs['message'])
                    skill_set = self.llm_agent.update_skill_set(obs)
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
            elif next_step_class == 'Error':
                self.cur_step_id += 1
                self.cur_decision_id +=1
                if try_count == max_try:
                    log.info("    Reach Maximum GPT generation Error try count : ", try_count)
                    log.info("   Terminate the task")
                    cur_decision_id = 999
                    cur_step_id = 999 
                log.info("    GPT generation Error, try count : ", try_count)
                try_count+=1
            else:
                raise NotImplementedError()
        ################################################

    def collect(self, task_d, args_dict):
        # task trajectory loading 
        task_name = task_d['task']
        repeat_idx = task_d['repeat_idx']
        traj_data = load_task_json(task_d)
        nl_inst = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        model_args = dotdict(args_dict)

        task_type = task_name.split('/')[0]
        trial_num = task_name.split('/')[1]
        
        # trajectory collection data generatation 
        text_trajectory_base_dir = self.cfg.dataset.collect_dir
        os.makedirs(text_trajectory_base_dir, exist_ok=True)
        base_file_name = f'{task_type}_{trial_num}_ann_{repeat_idx}'
        base_file_path = os.path.join(text_trajectory_base_dir, f'{base_file_name}.txt')
        
        if os.path.exists(base_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(text_trajectory_base_dir, f'{base_file_name}_{timestamp}.txt')
        else:
            collect_file_path = base_file_path
        print('collect_file_path: ', collect_file_path)
        task_idx = task_d['alfred_train_task_idx']
        train_set_num = task_d['total_train_num']
        print(f'\nData Collection for ALFRED train data [{task_idx}/{train_set_num}] | ({task_name} - reapeat_idx [{repeat_idx}])')
        print(f'Instruction:\n{nl_inst}\n')

        avail_skill_set = ["go to", "pick up", "put down", "open", "close", "turn on", "turn off", "slice"]
        print(f'Available skill set: {avail_skill_set}')
        
        # Trajectory collection 
        # instruction logging 
        print(f'Your task is to: {nl_inst}')
        with open(collect_file_path, 'w') as file:
            file.write(f'Your task is to: {nl_inst}\n')

        # Scene reconstruction 
        # Camera reconstruction 
        try:
            # setup scene
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']
            # scene_room = traj_data['scene']['floor_plan']

            scene_name = 'FloorPlan%d' % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            # initialize
            self.env.step(dict(traj_data['scene']['init_action']))
            self.env.set_task(traj_data, model_args, reward_type='dense')
            if self.cfg.llm_agent.working_memory:
                self.env.reset_working_memory()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error: ", repr(e))
            
        # initial react observation logging 
        init_obs = self.env.init_reset(traj_data)
        nl_inst_info = {'nl_inst': nl_inst, 'task_type': traj_data['task_type'],'message': None}
        
        print(f'{init_obs["text"]}')
        with open(collect_file_path, 'a') as file:
            file.write(f'{init_obs["text"]}\n')

        # interactive trajectory collection start
        while True:
            decision = input('Type decision ("r" or "a"): ')
            if decision == 'r':
                reasoning_text = input('Think: ')
                with open(collect_file_path, 'a') as file:
                    file.write(f'Think: {reasoning_text}\nOK.\n')
            elif decision == 'a':
                action_text = input('Act: ')
                if action_text in ['done', 'failure']:
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {action_text}')
                    break
                elif action_text.startswith('recall location of '):
                    target_obj = action_text.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(
                        self.env.working_memory, 
                        target_obj
                    )
                    print(obs_text)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {action_text}\n{obs_text}\n')
                else:
                    try:
                        action_ret = self.env.llm_skill_interact(action_text) 
                        obs_text = action_ret['message']
                        print('react.py obs_text: ', obs_text)
                        with open(collect_file_path, 'a') as file:
                            file.write(f'Act: {action_text}\n{obs_text}\n')     
                    except Exception as e:
                        print("Error. Type again.", e)
            else:
                print(f'Error: Type again. ("r" or "a") your input {decision}')
                pass

    def collect_llm(self, task_d, args_dict):
        task_name = task_d['task']
        repeat_idx = task_d['repeat_idx']
        traj_data = load_task_json(task_d)
        instruction_text = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        model_args = dotdict(args_dict)

        task_type = task_name.split('/')[0]
        trial_num = task_name.split('/')[1]
        
        text_trajectory_base_dir = self.cfg.dataset.collect_dir
        
        os.makedirs(text_trajectory_base_dir, exist_ok=True)
        base_file_name = f'{task_type}_{trial_num}_ann_{repeat_idx}'
        base_file_path = os.path.join(text_trajectory_base_dir, f'{base_file_name}.txt')
        
        if os.path.exists(base_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(text_trajectory_base_dir, f'{base_file_name}_{timestamp}.txt')
        else:
            collect_file_path = base_file_path


        try:
            # setup scene
            scene_num = traj_data['scene']['scene_num']
            object_poses = traj_data['scene']['object_poses']
            dirty_and_empty = traj_data['scene']['dirty_and_empty']
            object_toggles = traj_data['scene']['object_toggles']
            # scene_room = traj_data['scene']['floor_plan']

            scene_name = 'FloorPlan%d' % scene_num
            self.env.reset(scene_name)
            self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)

            # initialize
            self.env.step(dict(traj_data['scene']['init_action']))
            self.env.set_task(traj_data, model_args, reward_type='dense')
            if self.cfg.llm_agent.working_memory:
                self.env.reset_working_memory()

        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Error: ", repr(e))
            
        init_obs = self.env.init_reset(traj_data)
        init_obs_text = init_obs['text']
        nl_inst_info = {'nl_inst': instruction_text, 'task_type': traj_data['task_type'],'message': None}
        self.llm_agent.reset(nl_inst_info, init_obs_text)
        self.cur_step_id, self.cur_decision_id = 1, 1
        print(f'Your task is to: {instruction_text}')
        print(init_obs_text)

        with open(collect_file_path, 'w') as file:
            file.write(f'Your task is to: {instruction_text}\n{init_obs_text}\n')

        skill_set = self.llm_agent.update_skill_set(init_obs)

        try_count = 0
        max_try   = 5
        while True:
            if self.cur_decision_id > self.max_decisions:
                terminate_info = {'terminate': 'max_decision', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied()}
                print('Max decisions')
                with open(collect_file_path, 'a') as file:
                    file.write('Max decisions')
                return terminate_info
            
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                print(f'{next_step_class}: {next_step}')
            except Exception as error_message:
                ### TODO: except case
                terminate_info = {'terminate': 'plan_next_step_error', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied()}
                print(f"Plan Next Step Error: {error_message}")
                return terminate_info
            
            if next_step_class == 'Think':
                self.cur_decision_id += 1
                with open(collect_file_path, 'a') as file:
                    file.write(f'Think: {next_step}\nOK.\n')
                pass
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied()}
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n')
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id, 'success': self.env.get_goal_satisfied()}
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n')
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    print(obs_text)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n{obs_text}\n')
                    self.llm_agent.add_obs(obs_text)
                    self.cur_step_id += 1
                    self.cur_decision_id += 1
                else:
                    obs = self.env.llm_skill_interact(next_step)
                    obs_text = obs['message']
                    self.llm_agent.add_obs(obs['message'])
                    self.cur_step_id += 1
                    self.cur_decision_id +=1
                    print(obs['message'])
                    skill_set = self.llm_agent.update_skill_set(obs)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n{obs_text}\n')
            elif next_step_class == 'Error':
                self.cur_step_id += 1
                self.cur_decision_id +=1
                if try_count == max_try:
                    print("    Reach Maximum GPT generation Error try count : ", try_count)
                    print("   Terminate the task")
                    cur_decision_id = 999
                    cur_step_id = 999 
                print("    GPT generation Error, try count : ", try_count)
                try_count+=1
            else:
                raise NotImplementedError()