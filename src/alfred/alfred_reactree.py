import torch
import guidance

import os
import pdb
import datetime
from PIL import Image
from src.alfred.utils import dotdict, load_task_json
import src.alfred.utils as utils
from reactree import Reactree, ControlFlowNode, AgentNode

class AlfredReactree(Reactree):
    def run(self, task_d, args_dict, log):
        ############################
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
        
        # interactive trajectory collection start
        init_obs = self.env.init_reset(traj_data)
        init_obs_text = init_obs['text']
        nl_inst_info = {'nl_inst': instruction_text, 'message': None, 'task_type': traj_data['task_type']}

        self.env.vis_log = [{'action': 'init', 'images': Image.fromarray(self.env.last_event.frame)}]

        self.root_node = AlfredAgentNode(self.cfg, nl_inst_info, 1, self.llm_agent, self.env)
        terminate_info = self.root_node.run(traj_data, 1, 1, log)
        ###########################
        terminate_info['success'] = self.env.get_goal_satisfied()
        terminate_info['nl_inst'] = instruction_text
        return terminate_info

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
        collect_dir = self.cfg.dataset.collect_dir
        collect_file_base_name = f'{task_type}_{trial_num}_ann_{repeat_idx}'
        os.makedirs(collect_dir, exist_ok=True)

        task_idx = task_d['alfred_train_task_idx']
        train_set_num = task_d['total_train_num']
        print(f'\nData Collection for ALFRED train data [{task_idx}/{train_set_num}] | ({task_name} - reapeat_idx [{repeat_idx}])')
        print(f'Instruction:\n{nl_inst}\n')

        avail_skill_set = [
            "go to", "pick up", "put down", "open", "close", 
            "turn on", "turn off", "slice", "done", "failure"
        ]
        print(f'Available skill set: {avail_skill_set}')

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

        # interactive trajectory collection start
        nl_inst_info = {'nl_inst':nl_inst, 'task_type': task_d['task_type']}

        self.root_node = AlfredAgentNode(self.cfg, nl_inst_info, 1, self.llm_agent, self.env)
        self.root_node.collect(traj_data, 1, 1, collect_dir, collect_file_base_name)

    def collect_llm(self, task_d, args_dict):
        # task trajectory loading 
        task_name = task_d['task']
        repeat_idx = task_d['repeat_idx']
        traj_data = load_task_json(task_d)
        nl_inst = traj_data['turk_annotations']['anns'][repeat_idx]['task_desc']
        model_args = dotdict(args_dict)

        task_type = task_name.split('/')[0]
        trial_num = task_name.split('/')[1]

        # trajectory collection data generatation 
        collect_dir = self.cfg.dataset.collect_dir
        collect_file_base_name = f'{task_type}_{trial_num}_ann_{repeat_idx}'
        os.makedirs(collect_dir, exist_ok=True)

        print(f'Instruction:\n{nl_inst}\n')

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
        
        # interactive trajectory collection start
        init_obs = self.env.init_reset(traj_data)
        init_obs_text = init_obs['text']
        nl_inst_info = {'nl_inst': nl_inst, 'message': None, 'task_type': traj_data['task_type']}

        self.root_node = AlfredAgentNode(self.cfg, nl_inst_info, 1, self.llm_agent, self.env)
        terminate_info = self.root_node.collect_llm(traj_data, 1, 1, collect_dir, collect_file_base_name)
        terminate_info['success'] = self.env.get_goal_satisfied()
        terminate_info['nl_inst'] = nl_inst
        return terminate_info

class AlfredAgentNode(AgentNode):
    def run(self, traj_d, cur_step_id, cur_decision_id, log):
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['message'] = message
        nl_inst_info['depth'] = self.depth
        init_obs = self.env.init_reset(traj_d)
        init_obs_text = init_obs['text']
        self.llm_agent.reset(nl_inst_info, init_obs_text)

        skill_set = self.llm_agent.update_skill_set(init_obs)

        if message == None:
            log.info(f'Your task is to: {nl_inst}')
        elif message == '':
            log.info(f'Your task is to: {nl_inst}')
        else:
            log.info(message)
            log.info(f'Your task is to: {nl_inst}')
        log.info(init_obs_text)

        #######################################
        try_count = 0
        max_try   = 5
        while True:
            if cur_decision_id > self.cfg.llm_agent.max_decisions: # max decisions
                terminate_info = {'success': False, 'terminate': 'max_decision', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                log.info('Max decisions')
                return terminate_info
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
            except Exception as error_message:
                ### TODO: except case
                terminate_info = {'success': False, 'terminate': 'plan_next_step_error', 'step_id': cur_step_id, 'decision_id': cur_decision_id, 'success' : False}
                log.info(f"Plan Next Step Error: {error_message}")
                return terminate_info
            
            if next_step_class == 'Think':  
                cur_decision_id += 1
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id, 'success' : True}
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'step_id': cur_step_id, 'decision_id': cur_decision_id, 'success' : False}
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    log.info(obs_text)
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 1
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
                else:
                    obs = self.env.llm_skill_interact(next_step)
                    obs_text = obs['message']
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id +=1
                    skill_set = self.llm_agent.update_skill_set(obs)
                    log.info(obs_text)
                    self.env.vis_log.append({'action': next_step, 'images': Image.fromarray(self.env.last_event.frame)})
            elif next_step_class == 'Expand':
                cur_decision_id += 1
        
                control_flow = next_step['control_flow']
                subgoals = next_step['conditions'].split(', ')

                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)
                self.env.vis_log.append({'action': f'Expand: {control_flow}', 'images': Image.fromarray(self.env.last_event.frame)})

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': nl_inst_info['task_type']}
                    agent_node = AlfredAgentNode(self.cfg, subgoal_info, self.depth + 2, self.llm_agent, self.env)
                    control_flow_node.add_child(agent_node)
                
                return self.children[0].run(cur_step_id, cur_decision_id, log, traj_d=traj_d)
            elif next_step_class == 'Error':
                cur_step_id += 1
                cur_decision_id +=1
                if try_count == max_try:
                    log.info(f"    Reach Maximum GPT generation Error try count : {try_count}")
                    log.info(f"   Terminate the task")
                    cur_decision_id = 999
                    cur_step_id = 999 
                log.info(f"    GPT generation Error, try count : {try_count}")
                try_count+=1
            else:
                raise NotImplementedError()

        #######################################
            
    def collect(self, traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name):
        AVAILABLE_CONTROL_FLOW = ['sequence', 'fallback', 'parallel']
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['message'] = message

        # initial react observation logging 
        init_obs = self.env.init_reset(traj_d)

        collect_file_name = f'{collect_dir}/{collect_file_base_name}_dec{str(cur_decision_id).zfill(3)}_depth{str(self.depth).zfill(2)}'
        collect_file_path = collect_file_name+'.txt'

        if os.path.exists(collect_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(f'{collect_file_name}_{timestamp}.txt')
            with open(collect_file_path, 'w') as file:
                file.write(f'{init_obs["text"]}\n')
        else:
            with open(collect_file_path, 'w') as file:
                file.write(f'{init_obs["text"]}\n')
        init_obs_text = init_obs['text']
        
        if message == None:
            print(f'\nYour task is to: {nl_inst}\n')
            print(init_obs_text)
            with open(collect_file_path, 'w') as file:
                file.write(f'Your task is to: {nl_inst}\n{init_obs_text}\n')
        elif message == '':
            print(f'\nYour task is to: {nl_inst}\n')
            print(init_obs_text)
            with open(collect_file_path, 'w') as file:
                file.write(f'Your task is to: {nl_inst}\n{init_obs_text}\n')
        else:
            print(f'\n{message}\nYour task is to: {nl_inst}\n')
            print(init_obs_text)
            with open(collect_file_path, 'w') as file:
                file.write(f'{message}\nYour task is to: {nl_inst}\n{init_obs_text}\n')        
        
        while True:
            decision = input('Type decision ("r" or "a" or "e"): ')
            if decision == 'r':
                reasoning_text = input('Think: ')
                cur_decision_id += 1
                with open(collect_file_path, 'a') as file:
                    file.write(f'Think: {reasoning_text}\nOK.\n')

            elif decision == 'a':
                action_text = input('Act: ')
                if action_text in ['done', 'failure']:
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {action_text}')
                    if action_text == 'done':
                        terminate_info = {
                            'success': True, 'terminate': 'done', 
                            'step_id': cur_step_id, 
                            'decision_id': cur_decision_id
                        }
                    else:
                        terminate_info = {
                            'success': False, 'terminate': 'done', 
                            'step_id': cur_step_id, 
                            'decision_id': cur_decision_id
                        }
                    return terminate_info
                elif action_text.startswith('recall location of '):
                    target_obj = action_text.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(
                        self.env.working_memory, 
                        target_obj
                    )
                    print(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 1    
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {action_text}\n{obs_text}\n')
                else:
                    try:
                        action_ret = self.env.llm_skill_interact(action_text)  
                        obs_text = action_ret['message']
                        print(obs_text)
                        cur_step_id += 1
                        cur_decision_id += 1
                        with open(collect_file_path, 'a') as file:
                            file.write(f'Act: {action_text}\n{obs_text}\n')
                    except:
                        print("Error. Type again.")
                        pass 

            elif decision == 'e':
                control_flow = input('Control flow ("sequence" or "fallback" or "parallel"): ')
                if control_flow not in AVAILABLE_CONTROL_FLOW:
                    print(f"Not supported control flow {control_flow}. Type again.")
                    continue
                subgoals = input('Subgoals (connect with ","): ')
                with open(collect_file_path, 'a') as file:
                    file.write(f'Expand:\n- control flow: {control_flow}\n- subgoals: {subgoals}\n')
                subgoals = subgoals.split(', ')
                
                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': nl_inst_info['task_type']}
                    agent_node = AlfredAgentNode(
                        self.cfg, subgoal_info, 
                        self.depth + 2, self.llm_agent, self.env
                    )
                    control_flow_node.add_child(agent_node)
                return self.children[0].collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, traj_d=traj_d)      
            else:
                print("Error. Type again.")
                pass

    def collect_llm(self, traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name):
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['message'] = message
        nl_inst_info['depth'] = self.depth
        init_obs = self.env.init_reset(traj_d)
        init_obs_text = init_obs['text']
        self.llm_agent.reset(nl_inst_info, init_obs_text)

        skill_set = self.llm_agent.update_skill_set(init_obs)

        ### file save
        base_file_name = f'{collect_dir}/{collect_file_base_name}_dec{str(cur_decision_id).zfill(3)}_depth{str(self.depth).zfill(2)}'
        base_file_path = base_file_name + '.txt'

        if os.path.exists(base_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(f'{base_file_name}_{timestamp}.txt')
        else:
            collect_file_path = base_file_path
        
        if message == None:
            print(f'\nYour task is to: {nl_inst}\n')
            print(init_obs_text)
            with open(collect_file_path, 'w') as file:
                file.write(f'Your task is to: {nl_inst}\n{init_obs_text}\n')
        elif message == '':
            print(f'\nYour task is to: {nl_inst}\n')
            print(init_obs_text)
            with open(collect_file_path, 'w') as file:
                file.write(f'Your task is to: {nl_inst}\n{init_obs_text}\n')
        else:
            print(f'\n{message}\nYour task is to: {nl_inst}\n')
            print(init_obs_text)
            with open(collect_file_path, 'w') as file:
                file.write(f'{message}\nYour task is to: {nl_inst}\n{init_obs_text}\n') 
        
        try_count = 0
        max_try   = 5
        while True:
            if cur_decision_id > self.cfg.llm_agent.max_decisions: # max decisions
                terminate_info = {'success': False, 'terminate': 'max_decision', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                print('Max decisions')
                with open(collect_file_path, 'a') as file:
                    file.write('Max decisions')
                return terminate_info
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
            except Exception as error_message:
                ### TODO: except case
                terminate_info = {'success': False, 'terminate': 'plan_next_step_error', 'step_id': cur_step_id, 'decision_id': cur_decision_id, 'success' : False}
                return terminate_info
            
            if next_step_class == 'Think':  
                cur_decision_id += 1
                print(f'Think: {next_step}\n')
                with open(collect_file_path, 'a') as file:
                    file.write(f'Think: {next_step}\nOK.\n')
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id, 'success' : True}
                    print(f'Act: {next_step}\n')
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n')
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'step_id': cur_step_id, 'decision_id': cur_decision_id, 'success' : False}
                    print(f'Act: {next_step}\n')
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n')
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    print(f'Act: {next_step}\n{obs_text}\n')
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n{obs_text}\n')
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 1
                else:
                    obs = self.env.llm_skill_interact(next_step)
                    obs_text = obs['message']
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id +=1
                    skill_set = self.llm_agent.update_skill_set(obs)
                    print(f'Act: {next_step}\n{obs_text}\n')
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {next_step}\n{obs_text}\n')
            elif next_step_class == 'Expand':
                cur_decision_id += 1
                print(f'Expand:\n- control flow: {next_step["control_flow"]}\n- subgoals: {next_step["conditions"]}\n')
                with open(collect_file_path, 'a') as file:
                    file.write(f'Expand:\n- control flow: {next_step["control_flow"]}\n- subgoals: {next_step["conditions"]}\n')
                
                control_flow = next_step['control_flow']
                subgoals = next_step['conditions'].split(', ')

                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': nl_inst_info['task_type']}
                    agent_node = AlfredAgentNode(self.cfg, subgoal_info, self.depth + 2, self.llm_agent, self.env)
                    control_flow_node.add_child(agent_node)
                
                return self.children[0].collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id=0, traj_d=traj_d )
            elif next_step_class == 'Error':
                cur_step_id += 1
                cur_decision_id +=1
                if try_count == max_try:
                    print(f"    Reach Maximum GPT generation Error try count : {try_count}")
                    print(f"   Terminate the task")
                    cur_decision_id = 999
                    cur_step_id = 999 
                print(f"    GPT generation Error, try count : {try_count}")
                try_count+=1
            else:
                raise NotImplementedError()
    
    def make_message(self):
        if self.cfg.llm_agent.message_type == 'goal_information':
            message = self.make_message_goal_information()
        else:
            message = None
        return message
    
    def make_message_goal_information(self):
        if self.parent == None:
            return ''
        else:
            control_flow_node = self.parent
            superagent_node = control_flow_node.parent
            sibling_nodes = control_flow_node.children
            
            supergoal = superagent_node.content['nl_inst']
            control_flow = control_flow_node.content
            sibling_goals = [sibling_node.content['nl_inst'] for sibling_node in sibling_nodes]
            
            ## V2 
            if control_flow == 'sequence':
                control_flow_phrase = 'in sequence'
            elif control_flow == 'fallback':
                control_flow_phrase = 'using a fallback strategy'
            elif control_flow == 'parallel':
                control_flow_phrase = 'in parallel'
            else:
                raise NotImplementedError()
            
            sibling_goals_phrase = ', '.join(sibling_goals[:-1]) + ', and ' + sibling_goals[-1] if len(sibling_goals) > 1 else sibling_goals[0]
            message = f'Your primary goal is to: {supergoal}\nTo achieve this, you should perform your sibling tasks {control_flow_phrase}. At this level, your sibling tasks are: {sibling_goals_phrase}.'
            return message