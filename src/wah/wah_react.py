from react import React
import os
import datetime
import wah.wah_utils as utils

class WahReact(React):
    def run(self, task_d, log):
        self.env.reset(task_d)
        task_id, nl_inst = task_d['task_id'], task_d['nl_instructions'][0]
        init_obs = self.env.get_init_obs()
        init_obs_text = init_obs['text']
        nl_inst_info = {'nl_inst': nl_inst, 'task_type': task_d['task_name'], 'message': None}
        self.llm_agent.reset(nl_inst_info, init_obs_text)
        self.cur_step_id, self.cur_decision_id = 1, 1
        
        log.info(f'Task ID: {task_id}')
        log.info(f'Your task is to: {nl_inst}')
        log.info(init_obs_text)
        
        while True:
            if self.cur_step_id > self.max_steps:
                terminate_info = {'terminate': 'max_step', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                log.info('Max steps')
                return terminate_info
            if self.cur_decision_id > self.max_decisions:
                terminate_info = {'terminate': 'max_decision', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                log.info('Max decisions')
                return terminate_info

            skill_set = self.env.get_possible_skill_set()
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
            except Exception as error_message:
                terminate_info = {'terminate': 'plan_next_step_error', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                log.info(f"Plan Next Step Error: {error_message}")
                return terminate_info
            if next_step_class == 'Think':
                self.cur_decision_id += 1
                pass
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    self.llm_agent.add_obs(obs_text)
                    self.cur_step_id += 1
                    self.cur_decision_id +=1
                    log.info(obs_text)
                else:
                    obs = self.env.step(next_step)
                    self.llm_agent.add_obs(obs['text'])
                    self.cur_step_id += 1
                    self.cur_decision_id +=1
                    log.info(obs['text'])
            elif next_step_class == 'Error':
                self.cur_step_id += 1
                self.cur_decision_id +=1
            else:
                raise NotImplementedError()
    
    def collect(self, task_d):
        self.env.reset(task_d)
        task_id, nl_inst, task_type = task_d['task_id'], task_d['nl_instructions'][0], task_d['task_name']
        init_obs = self.env.get_init_obs()
        init_obs_text = init_obs['text']
        
        collect_dir = self.cfg.dataset.collect_dir
        os.makedirs(collect_dir, exist_ok=True)
        collect_file_name = f'{task_id}_{task_type}'
        collect_file_path = os.path.join(collect_dir, f'{collect_file_name}.txt')
        if os.path.exists(collect_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(collect_dir, f'{collect_file_name}_{timestamp}.txt')
         
        print(f'\nData Collection for {task_id}-th WAH-NL data   |   ({task_type})')
        print(f'\nYour task is to: {nl_inst}\n')
        print(init_obs_text)

        with open(collect_file_path, 'w') as file:
            file.write(f'Your task is to: {nl_inst}\n{init_obs_text}\n')
        
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
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    print(obs_text)
                else:
                    try:
                        obs = self.env.step(action_text)
                        obs_text = obs['text']
                        print(obs_text)
                    except:
                        print("Error. Type again.")
                        continue 
                with open(collect_file_path, 'a') as file:
                    file.write(f'Act: {action_text}\n{obs_text}\n')
            else:
                print("Error. Type again.")
                pass

    def collect_llm(self, task_d, task_id, trial_id):
        import pdb
        self.env.reset(task_d)
        nl_inst, task_type = task_d['nl_instructions'][0], task_d['task_name']
        init_obs = self.env.get_init_obs()
        init_obs_text = init_obs['text']
        
        nl_inst_info = {'nl_inst': nl_inst, 'task_type': task_d['task_name'], 'message': None}
        
        if trial_id == 0:
            self.llm_agent.reset(nl_inst_info, init_obs_text)
            self.cur_step_id, self.cur_decision_id = 1, 1
        elif trial_id > 0:
            pdb.set_trace()
        else:
            raise NotImplementedError()

        collect_dir = self.cfg.dataset.collect_dir
        os.makedirs(collect_dir, exist_ok=True)
        collect_file_name = f'{task_id}_{task_type}'
        collect_file_path = os.path.join(collect_dir, f'{collect_file_name}.txt')
        if os.path.exists(collect_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(collect_dir, f'{collect_file_name}_{timestamp}.txt')
         
        print(f'\nData Collection for {task_id}-th WAH-NL data   |   ({task_type})')
        print(f'\nYour task is to: {nl_inst}\n')

        with open(collect_file_path, 'w') as file:
            file.write(f'Your task is to: {nl_inst}\n{init_obs_text}\n')
        
        while True:
            if self.cur_step_id > self.max_steps:
                terminate_info = {'terminate': 'max_step', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                with open(collect_file_path, 'a') as file:
                    file.write('Max steps')
                return terminate_info
            if self.cur_decision_id > self.max_decisions:
                terminate_info = {'terminate': 'max_decision', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                with open(collect_file_path, 'a') as file:
                    file.write('Max decisions')
                return terminate_info

            skill_set = self.env.get_possible_skill_set()
            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
            except Exception as error_message:
                ### TODO: except case
                terminate_info = {'terminate': 'plan_next_step_error', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                with open(collect_file_path, 'a') as file:
                    file.write(f"Plan Next Step Error: {error_message}")
                return terminate_info
            if next_step_class == 'Think':
                with open(collect_file_path, 'a') as file:
                    file.write(f'{next_step_class}: {next_step}\nOK.\n')
                self.cur_decision_id += 1
                pass
            elif next_step_class == 'Act':
                with open(collect_file_path, 'a') as file:
                    file.write(f'{next_step_class}: {next_step}')
                if next_step == 'done':
                    terminate_info = {'terminate': 'done', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'terminate': 'failure', 'step_id': self.cur_step_id, 'decision_id': self.cur_decision_id}
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'\n{obs_text}\n')
                    self.llm_agent.add_obs(obs_text)
                    self.cur_step_id += 1
                    self.cur_decision_id +=1            
                else:
                    obs = self.env.step(next_step)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'\n{obs["text"]}\n')
                    self.llm_agent.add_obs(obs['text'])
                    self.cur_step_id += 1
                    self.cur_decision_id +=1
            else:
                raise NotImplementedError()
        