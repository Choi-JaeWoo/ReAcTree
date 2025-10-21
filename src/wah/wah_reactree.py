from reactree import Reactree, ControlFlowNode, AgentNode
import os
import datetime
import wah.wah_utils as utils


class WahReactree(Reactree):
    def run(self, task_d, log):
        self.env.reset(task_d)
        task_id, nl_inst = task_d['task_id'], task_d['nl_instructions'][0]
        log.info(f'Task ID: {task_id}')
        
        nl_inst_info = {'nl_inst': nl_inst, 'task_type': task_d['task_name']}
        self.root_node = WahAgentNode(self.cfg, nl_inst_info, 1, self.llm_agent, self.env)
        terminate_info = self.root_node.run(1, 1, log)
        
        return terminate_info
    
    def collect(self, task_d):
        self.env.reset(task_d)
        task_id, nl_inst, task_type = task_d['task_id'], task_d['nl_instructions'][0], task_d['task_name']
        
        print(f'\nData Collection for {task_id}-th WAH-NL data   |   ({task_type})')

        collect_dir = self.cfg.dataset.collect_dir
        os.makedirs(collect_dir, exist_ok=True)
        collect_file_base_name = f'{task_id}_{task_type}'

        nl_inst_info = {'nl_inst': nl_inst, 'task_type': task_d['task_name']}
        self.root_node = WahAgentNode(self.cfg, nl_inst_info, 1, None, self.env)
        self.root_node.collect(1, 1, collect_dir, collect_file_base_name)
    
    def collect_llm(self, task_d, task_id, trial_id):
        self.env.reset(task_d)
        nl_inst, task_type = task_d['nl_instructions'][0], task_d['task_name']
        
        print(f'\nData Collection for {task_id}-th WAH-NL data   |   ({task_type})')

        collect_dir = self.cfg.dataset.collect_dir
        os.makedirs(collect_dir, exist_ok=True)
        collect_file_base_name = f'{task_id}_{task_type}'

        nl_inst_info = {'nl_inst': nl_inst, 'task_type': task_d['task_name']}
        self.root_node = WahAgentNode(self.cfg, nl_inst_info, 1, self.llm_agent, self.env)
        self.root_node.collect_llm(1, 1, collect_dir, collect_file_base_name, trial_id)

class WahAgentNode(AgentNode):
    def run(self, cur_step_id, cur_decision_id, log):
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['message'] = message
        init_obs = self.env.get_init_obs()
        init_obs_text = init_obs['text']
        self.llm_agent.reset(nl_inst_info, init_obs_text)
        
        if message is not None: log.info(f'{message}')
        log.info(f'Your task is to: {nl_inst}')
        log.info(init_obs_text)
        
        while True:
            if cur_step_id > self.cfg.llm_agent.max_steps:
                terminate_info = {'success': False, 'terminate': 'max_step', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                log.info('Max steps')
                return terminate_info
            if cur_decision_id > self.cfg.llm_agent.max_decisions:
                terminate_info = {'success': False, 'terminate': 'max_decision', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                log.info('Max decisions')
                return terminate_info
            
            skill_set = self.env.get_possible_skill_set()

            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
                log.info(f'{next_step_class}: {next_step}')
            except Exception as error_message:
                terminate_info = {'success': False, 'terminate': 'plan_next_step_error', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                log.info(f"Plan Next Step Error: {error_message}")
                return terminate_info
            if next_step_class == 'Think':
                cur_decision_id += 1
            elif next_step_class == 'Act':
                if next_step == 'done':
                    terminate_info = {'success': True, 'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'success': False, 'terminate': 'failure', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    log.info(obs_text)
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 1
                else:
                    obs = self.env.step(next_step)
                    log.info(obs['text'])
                    self.llm_agent.add_obs(obs['text'])
                    cur_step_id += 1
                    cur_decision_id += 1
            elif next_step_class == 'Expand':
                cur_decision_id += 1
                control_flow = next_step['control_flow']
                subgoals = next_step['conditions'].split(', ')

                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': nl_inst_info['task_type']}
                    agent_node = WahAgentNode(self.cfg, subgoal_info, self.depth + 2, self.llm_agent, self.env)
                    control_flow_node.add_child(agent_node)

                return self.children[0].run(cur_step_id, cur_decision_id, log)
            elif next_step_class == 'Error':
                cur_step_id += 1
                cur_decision_id +=1
            else:
                raise NotImplementedError()
            
    def collect(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name):
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['message'] = message

        init_obs = self.env.get_init_obs()
        init_obs_text = init_obs['text']

        collect_file_name = f'{collect_file_base_name}_dec{str(cur_decision_id).zfill(3)}_depth{str(self.depth).zfill(2)}'
        collect_file_path = os.path.join(collect_dir, f'{collect_file_name}.txt')

        if os.path.exists(collect_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(collect_dir, f'{collect_file_name}_{timestamp}.txt')
        
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
                        terminate_info = {'success': True, 'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    else:
                        terminate_info = {'success': False, 'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
                elif action_text.startswith('recall location of '):
                    target_obj = action_text.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    print(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 1    
                    with open(collect_file_path, 'a') as file:
                        file.write(f'Act: {action_text}\n{obs_text}\n')
                else:
                    try:
                        obs = self.env.step(action_text)
                        obs_text = obs['text']
                        print(obs_text)
                        cur_step_id += 1
                        cur_decision_id += 1
                        with open(collect_file_path, 'a') as file:
                            file.write(f'Act: {action_text}\n{obs_text}\n')
                    except:
                        print("Error. Type again.")
                        pass 
            elif decision == 'e':
                cur_decision_id += 1
                control_flow = input('Control flow ("sequence" or "fallback" or "parallel"): ')
                subgoals = input('Subgoals (connect with ","): ')
                with open(collect_file_path, 'a') as file:
                    file.write(f'Expand:\n- control flow: {control_flow}\n- subgoals: {subgoals}\n')
                subgoals = subgoals.split(', ')
                
                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': nl_inst_info['task_type']}
                    agent_node = WahAgentNode(self.cfg, subgoal_info, self.depth + 2, None, self.env)
                    control_flow_node.add_child(agent_node)
                return self.children[0].collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)           
            else:
                print("Error. Type again.")
                pass
    
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
        
    def collect_llm(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id):
        import pdb
        message = self.make_message()
        nl_inst_info = self.content
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['message'] = message
        init_obs = self.env.get_init_obs()
        init_obs_text = init_obs['text']
        if trial_id == 0:
            self.llm_agent.reset(nl_inst_info, init_obs_text)
        elif trial_id > 0:
            pdb.set_trace()
        else:
            raise NotImplementedError()
        
        collect_file_name = f'{collect_file_base_name}_dec{str(cur_decision_id).zfill(3)}_depth{str(self.depth).zfill(2)}'
        collect_file_path = os.path.join(collect_dir, f'{collect_file_name}.txt')

        if os.path.exists(collect_file_path):
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            collect_file_path = os.path.join(collect_dir, f'{collect_file_name}_{timestamp}.txt')
        
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
            if cur_step_id > self.cfg.llm_agent.max_steps:
                terminate_info = {'success': False, 'terminate': 'max_step', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                with open(collect_file_path, 'a') as file:
                    file.write('Max steps')
                return terminate_info
            if cur_decision_id > self.cfg.llm_agent.max_decisions:
                terminate_info = {'success': False, 'terminate': 'max_decision', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                with open(collect_file_path, 'a') as file:
                    file.write('Max decisions')
                return terminate_info
            
            skill_set = self.env.get_possible_skill_set()

            try:
                next_step_info = self.llm_agent.plan_next_step(skill_set)
                next_step_class, next_step = next_step_info['next_step_class'], next_step_info['next_step']
            except Exception as error_message:
                terminate_info = {'success': False, 'terminate': 'plan_next_step_error', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                with open(collect_file_path, 'a') as file:
                    file.write(f"Plan Next Step Error: {error_message}")
                return terminate_info

            if next_step_class == 'Think':
                with open(collect_file_path, 'a') as file:
                    file.write(f'{next_step_class}: {next_step}\nOK.\n')
                cur_decision_id += 1
            elif next_step_class == 'Act':
                with open(collect_file_path, 'a') as file:
                    file.write(f'{next_step_class}: {next_step}')
                if next_step == 'done':
                    terminate_info = {'success': True, 'terminate': 'done', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
                elif next_step == 'failure':
                    terminate_info = {'success': False, 'terminate': 'failure', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
                elif next_step.startswith('recall location of '):
                    target_obj = next_step.split('recall location of ')[1]
                    obs_text = utils.recall_working_memory(self.env.working_memory, target_obj)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'\n{obs_text}\n')
                    self.llm_agent.add_obs(obs_text)
                    cur_step_id += 1
                    cur_decision_id += 1            
                else:
                    obs = self.env.step(next_step)
                    with open(collect_file_path, 'a') as file:
                        file.write(f'\n{obs["text"]}\n')
                    self.llm_agent.add_obs(obs['text'])
                    cur_step_id += 1
                    cur_decision_id += 1
            elif next_step_class == 'Expand':
                cur_decision_id += 1
                with open(collect_file_path, 'a') as file:
                    file.write(f'Expand:\n- control flow: {next_step["control_flow"]}\n- subgoals: {next_step["conditions"]}\n')

                control_flow = next_step['control_flow']
                subgoals = next_step['conditions'].split(', ')

                control_flow_node = ControlFlowNode(self.cfg, control_flow, self.depth + 1)
                self.add_child(control_flow_node)

                for idx, subgoal in enumerate(subgoals):
                    subgoal_info = {'nl_inst': subgoal, 'task_type': nl_inst_info['task_type']}
                    agent_node = WahAgentNode(self.cfg, subgoal_info, self.depth + 2, self.llm_agent, self.env)
                    control_flow_node.add_child(agent_node)

                return self.children[0].collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
            else:
                raise NotImplementedError()