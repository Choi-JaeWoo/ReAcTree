class Reactree():
    def __init__(self, cfg, llm_agent, env):
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.max_steps = cfg.llm_agent.max_steps
        self.max_decisions = cfg.llm_agent.max_decisions
        self.cur_step_id = 1
        self.cur_decision_id = 1
    def run(self, task_d):
        raise NotImplementedError()
    def collect(self):
        raise NotImplementedError()

class TreeNode:
    def __init__(self, cfg, content, depth):
        self.cfg = cfg
        self.content = content
        self.depth = depth
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)
    
    def run(self, cur_step_id, cur_decision_id, log):
        raise NotImplementedError()

class ControlFlowNode(TreeNode):
    def run(self, cur_step_id, cur_decision_id, log, traj_d=None):
        if self.depth > self.cfg.llm_agent.max_depth:
            terminate_info = {'success': False, 'terminate': 'max_depth', 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            log.info('Max depth')
            return terminate_info

        if self.content == 'sequence':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.run()
                    terminate_info = child.run(cur_step_id, cur_decision_id, log)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.run()
                    terminate_info = child.run(traj_d, cur_step_id, cur_decision_id, log)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'fallback':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.run()
                    terminate_info = child.run(cur_step_id, cur_decision_id, log)
                else:
                    terminate_info = child.run(traj_d, cur_step_id, cur_decision_id, log)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if terminate_info['success']:
                    terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.run()
                    terminate_info = child.run(cur_step_id, cur_decision_id, log)
                else:
                    terminate_info = child.run(traj_d, cur_step_id, cur_decision_id, log)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            terminate_info = {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        else:
            raise NotImplementedError()
    
    def collect(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, traj_d=None):
        if self.content == 'sequence':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect()
                    terminate_info = child.collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect()
                    terminate_info = child.collect(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'fallback':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect()
                    terminate_info = child.collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect()
                    terminate_info = child.collect(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if terminate_info['success']:
                    terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect()
                    terminate_info = child.collect(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect()
                    terminate_info = child.collect(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            terminate_info = {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        else:
            raise NotImplementedError()
        
    def collect_llm(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id, traj_d=None):
        if self.content == 'sequence':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect_llm()
                    terminate_info = child.collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect_llm()
                    terminate_info = child.collect_llm(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'fallback':
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect_llm()
                    terminate_info = child.collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect_llm()
                    terminate_info = child.collect_llm(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if terminate_info['success']:
                    terminate_info = {'success': True, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
                    return terminate_info
            terminate_info = {'success': False, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        elif self.content == 'parallel':
            is_success = True
            for child in self.children:
                if traj_d is None: # for wah --> src.wah.wah_reactree.AgentNode.collect_llm()
                    terminate_info = child.collect_llm(cur_step_id, cur_decision_id, collect_dir, collect_file_base_name, trial_id)
                else: # for alfred --> src.alfred.alfred_reactree.AlfredAgentNode.collect_llm()
                    terminate_info = child.collect_llm(traj_d, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name)
                cur_step_id, cur_decision_id = terminate_info['step_id'], terminate_info['decision_id']
                if not terminate_info['success']:
                    is_success = False
            terminate_info = {'success': is_success, 'step_id': cur_step_id, 'decision_id': cur_decision_id}
            return terminate_info
        else:
            raise NotImplementedError()

    def make_message(self):
        return None

class AgentNode(TreeNode):
    def __init__(self, cfg, content, depth, llm_agent, env):
        super().__init__(cfg, content, depth)
        self.llm_agent = llm_agent
        self.env = env
    def run(self, cur_step_id, cur_decision_id, log):
        raise NotImplementedError()
    def collect(self, cur_step_id, cur_decision_id, collect_dir, collect_file_base_name):
        raise NotImplementedError()
