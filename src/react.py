class React():
    def __init__(self, cfg, llm_agent, env):
        self.cfg = cfg
        self.llm_agent = llm_agent
        self.env = env
        self.max_steps = cfg.llm_agent.max_steps
        self.max_decisions = cfg.llm_agent.max_decisions
        self.cur_step_id = 1
        self.cur_decision_id = 1
    def run(self, task_d, log):
        raise NotImplementedError()
    def collect(self, task_d):
        raise NotImplementedError()