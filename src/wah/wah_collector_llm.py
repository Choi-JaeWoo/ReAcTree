import json

from wah.wah_env import WahUnityEnv
from wah.wah_react import WahReact
from wah.wah_reactree import WahReactree
from wah.wah_llm_agent import WahLlmAgent 

from tqdm import tqdm

class WahCollectorLLM():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def collect(self):
        cfg = self.cfg
        with open(cfg.dataset.wah_trainset, 'r') as json_file:
            train_set = json.load(json_file)
        wah_env = WahUnityEnv(cfg)
        wah_llm_agent = WahLlmAgent(cfg)
        
        if cfg.task_planner == 'react':
            tp = WahReact(cfg, wah_llm_agent, wah_env)
        elif cfg.task_planner == 'reactree':
            tp = WahReactree(cfg, wah_llm_agent, wah_env)
        else:
            raise NotImplementedError()
        for trial_id in range(1):
            for task_id, task_d in tqdm(enumerate(train_set), total=len(train_set)):
                terminate_info = tp.collect_llm(task_d, task_id, trial_id)
                print(f'Trajectory collection LLM #{trial_id} for {task_id}-th task is finished.')