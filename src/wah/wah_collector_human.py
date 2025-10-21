import json

from wah.wah_env import WahUnityEnv
from wah.wah_react import WahReact
from wah.wah_reactree import WahReactree


class WahCollectorHuman():
    def __init__(self, cfg):
        self.cfg = cfg

    def collect(self):
        cfg = self.cfg
        with open(cfg.dataset.wah_trainset, 'r') as json_file:
            train_set = json.load(json_file)
        wah_env = WahUnityEnv(cfg)
        if cfg.task_planner == 'react':
            tp = WahReact(cfg, None, wah_env)
        elif cfg.task_planner == 'reactree':
            tp = WahReactree(cfg, None, wah_env)
        
        while True:
            task_id = int(input('Type target task ID (0~249): '))
            task_d = train_set[task_id]
            task_d['task_id'] = task_id
            terminate_info = tp.collect(task_d)
            print(f'Trajectory collection for {task_id}-th task is finished.')