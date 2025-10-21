import hydra
import random
import numpy as np
import torch
import os

from wah.wah_collector_human import WahCollectorHuman
from alfred.alfred_collector_human import AlfredCollectorHuman

@hydra.main(version_base=None, config_path='../conf', config_name='config_wah_react')
def main(cfg):
    # set random seed
    random.seed(cfg.llm_agent.random_seed)
    torch.manual_seed(cfg.llm_agent.random_seed)
    np.random.seed(cfg.llm_agent.random_seed)
    
    if cfg.dataset_type == 'alfred':
        cfg.dataset.collect_dir = os.path.join(cfg.dataset.collect_ex_root_dir, cfg.exp_type, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}')
        if cfg.exp_type == 'collect_human':
            collector = AlfredCollectorHuman(cfg)
        else:
            raise NotImplementedError()
    elif cfg.dataset_type == 'wah':
        cfg.dataset.collect_dir = os.path.join(cfg.dataset.collect_ex_root_dir, cfg.exp_type, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}')
        if cfg.exp_type == 'collect_human':
            collector = WahCollectorHuman(cfg)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    collector.collect()


if __name__=='__main__':
    main()