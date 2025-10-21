import hydra
import random
import numpy as np
import torch

from wah.wah_evaluator import WahEvaluator
from alfred.alfred_evaluator import AlfredEvaluator

@hydra.main(version_base=None, config_path='../conf', config_name='config_wah_react')
def main(cfg):
    # set random seed
    random.seed(cfg.llm_agent.random_seed)
    torch.manual_seed(cfg.llm_agent.random_seed)
    np.random.seed(cfg.llm_agent.random_seed)

    if cfg.dataset_type == 'alfred':
        evaluator = AlfredEvaluator(cfg)
    elif cfg.dataset_type == 'wah':
        evaluator = WahEvaluator(cfg)
    else:
        raise NotImplementedError()
    evaluator.evaluate()


if __name__=='__main__':
    main()