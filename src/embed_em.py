import hydra
import random
import numpy as np
import torch
import os

from wah.wah_embedder import WahEmbedder
from alfred.alfred_embedder import AlfredEmbedder

@hydra.main(version_base=None, config_path='../conf', config_name='config_wah_react')
def main(cfg):
    # set random seed
    random.seed(cfg.llm_agent.random_seed)
    torch.manual_seed(cfg.llm_agent.random_seed)
    np.random.seed(cfg.llm_agent.random_seed)
    
    if cfg.dataset_type == 'alfred':
        embedder = AlfredEmbedder(cfg)
    elif cfg.dataset_type == 'wah':
        embedder = WahEmbedder(cfg)
    else:
        raise NotImplementedError()

    embedder.extract_success_traj()
    embedder.embedding()


if __name__=='__main__':
    main()