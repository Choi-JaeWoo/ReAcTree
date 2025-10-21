import math
import os, json
import pprint
import random
import textwrap
import time, datetime
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from PIL import Image, ImageDraw, ImageFont

from alfred.data.preprocess import Dataset
from src.alfred.alfred_react import AlfredReact
from src.alfred.alfred_reactree import AlfredReactree
from src.alfred.alfred_env import ThorConnector
from src.alfred.alfred_llm_agent import AlfredLlmAgent
from src.alfred.utils import dotdict, load_task_json, save_vis_log
from tqdm import tqdm

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import pdb

font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
log = logging.getLogger(__name__)

class AlfredEvaluator():
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self):
        cfg = self.cfg
        log.info(OmegaConf.to_yaml(cfg))
        
        alfred_env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)
        alfred_llm_agent = AlfredLlmAgent(cfg)

        # LLM Planner
        if cfg.task_planner == 'react':
            tp = AlfredReact(cfg, alfred_llm_agent, alfred_env)
        elif cfg.task_planner =='reactree':
            tp = AlfredReactree(cfg, alfred_llm_agent, alfred_env)
        else:
            raise NotImplementedError()

        # prepare
        splits = cfg.alfred.splits
        args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}
        self.args_dict = args_dict
        
        with open(splits) as f:
            self.splits = json.load(f)
            pprint.pprint({k: len(v) for k, v in self.splits.items()})

        # preprocessing
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        if do_preprocessing:
            log.info("\nPreprocessing dataset... Do this once as required:")
            vocab = None
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(self.splits)

        # load tasks
        assert cfg.dataset.eval_set in self.splits.keys()
        files = []
        for e in self.splits[cfg.dataset.eval_set]:
            files.append(e)

        # select a subset of tasks
        if cfg.alfred.eval_portion_in_percent < 100:
            seed = cfg.alfred.random_seed_for_eval_subset
            random.seed(seed)  # set random seed for reproducibility
            n_sample = int(len(files) * cfg.alfred.eval_portion_in_percent / 100)
            files = random.sample(files, n_sample)
            random.seed(cfg.planner.random_seed)
        
        # run
        start = time.time()
        save_path = cfg.out_dir
        
        ########################
        results = []
        for i, task_d in enumerate(tqdm(files)):
            terminate_info = tp.run(task_d, args_dict, log)
            
            result = {'task': task_d['task'],
                      'repeat_idx': task_d['repeat_idx'],
                      'success': terminate_info['success'],
                    #   'terminate': terminate_info['terminate'],
                      'nl_inst': terminate_info['nl_inst']}
            
            task_name = task_d['task']
            task_type = task_name.split('/')[0]
            trial_num = task_name.split('/')[1]
            repeat_idx = task_d['repeat_idx']
            if result['success']:
                vis_log_name = f'success_{task_type}_{trial_num}_ann_{repeat_idx}'
            else:
                vis_log_name = f'failure_{task_type}_{trial_num}_ann_{repeat_idx}'
            save_vis_log(cfg, alfred_env.vis_log, vis_log_name, terminate_info['nl_inst'])
            results.append(result)
            
        ########################

        # print results
        log.info(results)
        n = len(results)
        n_success = 0
        for e in results:
            if e['success']:
                n_success += 1
        log.info(f'success rate: {n_success / n * 100:.2f} % ({n_success}/{n})')
        log.info(f'elapsed: {str(datetime.timedelta(seconds=(time.time() - start)))}')
        log.info('------------------------')
        log.info(OmegaConf.to_yaml(cfg))  # print cfg once again for easier configuration lookup
