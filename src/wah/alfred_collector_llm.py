import sys
import json
import os
import pickle
import random
from tqdm import tqdm
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from alfred.data.preprocess import Dataset
from src.alfred.alfred_env import ThorConnector
from src.alfred.alfred_react import AlfredReact
from src.alfred.alfred_reactree import AlfredReactree
from src.alfred.utils import dotdict
from src.alfred.alfred_llm_agent import AlfredLlmAgent

import pdb


class AlfredCollectorLLM():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def collect(self):
        cfg = self.cfg
        alfred_env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)
        alfred_llm_agent = AlfredLlmAgent(cfg)

        if cfg.task_planner == 'react':
            tp = AlfredReact(cfg, alfred_llm_agent, alfred_env)
        elif cfg.task_planner =='reactree':
            tp = AlfredReactree(cfg, alfred_llm_agent, alfred_env)
        
        # load train set 
        splits = 'alfred/data/splits/oct21.json'
        args_dict = {'data': 'alfred/data/json_2.1.0', 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': 'alfred/models/config/rewards.json', 'max_steps': 1000}
        
        self.args_dict = args_dict
        with open(splits) as f:
            splits = json.load(f)

        # preprocessing
        number_of_dirs = len(list(os.listdir(args_dict['data'])))
        do_preprocessing = number_of_dirs < 50  # one-time process
        if do_preprocessing:
            vocab = None
            dataset = Dataset(dotdict(args_dict), vocab)
            dataset.preprocess_splits(splits)

        # load tasks
        assert cfg.dataset.train_set in splits.keys()
        files = []
        
        for e in splits[cfg.dataset.train_set]:
            files.append(e)
        
        # select a subset of tasks
        if cfg.alfred.eval_portion_in_percent < 100:
            seed = cfg.alfred.random_seed_for_eval_subset
            random.seed(seed)  # set random seed for reproducibility
            n_sample = int(len(files) * cfg.alfred.eval_portion_in_percent / 100)
            files = random.sample(files, n_sample)
            random.seed(cfg.planner.random_seed)
        
        if cfg.alfred.diverse_task:
            tgt_tasks = ['look_at_obj_in_light', 'pick_and_place_simple', 'pick_and_place_with_movable_recep', 'pick_clean_then_place_in_recep', 'pick_cool_then_place_in_recep', 'pick_heat_then_place_in_recep', 'pick_two_obj_and_place']
            collect_results = dict()
            for t_task in tgt_tasks:
                collect_results[t_task] = []
            os.makedirs(cfg.dataset.collect_dir, exist_ok=True)
            result_log_path = os.path.join(cfg.dataset.collect_dir, 'result.log')
            smp_per_task = cfg.alfred.reflection_smp_per_task
            
                
            ### TODO: diverse task collection
            success_task_names = []
            for i, task_d in enumerate(tqdm(files)):
                print("-------------[Collection Current Status]-------------")
                for t_task in tgt_tasks:
                    print(f"Task [{t_task}] [{round(100*(len(collect_results[t_task])/smp_per_task),2)} percent done ({len(collect_results[t_task])}/{smp_per_task})]")
                print("------------------------------------------------------")
                with open(result_log_path, 'a') as result_log:
                    result_log.write(f"{task_d} - ")
                
                cur_task_type = list(self.parse_type([task_d]).keys())[0]
                
                if not cur_task_type in tgt_tasks:
                    print(f"Task [{cur_task_type}] is not in target tasks. Skip this task.")
                    with open(result_log_path, 'a') as result_log:
                        result_log.write(f"skip\n")
                    continue
                if len(collect_results[cur_task_type]) >= smp_per_task:
                    print(f"Task [{cur_task_type}] is already done. Skip this task.")
                    with open(result_log_path, 'a') as result_log:
                        result_log.write(f"skip\n")
                    continue
                if task_d["task"].split('/')[-1] in success_task_names:
                    print(f"Trial is already success. Skip this task.")
                    with open(result_log_path, 'a') as result_log:
                        result_log.write(f"skip\n")
                    continue
                
                terminate_info = tp.collect_llm(task_d, args_dict)
                
                success = terminate_info['success']
                if success:
                    success_task_names.append(task_d["task"].split('/')[-1])
                    collect_results[cur_task_type].append(task_d["task"].split('/')[-1])
                    with open(result_log_path, 'a') as result_log:
                        result_log.write(f"success\n")
                else:
                    with open(result_log_path, 'a') as result_log:
                        result_log.write(f"failure\n")
        else:
            raise NotImplementedError()
        
    def parse_type(self, files):
        from tqdm import tqdm
        def load_json(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            return data
        
        task_type_dict=dict()
        print("Parsing task types...")
        for file in tqdm(files):
            task_root = file['task']
            repeat_idx= file['repeat_idx']
            json_path = os.path.join(
                self.args_dict['data'], 
                task_root, 'pp', 
                'ann_{}.json'.format(repeat_idx)
            )
            traj_d = load_json(json_path)
            task_type = traj_d['task_type']
            if not task_type in task_type_dict.keys():
                task_type_dict[task_type]=[]
            task_type_dict[task_type].append(file)
        return task_type_dict

    