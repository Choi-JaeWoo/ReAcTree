import sys
import json
import os
import pickle

sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from alfred.data.preprocess import Dataset
from src.alfred.alfred_env import ThorConnector
from src.alfred.alfred_react import AlfredReact
from src.alfred.alfred_reactree import AlfredReactree
from src.alfred.utils import dotdict

import pdb


class AlfredCollectorHuman():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def collect(self):
        cfg = self.cfg
        env = ThorConnector(cfg=cfg, x_display=cfg.alfred.x_display)

        if cfg.task_planner == 'react':
            tp = AlfredReact(cfg, None, env)
        elif cfg.task_planner =='reactree':
            tp = AlfredReactree(cfg, None, env)
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
        print("Total Number of tasks: ", len(files))
        self.train_set = files        

        # train set 21,023 tasks    
        print(f"Alfred Collector task selection mode : {cfg.dataset.task_selection_mode}")

        if cfg.dataset.task_selection_mode:
            if not os.path.exists('task_type_data.pkl'):
                task_type_data = self.parse_type(files)
                # save pickle 
                with open('task_type_data.pkl', 'wb') as f:
                    pickle.dump(task_type_data, f)
            else: 
                with open('task_type_data.pkl', 'rb') as f:
                    task_type_data = pickle.load(f)
            available_task_type=list(task_type_data.keys())

        while True:
            # task selection mode 
            if cfg.dataset.task_selection_mode: 
                print("Available task type ")
                for i, t_type in enumerate(available_task_type):
                    print(f'[{i}] ', t_type)
                task_type = int(input('\n> Select task type  : '))
                if task_type >= len(available_task_type):
                    print(f"Selected Number is out of bounds [0] ~ [{len(available_task_type)}]", task_type)
                    continue
                if available_task_type[task_type] not in available_task_type:
                    print("Invalid task type", available_task_type[task_type])
                    continue
                print("Selected task type : ", available_task_type[task_type])
                task_id = int(
                    input('\n> Type target task ID (0~{}): '.format(
                    len(task_type_data[available_task_type[task_type]])-1)))
                # task trajectory loading 
                task_d = task_type_data[available_task_type[task_type]][task_id]
                task_d['task_type']=available_task_type[task_type]
            # number selection mode 
            else:
                task_id = int(input('\n> Type target task ID (0~{}): '.format(
                    len(self.train_set))))
                task_d = self.train_set[task_id]
                task_d['task_type']=list(self.parse_type([task_d]).keys())[0]
            
            task_d['alfred_train_task_idx']=task_id
            task_d['total_train_num']=len(self.train_set)
            terminate_info = tp.collect(task_d, args_dict=args_dict)
            print(f'Trajectory collection for {task_id}-th task is finished.')
    
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
    
    