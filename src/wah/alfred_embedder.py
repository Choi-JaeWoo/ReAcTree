import sys
import json
import os
import pickle
import shutil
import re
import ast

from collections import defaultdict
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from alfred.data.preprocess import Dataset
from src.alfred.alfred_env import ThorConnector
from src.alfred.alfred_react import AlfredReact
from src.alfred.alfred_reactree import AlfredReactree
from src.alfred.utils import dotdict, load_task_json

import pdb

class AlfredEmbedder():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def extract_success_traj(self):
        cfg = self.cfg

        text_trajectory_dir = os.path.join(cfg.dataset.embedding_root_dir, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}')
        text_traj_names = [name for name in os.listdir(text_trajectory_dir) if name.endswith('.txt')]

        em_dir = os.path.join(cfg.dataset.em_root_dir, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}','text_traj')
        os.makedirs(em_dir, exist_ok=True)
        if cfg.dataset.check_success:
            result_log_path = os.path.join(text_trajectory_dir, 'result.log')
            success_entries = extract_success_dicts(result_log_path)
            for success_entry in success_entries:
                task_type, trial = success_entry['task'].split('/')[0], success_entry['task'].split('/')[1]
                repeat_idx = success_entry['repeat_idx']
                base_name = f'{task_type}_{trial}_ann_{repeat_idx}'
                matching_trajs = [name for name in text_traj_names if base_name in name]
                for source_traj in matching_trajs:
                    source_path = os.path.join(text_trajectory_dir, source_traj)
                    dest_path = os.path.join(em_dir, source_traj)
                    if not os.path.exists(dest_path):
                        shutil.copy(source_path, dest_path)
                    else:
                        print('File already exists. Skipping copy.')
        else:
            for text_traj_name in text_traj_names:
                file_path = os.path.join(text_trajectory_dir, text_traj_name)
                dest_path = os.path.join(em_dir, text_traj_name)
                if not os.path.exists(dest_path):
                    shutil.copy(file_path, dest_path)
                else:
                    print('File already exists. Skipping copy.')

    def embedding(self):
        cfg = self.cfg
        sbert = SentenceTransformer('all-MiniLM-L6-v2')
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
        
        em_dir = os.path.join(cfg.dataset.em_root_dir, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}')
        em_text_traj_dir = os.path.join(em_dir, 'text_traj')
        em_embed_dir = os.path.join(em_dir, 'embed')
        
        os.makedirs(em_embed_dir, exist_ok=True)
        
        for traj_file in os.listdir(em_text_traj_dir):
            text_trajectory_path = os.path.join(em_text_traj_dir, traj_file)

            with open(text_trajectory_path) as file:
                text_traj = file.read()
            parsing_result = parsing_text_traj(text_traj)

            task_goal_text = parsing_result['task_goal']
            goal_embedding = sbert.encode(task_goal_text.split('Your task is to: ')[1])

            tokens = tokenizer(text_traj)['input_ids']
            token_count = len(tokens)
            state_embedding = sbert.encode(parsing_result['initial_state'])

            embed_name = traj_file.replace('.txt', '.pkl')
            embedding = {'text_trajectory': text_traj,
                'embedding': goal_embedding,
                'text_traj_path': text_trajectory_path,
                'token_count': token_count,
                'state_embedding': state_embedding}
            
            em_embed_path = os.path.join(em_embed_dir, embed_name)
            with open(em_embed_path, 'wb') as pickle_file:
                pickle.dump(embedding, pickle_file)

            print(token_count)


            
def extract_success_dicts(result_log_path):
    success_entries = []
    with open(result_log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith('- True'):
                dict_str = line.rsplit(' - ', 1)[0]
                try:
                    entry = ast.literal_eval(dict_str)
                    success_entries.append(entry)
                except (ValueError, SyntaxError) as e:
                    print(f"Failed to parse line: {line}\nError: {e}")
    return success_entries


def parsing_text_traj(text_traj):
    
    pattern = re.search(
        r'(?:(Your primary goal is to:.*?)\n(To achieve this,.*?)\n(Your task is to:.*?)\n(You are in the house,.*?)\n)|(?:(Your task is to:.*?)\n(You are in the house,.*?)\n)',
        text_traj, re.DOTALL
    )
    
    if pattern.group(1):
        parsing_result = {
            'primary_goal': pattern.group(1).strip(),
            'sibling_goals': pattern.group(2).strip(),
            'task_goal': pattern.group(3).strip(),
            'initial_state': pattern.group(4).strip()
        }
    else:
        parsing_result = {
            'task_goal': pattern.group(5).strip(),
            'initial_state': pattern.group(6).strip()
        }
    return parsing_result