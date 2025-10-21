import re
import os
import json
import shutil
import pickle
import pdb

from wah.wah_env import WahUnityEnv
from wah.wah_utils import check_goal_condition
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class WahEmbedder():
    def __init__(self, cfg):
        self.cfg = cfg
    
    def extract_success_traj(self):
        cfg = self.cfg
        text_trajectory_dir = os.path.join(cfg.dataset.embedding_root_dir, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}')

        filenames = os.listdir(text_trajectory_dir)

        dict_taskid2filename = taskid2filename(filenames)
        wah_env = WahUnityEnv(cfg)

        with open(cfg.dataset.wah_trainset, 'r') as json_file:
            train_set = json.load(json_file)

        for task_id in list(dict_taskid2filename.keys()):
            sorted_filenames = sorted(dict_taskid2filename[task_id], key=extract_task_id_and_suffix)
            trajectory_path_list = [os.path.join(text_trajectory_dir, filename) for filename in sorted_filenames]
            task_d = train_set[task_id]
            
            if sorted_filenames == []:
                continue

            goal_success_rate, subgoal_success_rate, task_id = check_success(trajectory_path_list, wah_env, task_d, task_id)
            
            if goal_success_rate == 1:
                em_dir = os.path.join(cfg.dataset.em_root_dir, f'{cfg.task_planner}{"_wm" if cfg.llm_agent.working_memory else ""}','text_traj', f'{task_id}'.zfill(3))
                os.makedirs(em_dir, exist_ok=True)
                for trajectory_path in trajectory_path_list:
                    filename = os.path.basename(trajectory_path)
                    dest_path = os.path.join(em_dir, filename)
                    if not os.path.exists(dest_path):
                        shutil.copy(trajectory_path, dest_path)
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

        for task_id in os.listdir(em_text_traj_dir):
            success_em_text_traj_dir = os.path.join(em_text_traj_dir, task_id)
            for file_name in os.listdir(success_em_text_traj_dir):
                text_trajectory_path = os.path.join(success_em_text_traj_dir, file_name)

                with open(text_trajectory_path) as file:
                    text_traj = file.read()

                parsing_result = parsing_text_traj(text_traj, cfg.llm_agent.agent_type)
                
                task_goal_text = parsing_result['task_goal']
                goal_embedding = sbert.encode(task_goal_text.split('Your task is to: ')[1])
                
                tokens = tokenizer(text_traj)['input_ids']
                token_count = len(tokens)
                
                state_embedding = sbert.encode(parsing_result['initial_state'])

                embed_name = file_name.replace('.txt', '.pkl')
                embedding = {'text_trajectory': text_traj,
                    'embedding': goal_embedding,
                    'text_traj_path': text_trajectory_path,
                    'token_count': token_count,
                    'state_embedding': state_embedding}
                
                em_embed_path = os.path.join(em_embed_dir, embed_name)
                with open(em_embed_path, 'wb') as pickle_file:
                    pickle.dump(embedding, pickle_file)

                print(token_count)




def extract_task_id_and_suffix(filename):
    match = re.search(r'(\d+)(.*)', filename)
    if match:
        return int(match.group(1)), match.group(2)
    return None

def taskid2filename(filenames):
    dict_taskid2filename = {task_id: [] for task_id in range(250)}
    for filename in filenames:
        match = re.search(r'(\d+)(.*)', filename)
        if match:
            dict_taskid2filename[int(match.group(1))].append(filename)
        else:
            pdb.set_trace()
    return dict_taskid2filename

def extract_action_seq(trajectory_path):
    action_seq = []
    with open(trajectory_path) as file:
        txt_content = file.read()
        for line in txt_content.splitlines():
            if line.startswith('Act: '):
                action_seq.append(line)
    return action_seq

def evaluate_task_completion(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim):
    subgoal_success_rate = check_goal_condition(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim)
    if subgoal_success_rate == 1:
        goal_success_rate = 1
    else:
        goal_success_rate = 0
    return goal_success_rate, subgoal_success_rate

def check_success(trajectory_path_list, env, task_d, task_id):
    env.reset(task_d)
    action_seq = []
    for trajectory_path in trajectory_path_list:
        action_seq += extract_action_seq(trajectory_path) 

    for action in action_seq:
        nl_step = action.split('Act: ')[1]
        if nl_step in ['done', 'failure']:
            pass
        elif 'recall location of ' in nl_step:
            pass
        else:
            env.step(nl_step)
    
    task_goal, graph = task_d['task_goal'], env.get_graph()
    name_id_dict_sim2nl, name_id_dict_nl2sim = env.name_id_dict_sim2nl, env.name_id_dict_nl2sim
    goal_success_rate, subgoal_success_rate = evaluate_task_completion(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim)
    print(f'Task ID: {task_id} --> {goal_success_rate}% & {subgoal_success_rate}%')
    return goal_success_rate, subgoal_success_rate, task_id


def parsing_text_traj(text_traj, agent_type):
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


