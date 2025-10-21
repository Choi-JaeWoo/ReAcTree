import os
from llm_agent import LlmAgent
import wah.wah_utils as utils
import pickle
import pdb
import torch

import numpy as np
from numpy.linalg import norm
from transformers import AutoTokenizer

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
        

def read_txt_file(file_path):
    with open(file_path) as file:
        txt_content = file.read()
    return txt_content


class WahLlmAgent(LlmAgent):
    def load_prompt(self, nl_inst_info, init_obs_text):
        prompt_path = os.path.join(self.cfg.prompt.sys_prompt_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}.txt')
        system_prompt = read_txt_file(prompt_path)
        nl_inst = nl_inst_info['nl_inst']
        nl_inst_info['init_obs'] = init_obs_text
        ic_ex_files = self.ic_ex_select(nl_inst_info)
        ic_exs = [read_txt_file(ic_ex_file) for ic_ex_file in ic_ex_files]
        ic_ex_prompt = '\n'.join(ic_ex for ic_ex in ic_exs)
        if nl_inst_info['message'] == None:
            prompt = f'{system_prompt}\nSource domain:\n{ic_ex_prompt}\nTarget_domain:\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        elif nl_inst_info['message'] == '':
            prompt = f'{system_prompt}\nSource domain:\n{ic_ex_prompt}\nTarget_domain:\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        else:
            prompt = f'{system_prompt}\nSource domain:\n{ic_ex_prompt}\nTarget_domain:\n{nl_inst_info["message"]}\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        
        return prompt

    def ic_ex_select(self, nl_inst_info):
        ic_ex_select_type = self.cfg.llm_agent.ic_ex_select_type
        if ic_ex_select_type == 'rag':
            ic_ex_embedding_dir = os.path.join(self.cfg.prompt.ic_ex_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}', 'embed')
            sbert = self.sbert
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = sbert.encode(nl_inst)
            ic_ex_embedding_list = []
            for ic_ex_embedding_name in os.listdir(ic_ex_embedding_dir):
                if ic_ex_embedding_name.endswith('.pkl'):
                    ic_ex_embedding_path = os.path.join(ic_ex_embedding_dir, ic_ex_embedding_name)

                    with open(ic_ex_embedding_path, 'rb') as file:
                        ic_ex_embedding = pickle.load(file)
                    
                    similarity = cosine_similarity(ic_ex_embedding['embedding'], nl_inst_embedding)

                    ic_ex_embedding['similarity'] = similarity
                    ic_ex_embedding_list.append(ic_ex_embedding)
            sorted_ic_ex_embedding_list = sorted(ic_ex_embedding_list, key=lambda x: x['similarity'], reverse=True)
            sorted_ic_ex_embedding_list = utils.sort_with_same_similarity(sorted_ic_ex_embedding_list)
            ic_ex_files = []
            current_token_sum = 0
            for embedding in sorted_ic_ex_embedding_list:
                token_count = embedding['token_count']
                if current_token_sum + token_count > 5000:
                    break
                ic_ex_files.append(embedding['text_traj_path'])
                current_token_sum += token_count
                
        elif 'fix_' in ic_ex_select_type:
            ic_ex_root_dir = os.path.join(self.cfg.prompt.ic_ex_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}', 'text_traj')
            if ic_ex_select_type == 'fix_1':
                ic_ex_dir = os.path.join(ic_ex_root_dir, '001')
            elif ic_ex_select_type == 'fix_21':
                ic_ex_dir = os.path.join(ic_ex_root_dir, '021')
            elif ic_ex_select_type == 'fix_41':
                ic_ex_dir = os.path.join(ic_ex_root_dir, '041')
            elif ic_ex_select_type == 'fix_61':
                ic_ex_dir = os.path.join(ic_ex_root_dir, '061')
            elif ic_ex_select_type == 'fix_151':
                ic_ex_dir = os.path.join(ic_ex_root_dir, '151')
            
            ic_ex_file_names = os.listdir(ic_ex_dir)
            ic_ex_file_names.sort()
            ic_ex_files = [os.path.join(ic_ex_dir, file_names) for file_names in ic_ex_file_names]
               
        elif ic_ex_select_type == 'rag_task':
            ic_ex_embedding_dir = os.path.join(self.cfg.prompt.ic_ex_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}', 'embed')
            sbert = self.sbert
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = sbert.encode(nl_inst)
            
            ic_ex_embedding_list = []
            task_level_goal_list = [file_name for file_name in os.listdir(ic_ex_embedding_dir) if 'depth01' in file_name]

            for ic_ex_embedding_name in task_level_goal_list:
                if ic_ex_embedding_name.endswith('.pkl'):
                    ic_ex_embedding_path = os.path.join(ic_ex_embedding_dir, ic_ex_embedding_name)
                    with open(ic_ex_embedding_path, 'rb') as file:
                        ic_ex_embedding = pickle.load(file)
                    
                    similarity = cosine_similarity(ic_ex_embedding['embedding'], nl_inst_embedding)

                    ic_ex_embedding['similarity'] = similarity
                    ic_ex_embedding_list.append(ic_ex_embedding)
            sorted_ic_ex_embedding_list = sorted(ic_ex_embedding_list, key=lambda x: x['similarity'], reverse=True)
            
            task_level_example_trajectory_dirs = [os.path.dirname(ic_emb['text_traj_path']) for ic_emb in sorted_ic_ex_embedding_list]
            
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')
            current_token_sum = 0
            ic_ex_files = []
            for task_traj_dir in task_level_example_trajectory_dirs:
                task_files = os.listdir(task_traj_dir)
                task_files.sort()
                task_token = 0
                for task_file in task_files:
                    file_path = os.path.join(task_traj_dir, task_file)
                    with open(file_path) as file:
                        text_traj = file.read()
                    tokens = tokenizer(text_traj)['input_ids']
                    token_count = len(tokens)
                    task_token += token_count
                if current_token_sum + task_token > 5000:
                    break
                ic_ex_files += [os.path.join(task_traj_dir, task_file) for task_file in task_files]

        elif ic_ex_select_type == 'obs_rag':
            ic_ex_embedding_dir = os.path.join(self.cfg.prompt.ic_ex_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}', 'embed')
            sbert = self.sbert
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = sbert.encode(nl_inst)
            init_obs_embedding = sbert.encode(nl_inst_info['init_obs'])
            ic_ex_encode_list = []
            for ic_ex_encode_name in os.listdir(ic_ex_embedding_dir):
                if ic_ex_encode_name.endswith('.pkl'):
                    ic_ex_encode_path = os.path.join(ic_ex_embedding_dir, ic_ex_encode_name)

                    with open(ic_ex_encode_path, 'rb') as file:
                        ic_ex_encoding = pickle.load(file)
                    
                    similarity_1 = cosine_similarity(ic_ex_encoding['embedding'], nl_inst_embedding)
                    similarity_2 = cosine_similarity(ic_ex_encoding['state_embedding'], init_obs_embedding)
                    ic_ex_encoding['similarity'] = similarity_1*0.9 + similarity_2*0.1
                    ic_ex_encode_list.append(ic_ex_encoding)
            
            sorted_ic_ex_embedding_list = sorted(ic_ex_encode_list, key=lambda x: x['similarity'], reverse=True)
            sorted_ic_ex_embedding_list = utils.sort_with_same_similarity(sorted_ic_ex_embedding_list)
            ic_ex_files = []
            current_token_sum = 0
            for embedding in sorted_ic_ex_embedding_list:
                token_count = embedding['token_count']
                if current_token_sum + token_count > 5000:
                    break
                ic_ex_files.append(embedding['text_traj_path'])
                current_token_sum += token_count
        
        elif ic_ex_select_type == 'nothing':
            ic_ex_files = []

        elif ic_ex_select_type == 'rerank':
            ic_ex_embedding_dir = os.path.join(self.cfg.prompt.ic_ex_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}', 'embed')
            sbert = self.sbert
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = sbert.encode(nl_inst)
            ic_ex_embedding_list = []
            for ic_ex_embedding_name in os.listdir(ic_ex_embedding_dir):
                if ic_ex_embedding_name.endswith('.pkl'):
                    ic_ex_embedding_path = os.path.join(ic_ex_embedding_dir, ic_ex_embedding_name)

                    with open(ic_ex_embedding_path, 'rb') as file:
                        ic_ex_embedding = pickle.load(file)
                    
                    similarity = cosine_similarity(ic_ex_embedding['embedding'], nl_inst_embedding)

                    ic_ex_embedding['similarity'] = similarity
                    ic_ex_embedding_list.append(ic_ex_embedding)
            sorted_ic_ex_embedding_list = sorted(ic_ex_embedding_list, key=lambda x: x['similarity'], reverse=True)
            ranking_list = sorted_ic_ex_embedding_list[:30]

            rerank_tokenizer = self.rerank_tokenizer
            rerank_model = self.rerank_model
            rerank_model.eval()
            pairs = [[nl_inst, ic_ex['text_trajectory']] for ic_ex in ranking_list]
            with torch.no_grad():
                inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()
                print(scores)

            for ic_ex, score in zip(ranking_list, scores):
                ic_ex['similarity'] = score.item()
            sorted_ranking_list = sorted(ranking_list, key=lambda x: x['similarity'], reverse=True)
            
            ic_ex_files = []
            current_token_sum = 0
            for embedding in sorted_ranking_list:
                token_count = embedding['token_count']
                if current_token_sum + token_count > 5000:
                    break
                ic_ex_files.append(embedding['text_traj_path'])
                current_token_sum += token_count

        return ic_ex_files