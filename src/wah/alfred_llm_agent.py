import pdb
import pickle
import numpy as np
from numpy.linalg import norm
import os
from llm_agent import LlmAgent
import src.alfred.utils as utils

def read_txt_file(file_path):
    with open(file_path) as file:
        txt_content = file.read()
    return txt_content

def cosine_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

class AlfredLlmAgent(LlmAgent):
    def load_prompt(self, nl_inst_info, init_obs_text):
        prompt_path = os.path.join(self.cfg.prompt.sys_prompt_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}.txt')
        system_prompt = read_txt_file(prompt_path)        
        nl_inst = nl_inst_info['nl_inst']

        ic_ex_files = self.ic_ex_select(nl_inst_info)
        ic_exs = [read_txt_file(ic_ex_file) for ic_ex_file in ic_ex_files]
        self.ic_ex_prompt = '\n'.join(ic_ex for ic_ex in ic_exs)
        if nl_inst_info['message'] == None:
            prompt = f'{system_prompt}\nSource domain:\n{self.ic_ex_prompt}\nTarget_domain:\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        elif nl_inst_info['message'] == '':
            prompt = f'{system_prompt}\nSource domain:\n{self.ic_ex_prompt}\nTarget_domain:\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        else:
            prompt = f'{system_prompt}\nSource domain:\n{self.ic_ex_prompt}\nTarget_domain:\n{nl_inst_info["message"]}\nYour task is to: {nl_inst}\n{init_obs_text}\n'
        return prompt
    
    def get_ic_ex_samples(self):
        return self.system_prompt, self.ic_ex_prompt
    
    def load_predefined_prompt(self, file_path):
        with open(file_path) as file:
            prompt = file.read()
        return prompt
    
    def ic_ex_select(self, nl_inst_info):
        ic_ex_select_type = self.cfg.llm_agent.ic_ex_select_type        
        if ic_ex_select_type == 'rag':
            ic_ex_encode_dir = os.path.join(self.cfg.prompt.ic_ex_root_dir, f'{self.cfg.task_planner}{"_wm" if self.cfg.llm_agent.working_memory else ""}', 'embed')
            
            sbert = self.sbert
            nl_inst = nl_inst_info['nl_inst']
            nl_inst_embedding = sbert.encode(nl_inst)

            ic_ex_encode_list = []
            for ic_ex_encode_name in os.listdir(ic_ex_encode_dir):
                if ic_ex_encode_name.endswith('.pkl'):
                    ic_ex_encode_path = os.path.join(ic_ex_encode_dir, ic_ex_encode_name)
        
                    with open(ic_ex_encode_path, 'rb') as file:
                        ic_ex_encoding = pickle.load(file)
                    
                    similarity = cosine_similarity(ic_ex_encoding['embedding'], nl_inst_embedding)
                    ic_ex_encoding['similarity'] = similarity
                    ic_ex_encode_list.append(ic_ex_encoding)
            
            sorted_ic_ex_encode_list = sorted(ic_ex_encode_list, key=lambda x: x['similarity'], reverse=True)

            ic_ex_files = []
            current_token_sum = 0
            for encoding in sorted_ic_ex_encode_list:
                token_count = encoding['token_count']
                if current_token_sum + token_count > 5000:
                    break
                ic_ex_files.append(encoding['text_traj_path'])
                current_token_sum += token_count
        return ic_ex_files
        
    def update_skill_set(self, obs):
        self.is_init = False
        nl_obs_partial_objs_info = obs['nl_obs_partial_objs_info']
        nl_obs_only_put_obs_info = obs['put_obj']

        if obs['init_obs']:
            self.is_init = True
            skill_set = ['done', 'failure']
        else:
            skill_set = []
            self.is_init = False

        if nl_obs_partial_objs_info is None:
            return self.init_skill_set
        else:
            for partial_obj_info in nl_obs_partial_objs_info:
                partial_obj_name = partial_obj_info.split(' ')[0]
                partial_obj = utils.ungroup_objects(partial_obj_info)
                for i in range(len(partial_obj)):
                    skill_set.append(f'go to {partial_obj[i]}')
                    if partial_obj_name in utils.ALFRED_PICK_OBJ:
                        skill_set.append(f'pick up {partial_obj[i]}')
                        if nl_obs_only_put_obs_info is not None:
                            skill_set.append(f'put down {nl_obs_only_put_obs_info}')    
                    if partial_obj_name in utils.AFLRED_OPEN_OBJ:
                        skill_set.append(f'open {partial_obj[i]}')
                        skill_set.append(f'close {partial_obj[i]}')
                    if partial_obj_name in utils.ALFRED_TOGGLE_OBJ:
                        skill_set.append(f'turn on {partial_obj[i]}')
                        skill_set.append(f'turn off {partial_obj[i]}')
                    if partial_obj_name in utils.ALFRED_SLICE_OBJ:
                        skill_set.append(f'slice {partial_obj[i]}')
            for objname in utils.ALFRED_PICK_OBJ:
                skill_set.append(f'recall location of {objname}')
            
            for objname in ['DeskLamp', 'FloorLamp']:
                skill_set.append(f'recall location of {objname}')
                        
            if obs['init_obs']:
                self.init_skill_set = skill_set
                return self.init_skill_set
            else:
                skill_set = self.init_skill_set + skill_set
                self.skill_set = list(set(skill_set))
                return self.skill_set