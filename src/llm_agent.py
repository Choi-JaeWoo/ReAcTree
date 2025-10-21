import guidance
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LlmAgent():
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.llm_agent.model_name
        self.agent_type = cfg.llm_agent.agent_type
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        if cfg.llm_agent.ic_ex_select_type == 'rerank':
            self.rerank_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')

        if self.model_name in ['gpt-3.5-turbo-instruct', 'gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
            self.llm = guidance.models.OpenAI(self.model_name, api_key=cfg.llm_agent.openai_api_key)
        else:
            model_args = {'trust_remote_code': True, 'torch_dtype': torch.float16}
            if cfg.llm_agent.use_accelerate_device_map:
                model_args['device_map'] = "auto"
                if cfg.llm_agent.load_in_8bit:
                    model_args['load_in_8bit'] = True
            model_args['use_auth_token'] = cfg.llm_agent.hf_auth_token
            self.llm = guidance.models.Transformers(self.model_name, echo=False, **model_args)

    def reset(self, nl_inst_info, init_obs):
        self.prompt = self.load_prompt(nl_inst_info, init_obs)
        if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
            self.llm = guidance.models.OpenAI(self.model_name, api_key=self.cfg.llm_agent.openai_api_key)
            self.llm.reset()
            with guidance.user():
                self.llm += self.prompt
        else:
            self.llm.reset()
            self.llm += self.prompt

    def plan_next_step(self, skill_set):
        if self.cfg.llm_agent.agent_type == 'react':
            next_step_info = self.react_plan_next_step(skill_set)
        elif self.cfg.llm_agent.agent_type == 'reactexpand':
            next_step_info = self.reactexpand_plan_next_step(skill_set)
        else:
            raise NotImplementedError()
        return next_step_info

    def react_plan_next_step(self, skill_set):
        ### TODO: OpenAI API
        if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
            try:
                with guidance.assistant():
                    self.llm += guidance.select(['Act: ', 'Think: '], name='choice') # 잘 나옴 
                    if self.llm['choice'] == 'Think: ':
                        self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\nOK.\n'
                        next_step_class, next_step = 'Think', self.llm['reasoning']
                    else:
                        self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
                        next_step_class, next_step = 'Act', self.llm['nl_skill']
                next_step_info = {'next_step_class': next_step_class, 'next_step': next_step}
                return next_step_info
            except Exception as e:
                with guidance.user():
                    self.llm += 'You should only output sentences that begin with Think or Act.\nIf you output Act, you should use one of actions of this list: [go to, pick up, put down, open, close, turn on, recall location of, done, failure].\n'
                next_step_info = {'next_step_class': 'Error', 'next_step': e}
                return next_step_info
        ### HuggingFace LLM
        else:
            self.llm += guidance.select(['Act: ', 'Think: '], name='choice')
            if self.llm['choice'] == 'Think: ':
                self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\nOK.\n'
                next_step_info = {'next_step_class': 'Think', 'next_step': self.llm['reasoning']}
            else:
                self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
                next_step_info = {'next_step_class': 'Act', 'next_step': self.llm['nl_skill']}
            return next_step_info
    
    def reactexpand_plan_next_step(self, skill_set):
        ### TODO: OpenAI API
        if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
            try:
                with guidance.assistant():
                    self.llm += guidance.select(['Act: ', 'Think: ', 'Expand:\n'], name='choice') # 잘 나옴 
                    if self.llm['choice'] == 'Think: ':
                        self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\nOK.\n'
                        next_step_info = {'next_step_class': 'Think', 'next_step': self.llm['reasoning']}
                    elif self.llm['choice'] == 'Act: ':
                        self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
                        next_step_info = {'next_step_class': 'Act', 'next_step': self.llm['nl_skill']}
                    elif self.llm['choice'] == 'Expand:\n':
                        self.llm += '- control flow: ' + guidance.select(['sequence', 'fallback', 'parallel'], name='control_flow') + '\n- subgoals: ' + guidance.gen(stop='\n', name='conditions', max_tokens=200, temperature=0) + '\nOK.\n'
                        next_step_info = {'next_step_class': 'Expand', 'next_step': {'control_flow': self.llm['control_flow'], 'conditions': self.llm['conditions']}}
                return next_step_info
            except Exception as e:
                with guidance.user():
                    self.llm += 'You should only output sentences that begin with Think, Act, or Expand.\nIf you output Act, you should use one of actions of this list: [go to, pick up, put down, open, close, turn on, recall location of, done, failure].\n'
                next_step_info = {'next_step_class': 'Error', 'next_step': e}
                return next_step_info
        ### HuggingFace LLM
        else:
            self.llm += guidance.select(['Act: ', 'Think: ', 'Expand:\n'], name='choice')
            if self.llm['choice'] == 'Think: ':
                self.llm += guidance.gen(stop='\n', name='reasoning', max_tokens=200, temperature=0) + '\nOK.\n'
                next_step_info = {'next_step_class': 'Think', 'next_step': self.llm['reasoning']}
            elif self.llm['choice'] == 'Act: ':
                self.llm += guidance.select(skill_set, name='nl_skill') + '\n'
                next_step_info = {'next_step_class': 'Act', 'next_step': self.llm['nl_skill']}
            elif self.llm['choice'] == 'Expand:\n':
                self.llm += '- control flow: ' + guidance.select(['sequence', 'fallback', 'parallel'], name='control_flow') + '\n- subgoals: ' + guidance.gen(stop='\n', name='conditions', max_tokens=200, temperature=0) + '\nOK.\n'
                next_step_info = {'next_step_class': 'Expand', 'next_step': {'control_flow': self.llm['control_flow'], 'conditions': self.llm['conditions']}}

            return next_step_info
    
    def add_obs(self, obs_text):
        ### TODO: OpenAI API Case
        if self.model_name in ['gpt-3.5-turbo-0125', 'gpt-4-turbo-2024-04-09', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-05-13']:
            with guidance.assistant():
                self.llm += f'{obs_text}\n'
        else:
            self.llm += f'{obs_text}\n'

    def load_prompt(self, nl_inst, init_obs):
        raise NotImplementedError()
    
    def answer_question(self, question):
        self.llm += f'{question}\nAnswer: ' + guidance.gen(name='answer', max_tokens=200)
        return self.llm['answer']