##### Evaluate

### 1. WAH-NL 
# ### LLaMA 3.1
# ### 1.1 ReAct
# python src/evaluate.py --config-name=wah_react exp_type=evaluate llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=False prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 
# ### 1.2 ReAct+WM
# python src/evaluate.py --config-name=wah_react exp_type=evaluate llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 
# ### 1.3 ReAcTree
# python src/evaluate.py --config-name=wah_reactree exp_type=evaluate llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=False prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 
### 1.4 ReAcTree+WM
python src/evaluate.py --config-name=wah_reactree exp_type=evaluate llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 

# ### QWEN 2.5
# ### 1.1 ReAct
# python src/evaluate.py --config-name=wah_react exp_type=evaluate llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=False prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 
# ### 1.2 ReAct+WM
# python src/evaluate.py --config-name=wah_react exp_type=evaluate llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 
# ### 1.3 ReAcTree
# python src/evaluate.py --config-name=wah_reactree exp_type=evaluate llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=False prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 
# ### 1.4 ReAcTree+WM
# python src/evaluate.py --config-name=wah_reactree exp_type=evaluate llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/wah/sys_prompt prompt.ic_ex_root_dir=resource/wah/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=199 llm_agent.max_decisions=199 


### 2. ALFRED Valid Seen
### LLaMA 3.1 
# ### 2.1 ReAct+WM
# python src/evaluate.py --config-name=alfred_react exp_type=evaluate dataset.eval_set=valid_seen llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100
# ### 2.2 ReAcTree+WM
# python src/evaluate.py --config-name=alfred_reactree exp_type=evaluate dataset.eval_set=valid_seen llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100

### QWEN 2.5
# ### 2.1 ReAct+WM
# python src/evaluate.py --config-name=alfred_react exp_type=evaluate dataset.eval_set=valid_seen llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100
# ### 2.2 ReAcTree+WM
# python src/evaluate.py --config-name=alfred_reactree exp_type=evaluate dataset.eval_set=valid_seen llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100


# ### 3. ALFRED Valid Unseen
# ### LLaMA 3.1 
# ### 2.1 ReAct+WM
# python src/evaluate.py --config-name=alfred_react exp_type=evaluate dataset.eval_set=valid_unseen llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100
# ### 2.2 ReAcTree+WM
# python src/evaluate.py --config-name=alfred_reactree exp_type=evaluate dataset.eval_set=valid_unseen llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100

# ### QWEN 2.5
# ### 2.1 ReAct+WM
# python src/evaluate.py --config-name=alfred_react exp_type=evaluate dataset.eval_set=unvalid_seen llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100
# ### 2.2 ReAcTree+WM
# python src/evaluate.py --config-name=alfred_reactree exp_type=evaluate dataset.eval_set=unvalid_seen llm_agent.model_name=Qwen/Qwen2.5-7B llm_agent.working_memory=True prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_llm llm_agent.ic_ex_select_type=rag llm_agent.max_steps=100 llm_agent.max_decisions=100