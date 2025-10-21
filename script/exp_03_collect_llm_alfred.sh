##### Collect Text Trajectories by LLM

### 2. ALFRED 
# ### 2.1 ReAct+WM
# python src/collect_llm.py --config-name=alfred_react exp_type=collect_llm llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_human llm_agent.ic_ex_select_type=rag llm_agent.max_decisions=100 alfred.diverse_task=True
### 2.2 ReAcTree+WM
python src/collect_llm.py --config-name=alfred_reactree exp_type=collect_llm llm_agent.model_name=meta-llama/Meta-Llama-3.1-8B llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred prompt.sys_prompt_root_dir=resource/alfred/sys_prompt prompt.ic_ex_root_dir=resource/alfred/em_human llm_agent.ic_ex_select_type=rag llm_agent.max_decisions=100 alfred.diverse_task=True
