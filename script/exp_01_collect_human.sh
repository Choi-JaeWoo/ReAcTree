##### Collect Text Trajectories by Human

### 1. WAH-NL 
# ### 1.1 ReAct
# python src/collect_human.py --config-name=wah_react llm_agent.working_memory=False dataset.collect_ex_root_dir=resource/wah
# ### 1.2 ReAct+WM
# python src/collect_human.py --config-name=wah_react llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/wah
# ### 1.3 ReAcTree
# python src/collect_human.py --config-name=wah_reactree llm_agent.working_memory=False dataset.collect_ex_root_dir=resource/wah
### 1.4 ReAcTree+WM
python src/collect_human.py --config-name=wah_reactree llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/wah


### 2. ALFRED 
# ### 2.1 ReAct+WM
# python src/collect_human.py --config-name=alfred_react llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred
# ### 2.2 ReAcTree+WM
# python src/collect_human.py --config-name=alfred_reactree llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred