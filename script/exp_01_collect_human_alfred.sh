##### Collect Text Trajectories by Human

### 2. ALFRED 
# ### 2.1 ReAct+WM
# python src/collect_human.py --config-name=alfred_react llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred
### 2.2 ReAcTree+WM
python src/collect_human.py --config-name=alfred_reactree llm_agent.working_memory=True dataset.collect_ex_root_dir=resource/alfred