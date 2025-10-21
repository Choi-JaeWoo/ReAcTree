##### Filter successful trajectories & embed them

### 2. ALFRED 
# ### 2.1 ReAct+WM
# python src/embed_em.py --config-name=alfred_react dataset.check_success=True llm_agent.working_memory=True dataset.embedding_root_dir='resource/alfred/collect_llm' dataset.em_root_dir='resource/alfred/em_llm' 
### 2.2 ReAcTree+WM
python src/embed_em.py --config-name=alfred_reactree dataset.check_success=True llm_agent.working_memory=True dataset.embedding_root_dir='resource/alfred/collect_llm' dataset.em_root_dir='resource/alfred/em_llm'
