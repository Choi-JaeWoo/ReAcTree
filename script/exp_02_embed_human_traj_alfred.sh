##### Filter successful trajectories & embed them

### 2. ALFRED 
# ### 2.1 ReAct+WM
# python src/embed_em.py --config-name=alfred_react llm_agent.working_memory=True dataset.embedding_root_dir='resource/alfred/collect_human' dataset.em_root_dir='resource/alfred/em_human'
### 2.2 ReAcTree+WM
python src/embed_em.py --config-name=alfred_reactree llm_agent.working_memory=True dataset.embedding_root_dir='resource/alfred/collect_human' dataset.em_root_dir='resource/alfred/em_human'
