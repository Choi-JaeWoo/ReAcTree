##### Filter successful trajectories & embed them

### 1. WAH-NL
# ### 1.1 ReAct
# python src/embed_em.py --config-name=wah_react exp_type=embed_em llm_agent.working_memory=False dataset.embedding_root_dir=resource/wah/collect_llm dataset.em_root_dir=resource/wah/em_llm
# ### 1.2 ReAct+WM
# python src/embed_em.py --config-name=wah_react exp_type=embed_em llm_agent.working_memory=True dataset.embedding_root_dir=resource/wah/collect_llm dataset.em_root_dir=resource/wah/em_llm
# ### 1.3 ReAcTree 
# python src/embed_em.py --config-name=wah_reactree exp_type=embed_em llm_agent.working_memory=False dataset.embedding_root_dir=resource/wah/collect_llm dataset.em_root_dir=resource/wah/em_llm
### 1.4 ReAcTree+WM
python src/embed_em.py --config-name=wah_reactree exp_type=embed_em llm_agent.working_memory=True dataset.embedding_root_dir=resource/wah/collect_llm dataset.em_root_dir=resource/wah/em_llm

### 2. ALFRED 
# ### 2.1 ReAct+WM
# python src/embed_em.py --config-name=alfred_react dataset.check_success=True llm_agent.working_memory=True dataset.embedding_root_dir='resource/alfred/collect_llm' dataset.em_root_dir='resource/alfred/em_llm' 
# ### 2.2 ReAcTree+WM
# python src/embed_em.py --config-name=alfred_reactree dataset.check_success=True llm_agent.working_memory=True dataset.embedding_root_dir='resource/alfred/collect_llm' dataset.em_root_dir='resource/alfred/em_llm'
