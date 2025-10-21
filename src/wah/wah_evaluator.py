import logging
import json
import time
from omegaconf import OmegaConf
from tqdm import tqdm

from wah.wah_llm_agent import WahLlmAgent
from wah.wah_env import WahUnityEnv
from wah.wah_react import WahReact
from wah.wah_reactree import WahReactree
from wah.wah_utils import save_vis_log, check_goal_condition

log = logging.getLogger(__name__)

class WahEvaluator():
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self):
        cfg = self.cfg
        log.info(OmegaConf.to_yaml(self.cfg))

        with open(cfg.dataset.wah_testset, 'r') as json_file:
            test_set = json.load(json_file)
        wah_env = WahUnityEnv(cfg)
        
        if cfg.task_planner == 'react':
            wah_llm_agent = WahLlmAgent(cfg)
            tp = WahReact(cfg, wah_llm_agent, wah_env)
        elif cfg.task_planner == 'reactree':
            wah_llm_agent = WahLlmAgent(cfg)
            tp = WahReactree(cfg, wah_llm_agent, wah_env)
        
        start = time.time()
        results = []
        for task_id, task_d in tqdm(enumerate(test_set), total=len(test_set)):
            terminate_info = tp.run(task_d, log)
            task_goal, graph = task_d['task_goal'], wah_env.get_graph()
            name_id_dict_sim2nl, name_id_dict_nl2sim = wah_env.name_id_dict_sim2nl, wah_env.name_id_dict_nl2sim
            goal_success_rate, subgoal_success_rate = self.evaluate_task_completion(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim)
            result = {'task_id': task_d['task_id'],
                      'nl_inst': task_d['nl_instructions'][0],
                      'goal_success_rate': goal_success_rate,
                      'subgoal_success_rate': subgoal_success_rate}
            if self.cfg.task_planner == 'reactree':
                def get_max_depth(root):
                    if not root.children:
                        return root.depth
                    return max(get_max_depth(child) for child in root.children)
                result['max_depth'] = get_max_depth(tp.root_node)
            log.info(result)
            results.append(result)
            if wah_env.cfg.vis_log:
                save_vis_log(self.cfg, wah_env.vis_log, task_id, task_d['nl_instructions'][0])
        log.info(results)
        num_task = len(results)

        avg_goal_success_rate = sum([result['goal_success_rate'] for result in results]) / num_task
        avg_subgoal_success_rate = sum([result['subgoal_success_rate'] for result in results]) / num_task
        
        log.info(f'average goal success rate: {avg_goal_success_rate * 100:.2f} %')
        log.info(f'average subgoal success rate: {avg_subgoal_success_rate * 100:.2f} %')
        log.info(f'took {(time.time() - start) / 60:.1f} mins')
    
    def evaluate_task_completion(self, task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim):
        subgoal_success_rate = check_goal_condition(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim)
        if subgoal_success_rate == 1:
            goal_success_rate = 1
        else:
            goal_success_rate = 0
        return goal_success_rate, subgoal_success_rate