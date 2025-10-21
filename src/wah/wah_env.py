import os
import sys
import json
import copy

curr_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(curr_dir, '../..'))
from virtualhome.simulation.environment.unity_environment import UnityEnvironment

import wah.wah_utils as utils

import pdb

class WahUnityEnv(UnityEnvironment):
    def __init__(self, cfg):
        super(WahUnityEnv, self).__init__(num_agents=1,
                                          observation_types=cfg.environment.observation_types,
                                          use_editor=cfg.environment.use_editor,
                                          base_port=cfg.environment.base_port,
                                          port_id=cfg.environment.port_id,
                                          executable_args=cfg.environment.executable_args,
                                          recording_options=cfg.environment.recording_options)
        self.full_graph = None
        with open(cfg.dataset.obj_dict_sim2nl, 'r') as file:
            self.obj_dict_sim2nl = json.load(file)
        with open(cfg.dataset.obj_dict_nl2sim, 'r') as file:
            self.obj_dict_nl2sim = json.load(file)
        self.agent_reset_id = cfg.environment.agent_reset_id
        self.agent_id = 1
        self.cfg = cfg.environment
        self.is_working_memory = cfg.llm_agent.working_memory
        print('VirtualHome environment is initialized!')
        
    def reset(self, task_d):
        # Make sure that characters are out of graph, and ids are ok
        self.task_id = task_d['task_id']
        self.init_graph = copy.deepcopy(task_d['init_graph'])
        self.init_room = task_d['init_room']
        self.task_goal = task_d['task_goal']
        self.task_name = task_d['task_name']
        self.env_id = task_d['env_id']
        print(f'Resetting... Envid: {self.env_id}. Taskid: {self.task_id}. Taskname: {self.task_name}')
        
        # comm & expand scene
        self.comm.reset(self.env_id)
        s,g = self.comm.environment_graph()
        edge_ids = set([edge['to_id'] for edge in g['edges']] + [edge['from_id'] for edge in g['edges']])
        node_ids = set([node['id'] for node in g['nodes']])
        if len(edge_ids - node_ids) > 0:
            raise ValueError("Graph structure error: Some edges refer to noneexistent nodes")
        if self.env_id not in self.max_ids.keys():
            max_id = max([node['id'] for node in g['nodes']])
            self.max_ids[self.env_id] = max_id
        max_id = self.max_ids[self.env_id]

        updated_graph = self.init_graph
        s, g = self.comm.environment_graph()
        updated_graph = utils.separate_new_ids_graph(updated_graph, max_id)
        success, m = self.comm.expand_scene(updated_graph)

        if not success:
            raise RuntimeError("Error expanding scene: " + m)

        self.offset_cameras = self.comm.camera_count()[1]
                
        self.comm.add_character(self.agent_info[self.agent_reset_id], initial_room=self.init_room)
        
        _, self.init_unity_graph = self.comm.environment_graph()
        
        self.changed_graph = True
        graph = self.get_graph()
        self.id2node = {node['id']: node for node in graph['nodes']}
        
        self.cur_recep_info = (None, None)
        self.name_id_dict_sim2nl, self.name_id_dict_nl2sim = utils.make_name_id_dict(graph, self.obj_dict_sim2nl)
        self.vis_log = [{'action': 'init', 'images': self.get_visual_obs()}]
        self.working_memory = {}

    def step(self, nl_skill):
        sim_receps = ['bathroomcabinet', 'bathroomcounter', 'bed', 'bench', 'boardgame', 'bookshelf', 'box', 'cabinet', 'chair', 'closet', 'clothespile', 'coffeetable', 'cuttingboard', 'desk', 'dishwasher', 'floor', 'folder', 'fridge', 'fryingpan', 'garbagecan', 'kitchencabinet', 'kitchencounter', 'kitchentable', 'microwave', 'mousemat', 'nightstand', 'oventray', 'plate', 'printer', 'radio', 'rug', 'sink', 'sofa', 'stove', 'toaster', 'toilet', 'towelrack', 'washingmachine']
        sim_skill_info = utils.decompose_nl_skill(nl_skill, self.name_id_dict_nl2sim, self.cur_recep_info)
        # Update current receptacle
        if (sim_skill_info['sim_act']) == 'walk' and (sim_skill_info['sim_obj_info'][0] in sim_receps):
            self.cur_recep_info = sim_skill_info['sim_obj_info']
        script = utils.make_script(sim_skill_info['sim_act'], sim_skill_info['sim_obj_info'], self.cur_recep_info)

        possible, feedback = self.check_step(sim_skill_info)
        obs = {}

        if possible:
            script_list = [script]
            if self.recording_options['recording']:
                success, message = self.comm.render_script(script_list,
                                                        find_solution=False,
                                                        processing_time_limit=60,
                                                        recording=True,
                                                        skip_animation=False,
                                                        camera_mode=list(self.recording_options['cameras']),
                                                        output_folder=self.recording_options['output_folder'],
                                                        file_name_prefix=self.recording_options['file_name_prefix'])
            else:
                success, message = self.comm.render_script(script_list,
                                                        find_solution=False,
                                                        recording=False,
                                                        skip_animation=True)
            self.changed_graph = True
            vis_log_text = nl_skill
        else:
            self.changed_graph = False
            vis_log_text = f'{nl_skill} (fail)'
        
        obs['text'] = self.get_text_obs(sim_skill_info, feedback)
        obs['images'] = self.get_visual_obs()
        obs_for_skill_set_update = self.get_skill_set_update_info_obs()
        obs['nl_obs_all_rooms_info'] = obs_for_skill_set_update['nl_obs_all_rooms_info']
        obs['nl_obs_partial_objs_info'] = obs_for_skill_set_update['nl_obs_partial_objs_info']
        self.vis_log.append({'action': vis_log_text, 'images': obs['images']})

        if self.is_working_memory:
            self.update_working_memory_obs(sim_skill_info, feedback)
            obs['working_memory'] = self.working_memory
        obs['feedback'] = feedback
        obs['possible'] = possible
        return obs

    def get_init_obs(self):
        init_obs = {}
        init_obs['text'] = self.get_text_obs()
        init_obs['images'] = self.get_visual_obs()
        obs_for_skill_set_update = self.get_skill_set_update_info_obs()
        init_obs['nl_obs_all_rooms_info'] = obs_for_skill_set_update['nl_obs_all_rooms_info']
        init_obs['nl_obs_partial_objs_info'] = obs_for_skill_set_update['nl_obs_partial_objs_info']
        if self.is_working_memory:
            self.update_working_memory_obs()
            init_obs['working_memory'] = self.working_memory
        return init_obs

    def get_visual_obs(self, info={}):
        # Images from agent cameras
        # camera_ids = [self.offset_cameras + 1, self.offset_cameras + 2, self.offset_cameras + 3, self.offset_cameras + 4, self.offset_cameras + 5, self.offset_cameras + 6]
        camera_ids = [self.offset_cameras + 3]
        if 'image_width' in info:
            image_width = info['image_width']
            image_height = info['image_height']
        else:
            image_width, image_height = self.default_image_width, self.default_image_height
        if 'mode' in info:
            current_mode = info['mode']
        else:
            current_mode = 'normal'
        s, images = self.comm.camera_image(camera_ids, mode=current_mode, image_width=image_width, image_height=image_height)
        if not s:
            raise RuntimeError("Can not load images from cameras")
        return images
    
    def get_partial_graph(self):
        full_graph = self.get_graph()
        filtered_sim_objs = ['alcohol', 'apple', 'bananas', 'barsoap', 'bathroom', 'bathroomcabinet', 'bathroomcounter', 'bed', 'bedroom', 'bedroom', 'bellpepper', 'bench', 'book', 'bookshelf', 'box', 'breadslice', 'bucket', 'cabinet', 'candybar', 'carrot', 'cellphone', 'cereal', 'chair', 'character', 'chicken', 'chinesefood', 'chips', 'chocolatesyrup', 'clock', 'closet', 'clothespants', 'clothesshirt', 'coffeepot', 'coffeetable', 'computer', 'condimentbottle', 'condimentshaker', 'cookingpot', 'crackers', 'creamybuns', 'cupcake', 'cutleryfork', 'cutleryknife', 'cutlets', 'cuttingboard', 'desk', 'dishbowl', 'dishwasher', 'dishwashingliquid', 'door', 'facecream', 'faucet', 'folder', 'fridge', 'fryingpan', 'garbagecan', 'glasses', 'juice', 'kitchen', 'kitchencabinet', 'kitchencounter', 'kitchentable', 'lightswitch', 'lime', 'livingroom', 'lotionbottle', 'magazine', 'microwave', 'milk', 'milkshake', 'mincedmeat', 'mug', 'nightstand', 'notes', 'oventray', 'pancake', 'pear', 'pie', 'plate', 'plum', 'poundcake', 'pudding', 'remotecontrol', 'salad', 'salmon', 'sink', 'sofa', 'sportsball', 'stove', 'sundae', 'teddybear', 'toaster', 'toilet', 'toiletpaper', 'towelrack', 'toy', 'tv', 'washingmachine', 'washingsponge', 'waterglass', 'whippedcream', 'wine', 'wineglass']
        filtered_graph = utils.extract_graph_by_class_names(full_graph, filtered_sim_objs)
        agent_id = self.agent_id
        
        partial_graph = utils.get_visible_nodes(filtered_graph, agent_id)
        return partial_graph
    
    def get_skill_set_update_info_obs(self):
        partial_graph = self.get_partial_graph()
        obs_all_rooms = utils.obs_all_rooms(partial_graph)
        obs_partial_objs = utils.obs_partial_objs(partial_graph, self.agent_id) 
        nl_obs_all_rooms_info = [self.name_id_dict_sim2nl[obs_room] for obs_room in obs_all_rooms]
        nl_obs_partial_objs_info = [self.name_id_dict_sim2nl[partial_obj] for partial_obj in obs_partial_objs]
        return {'nl_obs_all_rooms_info': nl_obs_all_rooms_info, 'nl_obs_partial_objs_info': nl_obs_partial_objs_info}
    
    def get_text_obs(self, sim_skill_info=None, feedback=None):
        full_graph = self.get_graph()
        agent_id = self.agent_id
        partial_graph = self.get_partial_graph()
        if sim_skill_info == None:
            sim_act = None
        else:
            sim_act = sim_skill_info['sim_act']
            sim_obj_info = sim_skill_info['sim_obj_info']

        if sim_act == None:
            obs_all_rooms = utils.obs_all_rooms(partial_graph)
            obs_agent_room = utils.obs_agent_room(partial_graph, agent_id)            
            obs_room_items = utils.obs_room_items(partial_graph)
            obs_agent_room_nl = self.name_id_dict_sim2nl[obs_agent_room]
            obs_text = f'You are in the house, and there are {len(obs_all_rooms)} rooms: {utils.merge_obs_list(obs_all_rooms, self.name_id_dict_sim2nl)}. '
            obs_text += f'You are in the middle of a {obs_agent_room_nl[0]} ({obs_agent_room_nl[1]}). '
            obs_text += f'Looking quickly around the room, you see {utils.merge_obs_list(obs_room_items, self.name_id_dict_sim2nl)}.'
        elif sim_act == 'walk':
            ### Observations for 1) go to room, or 2) go to object
            if sim_obj_info[0] in ['bathroom', 'bedroom', 'kitchen', 'livingroom']:
                obs_agent_room = utils.obs_agent_room(partial_graph, agent_id)
                obs_room_items = utils.obs_room_items(partial_graph)    
                obs_agent_room_nl = self.name_id_dict_sim2nl[obs_agent_room]
                obs_text = f'{feedback} '
                obs_text += f'Looking quickly around the room, you see {utils.merge_obs_list(obs_room_items, self.name_id_dict_sim2nl)}.'            
            else:
                obs_close_objs = utils.obs_close_objs(partial_graph, agent_id)
                obs_text = f'{feedback} '
                obs_text += f'You see {utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}'
        elif sim_act == 'grab':
            obs_text = feedback
        elif sim_act == 'putin':
            obs_text = feedback
        elif sim_act == 'putback':
            obs_text = feedback
        elif sim_act == 'open':
            obs_close_objs = utils.obs_close_objs(partial_graph, agent_id)
            obs_text = f'{feedback} '
            obs_text += f'You see {utils.merge_obs_list(obs_close_objs, self.name_id_dict_sim2nl)}'
        elif sim_act == 'close':
            obs_text = feedback
        elif sim_act == 'switchon':
            obs_text = feedback
        else:
            raise NotImplementedError()
        obs_grab_objs = utils.obs_agent_grab(partial_graph, agent_id)
        if not obs_grab_objs == None:
            obs_text += f' You hold {utils.merge_obs_list(obs_grab_objs, self.name_id_dict_sim2nl)}.'
        return obs_text
    
    def check_step(self, sim_skill_info):
        sim_act, sim_obj_info = sim_skill_info['sim_act'], sim_skill_info['sim_obj_info']
        graph = self.get_graph()
        name_id_dict_sim2nl = self.name_id_dict_sim2nl
        agent_id = self.agent_id
        nl_obj_info = name_id_dict_sim2nl[sim_obj_info]
        if sim_act == 'walk':
            if sim_obj_info[0] in ['bathroom', 'bedroom', 'kitchen', 'livingroom']:
                feedback = f'You move to the {nl_obj_info[0]} ({nl_obj_info[1]}).'
            else:
                feedback = f'You arrive at the {nl_obj_info[0]} ({nl_obj_info[1]}).'
                if utils.check_properties(graph, sim_obj_info[1], 'CAN_OPEN') and not(nl_obj_info[0] == 'desk'):
                    if utils.check_states(graph, sim_obj_info[1], 'CLOSED'):
                        feedback += f' The {nl_obj_info[0]} ({nl_obj_info[1]}) is closed.'
                    elif utils.check_states(graph, sim_obj_info[1], 'OPEN'):
                        feedback += f' The {nl_obj_info[0]} ({nl_obj_info[1]}) is open.'
            return True, feedback
        elif sim_act == 'grab':
            ### Check: 1) free hand 2) obj_close 3) obj_in_open_recep 4) obj_grabbable
            is_free_hand = utils.check_free_hand(graph, agent_id)
            is_obj_close = utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_in_open_recep, closed_recep_info = utils.check_obj_in_open_recep(graph, sim_obj_info[1])
            is_obj_grabbable = utils.check_properties(graph, sim_obj_info[1], 'GRABBABLE')  
            if not is_free_hand:
                feedback = 'You do not have an empty hand.'
            elif not is_obj_close:
                feedback = f'The {nl_obj_info[0]} is not close to you.'
            elif not is_obj_in_open_recep:
                feedback = f'The {nl_obj_info[0]} is inside closed {closed_recep_info[0]}.'
            elif not is_obj_grabbable:
                feedback = f'The {nl_obj_info[0]} cannot be grabbed.'
            else:
                feedback = f'You pick up {nl_obj_info[0]}.'
            possible = is_free_hand and is_obj_close and is_obj_in_open_recep and is_obj_grabbable
            return possible, feedback
        elif sim_act == 'putin':
            sim_recep_info = sim_skill_info['sim_recep_info']
            nl_recep_info = name_id_dict_sim2nl[sim_recep_info]
            is_holding_obj = utils.check_holding_obj(graph, agent_id, sim_obj_info[1])
            is_recep_close = utils.check_obj_close_to_agent(graph, agent_id, sim_recep_info[1])
            is_recep_container = utils.check_properties(graph, sim_recep_info[1], 'CONTAINERS')
            is_not_recep_closed = not utils.check_states(graph, sim_recep_info[1], 'CLOSED')
            if not is_holding_obj:
                feedback = f'You do not grab the {nl_obj_info[0]}.'
            elif not is_recep_close:
                feedback = f'The {nl_recep_info[0]} is not close to you.'
            elif not is_recep_container:
                feedback = f'The {nl_obj_info[0]} cannot be put down in {nl_recep_info[0]}.'
            elif not is_not_recep_closed:
                feedback = f'The {nl_recep_info[0]} is closed.'
            else:
                feedback = f'You put down {nl_obj_info[0]} in {nl_recep_info[0]}'
            possible = is_holding_obj and is_recep_close and is_recep_container and is_not_recep_closed
            return possible, feedback
        elif sim_act == 'putback':
            ### Check: 1) agent holding obj 2) recep close 3) recep surface
            sim_recep_info = sim_skill_info['sim_recep_info']
            nl_recep_info = name_id_dict_sim2nl[sim_recep_info]
            is_holding_obj = utils.check_holding_obj(graph, agent_id, sim_obj_info[1])
            is_recep_close = utils.check_obj_close_to_agent(graph, agent_id, sim_recep_info[1])
            is_recep_surface = utils.check_properties(graph, sim_recep_info[1], 'SURFACES')
            if not is_holding_obj:
                feedback = f'You do not grab the {nl_obj_info[0]}.'
            elif not is_recep_close:
                feedback = f'The {nl_recep_info[0]} is not close to you.'
            elif not is_recep_surface:
                feedback = f'The {nl_obj_info[0]} cannot be put down on {nl_recep_info[0]}.'
            else:
                feedback = f'You put down {nl_obj_info[0]} on {nl_recep_info[0]}'
            possible = is_holding_obj and is_recep_close and is_recep_surface
            return possible, feedback
        elif sim_act == 'open':
            ### Check: 1) free hand 2) obj close 3) obj opennable 4) obj closed
            is_free_hand = utils.check_free_hand(graph, agent_id)
            is_obj_close = utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_opennable = utils.check_properties(graph, sim_obj_info[1], 'CAN_OPEN')
            is_obj_closed = utils.check_states(graph, sim_obj_info[1], 'CLOSED')
            if not is_free_hand:
                feedback = 'You do not have an empty hand.'
            elif not is_obj_close:
                feedback = f'The {nl_obj_info[0]} is not close to you.'
            elif not is_obj_opennable:
                feedback = f'The {nl_obj_info[0]} cannot be opened'
            elif not is_obj_closed:
                feedback = f'The {nl_obj_info[0]} is already open.'
            else:
                feedback = f'You open {nl_obj_info[0]}.'
            possible = is_free_hand and is_obj_close and is_obj_opennable and is_obj_closed
            return possible, feedback
        elif sim_act == 'close':
            ### Check: 1) free hand 2) obj close 3) obj opennable 4) obj open
            is_free_hand = utils.check_free_hand(graph, agent_id)
            is_obj_close = utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_opennable = utils.check_properties(graph, sim_obj_info[1], 'CAN_OPEN')
            is_obj_open = utils.check_states(graph, sim_obj_info[1], 'OPEN')
            if not is_free_hand:
                feedback = 'You do not have an empty hand.'
            elif not is_obj_close:
                feedback = f'The {nl_obj_info[0]} is not close to you.'
            elif not is_obj_opennable:
                feedback = f'The {nl_obj_info[0]} cannot be opened'
            elif not is_obj_open:
                feedback = f'The {nl_obj_info[0]} is already closed.'
            else:
                feedback = f'You close {nl_obj_info[0]}.'
            possible = is_free_hand and is_obj_close and is_obj_opennable and is_obj_open
            return possible, feedback
        elif sim_act == 'switchon':
            ### Check: 1) free hand 2) obj close 3) obj has_switch 4) obj off
            is_free_hand = utils.check_free_hand(graph, agent_id)
            is_obj_close = utils.check_obj_close_to_agent(graph, agent_id, sim_obj_info[1])
            is_obj_hasswitch = utils.check_properties(graph, sim_obj_info[1], 'HAS_SWITCH')
            is_obj_off = utils.check_states(graph, sim_obj_info[1], 'OFF')
            if not is_free_hand:
                feedback = 'You do not have an empty hand.'
            elif not is_obj_close:
                feedback = f'The {nl_obj_info[0]} is not close to you.'
            elif not is_obj_hasswitch:
                feedback = f'The {nl_obj_info[0]} does not have a switch'
            elif not is_obj_off:
                feedback = f'The {nl_obj_info[0]} is already turned on.'
            else:
                feedback = f'You turn on {nl_obj_info[0]}.'
            possible = is_free_hand and is_obj_close and is_obj_hasswitch and is_obj_off
            return possible, feedback
        else:
            raise NotImplementedError()
    
    def get_possible_skill_set(self):
        obs = self.get_skill_set_update_info_obs()
        nl_obs_all_rooms_info, nl_obs_partial_objs_info = obs['nl_obs_all_rooms_info'], obs['nl_obs_partial_objs_info']

        nl_grabbable_names = ['alcohol', 'apple', 'bananas', 'bar soap', 'bell pepper', 'board game', 'book', 'box', 'slice of bread', 'bucket', 'candle', 'candy bar', 'carrot', 'cell phone', 'cereal', 'chair', 'chicken', 'Chinese food', 'chips', 'chocolate syrup', 'clock', 'pants', 'pile of clothes', 'shirt', 'coat rack', 'coffee pot', 'condiment bottle', 'condiment shaker', 'cooking pot', 'crackers', 'crayons', 'creamy buns', 'cupcake', 'cutlery fork', 'cutlery knife', 'cutlets', 'cutting board', 'bowl', 'dishwashing liquid', 'face cream', 'folder', 'frying pan', 'glasses', 'globe', 'hair product', 'hanger', 'juice', 'keyboard', 'lime', 'lotion bottle', 'magazine', 'milk', 'milkshake', 'minced meat', 'mouse', 'mug', 'notes', 'oven tray', 'pancake', 'paper', 'pear', 'pie', 'pillow', 'plate', 'plum', 'pound cake', 'pudding', 'radio', 'remote control', 'rug', 'salad', 'salmon', 'slippers', 'sports ball', 'sundae', 'teddy bear', 'toilet paper', 'toothbrush', 'toothpaste', 'towel', 'towel rack', 'toy', 'wall phone', 'wall picture frame', 'washing sponge', 'water glass', 'whipped cream', 'wine', 'wine glass']
        nl_open_names = ['bathroom cabinet', 'book', 'bookshelf', 'box', 'cabinet', 'closet', 'pile of clothes', 'coffee pot', 'cooking pot', 'curtains', 'desk', 'dishwasher', 'door', 'folder', 'fridge', 'garbage can', 'hair product', 'kitchen cabinet', 'lotion bottle', 'magazine', 'microwave oven', 'milk', 'nightstand', 'printer', 'radio', 'stove', 'toilet', 'toothpaste', 'washing machine', 'window']
        nl_switch_names = ['candle', 'cell phone', 'clock', 'computer', 'dishwasher', 'faucet', 'fridge', 'light switch', 'microwave oven', 'printer', 'radio', 'remote control', 'stove', 'toaster', 'tv', 'wall phone', 'washing machine']
        nl_recallable_names = []

        skill_set = ['done', 'failure']
        skill_set += [f'go to {room_info[0]} {room_info[1]}' for room_info in nl_obs_all_rooms_info]
        for partial_obj_info in nl_obs_partial_objs_info:
            nl_obj_name, nl_obj_id = partial_obj_info[0], partial_obj_info[1]
            skill_set.append(f'go to {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_grabbable_names:
                skill_set.append(f'pick up {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'put down {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_open_names:
                skill_set.append(f'open {nl_obj_name} {nl_obj_id}')
                skill_set.append(f'close {nl_obj_name} {nl_obj_id}')
            if nl_obj_name in nl_switch_names:
                skill_set.append(f'turn on {nl_obj_name} {nl_obj_id}')
        if self.is_working_memory:
            for nl_grabbalbe_obj in nl_grabbable_names:
                skill_set.append(f'recall location of {nl_grabbalbe_obj}')
        return skill_set
    
    def update_working_memory_obs(self, sim_skill_info=None, feedback=None):
        agent_id = self.agent_id
        partial_graph = self.get_partial_graph()
        if sim_skill_info == None:
            sim_act = None
        else:
            sim_act = sim_skill_info['sim_act']
            sim_obj_info = sim_skill_info['sim_obj_info']
        
        if sim_act == None:
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.name_id_dict_sim2nl[sim_agent_room_info]
            
            sim_room_item_infos = utils.obs_room_items(partial_graph)
            for sim_target_obj_info in sim_room_item_infos:
                nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                utils.update_working_memory(self.working_memory, nl_target_obj_info, nl_agent_room_info, None)
            
        elif sim_act == 'walk':
            ### Observations for 1) go to room, or 2) go to object
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.name_id_dict_sim2nl[sim_agent_room_info]

            if sim_obj_info[0] in ['bathroom', 'bedroom', 'kitchen', 'livingroom']:
                sim_room_item_infos = utils.obs_room_items(partial_graph)
                for sim_target_obj_info in sim_room_item_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, nl_agent_room_info, None)
            else:
                sim_close_obj_infos = utils.obs_close_objs(partial_graph, agent_id)
                sim_location_obj_info = sim_skill_info['sim_obj_info']
                nl_location_obj_info = self.name_id_dict_sim2nl[sim_location_obj_info]
                for sim_target_obj_info in sim_close_obj_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
            
            sim_grab_obj_infos = utils.obs_agent_grab(partial_graph, agent_id)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, ('agent', 1), ('agent', 1))    
        elif sim_act == 'grab':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.name_id_dict_sim2nl[sim_agent_room_info]

            sim_grab_obj_infos = utils.obs_agent_grab(partial_graph, agent_id)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'putin':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.name_id_dict_sim2nl[sim_agent_room_info]
            sim_location_obj_info = sim_skill_info['sim_recep_info']
            nl_location_obj_info = self.name_id_dict_sim2nl[sim_location_obj_info]
            sim_target_obj_info = sim_skill_info['sim_obj_info']
            nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]

            utils.update_working_memory(self.working_memory, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)
            sim_grab_obj_infos = utils.obs_agent_grab(partial_graph, agent_id)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'putback':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.name_id_dict_sim2nl[sim_agent_room_info]
            sim_location_obj_info = sim_skill_info['sim_recep_info']
            nl_location_obj_info = self.name_id_dict_sim2nl[sim_location_obj_info]
            sim_target_obj_info = sim_skill_info['sim_obj_info']
            nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
            utils.update_working_memory(self.working_memory, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)

            sim_grab_obj_infos = utils.obs_agent_grab(partial_graph, agent_id)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'open':
            sim_agent_room_info = utils.obs_agent_room(partial_graph, agent_id)
            nl_agent_room_info = self.name_id_dict_sim2nl[sim_agent_room_info]
            sim_close_obj_infos = utils.obs_close_objs(partial_graph, agent_id)
            sim_location_obj_info = sim_skill_info['sim_obj_info']
            nl_location_obj_info = self.name_id_dict_sim2nl[sim_location_obj_info]
            for sim_target_obj_info in sim_close_obj_infos:
                nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                utils.update_working_memory(self.working_memory, nl_target_obj_info, nl_agent_room_info, nl_location_obj_info)

            sim_grab_obj_infos = utils.obs_agent_grab(partial_graph, agent_id)
            if sim_grab_obj_infos:
                for sim_target_obj_info in sim_grab_obj_infos:
                    nl_target_obj_info = self.name_id_dict_sim2nl[sim_target_obj_info]
                    utils.update_working_memory(self.working_memory, nl_target_obj_info, ('agent', 1), ('agent', 1))
        elif sim_act == 'close':
            pass
        elif sim_act == 'switchon':
            pass
        else:
            raise NotImplementedError()
        