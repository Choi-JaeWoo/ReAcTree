import copy
from collections import defaultdict
import re
from PIL import Image, ImageDraw, ImageFont
import textwrap
import os

import pdb
##### Some functions are forked from wahtch_and_help github (https://github.com/xavierpuigf/watch_and_help)
def separate_new_ids_graph(graph, max_id):
    new_graph = copy.deepcopy(graph)
    for node in new_graph['nodes']:
        if node['id'] > max_id:
            node['id'] = node['id'] - max_id + 1000
    for edge in new_graph['edges']:
        if edge['from_id'] > max_id:
            edge['from_id'] = edge['from_id'] - max_id + 1000
        if edge['to_id'] > max_id:
            edge['to_id'] = edge['to_id'] - max_id + 1000
    return new_graph

def get_visible_nodes(graph, agent_id):
    # Obtains partial observation from the perspective of agent_id
    # That is, objects inside the same room as agent_id and not inside closed containers
    # NOTE: Assumption is that the graph has an inside transition that is not transitive
    state = graph
    id2node = {node['id']: node for node in state['nodes']}
    rooms_ids = [node['id'] for node in graph['nodes'] if node['category'] == 'Rooms']
    character = id2node[agent_id]

    # find character
    character_id = character["id"]
    inside_of, is_inside, edge_from = {}, {}, {}

    grabbed_ids = []
    for edge in state['edges']:
        if edge['relation_type'] == 'INSIDE':
            
            if edge['to_id'] not in is_inside.keys():
                is_inside[edge['to_id']] = []
            
            is_inside[edge['to_id']].append(edge['from_id'])
            inside_of[edge['from_id']] = edge['to_id']

        elif 'HOLDS' in edge['relation_type']:
            if edge['from_id'] == character['id']:
                grabbed_ids.append(edge['to_id'])

    character_inside_ids = inside_of[character_id]
    room_id =  character_inside_ids

    object_in_room_ids = is_inside[room_id]

    # Some object are not directly in room, but we want to add them
    curr_objects = list(object_in_room_ids)
    while len(curr_objects) > 0:
        objects_inside = []
        for curr_obj_id in curr_objects:
            new_inside = is_inside[curr_obj_id] if curr_obj_id in is_inside.keys() else []
            objects_inside += new_inside
        
        object_in_room_ids += list(objects_inside)
        curr_objects = list(objects_inside)
    
    # Only objects that are inside the room and not inside something closed
    # TODO: this can be probably speed up if we can ensure that all objects are either closed or open
    object_hidden = lambda ido: inside_of[ido] not in rooms_ids and 'OPEN' not in id2node[inside_of[ido]]['states']
    observable_object_ids = [object_id for object_id in object_in_room_ids if not object_hidden(object_id)] + rooms_ids
    observable_object_ids += grabbed_ids
    
    partilly_observable_state = {
        "edges": [edge for edge in state['edges'] if edge['from_id'] in observable_object_ids and edge['to_id'] in observable_object_ids],
        "nodes": [id2node[id_node] for id_node in observable_object_ids]
    }

    return partilly_observable_state


##### Utility functions for graph
def find_edges_connected_to_node(graph, node_id):
    edges_from_node = [edge for edge in graph['edges'] if edge['from_id']==node_id]
    edges_to_node = [edge for edge in graph['edges'] if edge['to_id']==node_id]
    return {'edges_from_node': edges_from_node, 
            'edges_to_node': edges_to_node}

def get_node_location_details(graph, node_id):
    id2node = {node['id']: node for node in graph['nodes']}
    edges_from_node = find_edges_connected_to_node(graph, node_id)['edges_from_node']
    room_node_ids, in_receptacle_ids, on_receptacle_ids = [], [], []

    for edge in edges_from_node:
        target_node_id = edge['to_id']
        if edge['relation_type'] == 'INSIDE':
            if id2node[target_node_id]['category'] == 'Rooms':
                room_node_ids.append(target_node_id)
            else:
                in_receptacle_ids.append(target_node_id)
        elif edge['relation_type'] == 'ON':
            on_receptacle_ids.append(target_node_id)
    location_details = {
        'room_ids': room_node_ids,
        'in_receptacle_ids': in_receptacle_ids,
        'on_receptacle_ids': on_receptacle_ids
    }
    return location_details

def extract_graph_by_class_names(graph, class_list):
    graph_extract = {}
    graph_extract['nodes'] = [node for node in graph['nodes'] if node['class_name'] in class_list]
    extracted_ids = [node['id'] for node in graph_extract['nodes']]
    graph_extract['edges'] = [edge for edge in graph['edges'] if (edge['from_id'] in extracted_ids and edge['to_id'] in extracted_ids)]
    return graph_extract

##### Observation functions
def obs_all_rooms(graph):
    return [(node['class_name'], node['id']) for node in graph['nodes'] if node['category'] == 'Rooms'] # (class name, id)

def obs_agent_room(graph, agent_id):
    id2node = {node['id']: node for node in graph['nodes']}
    agent_room_ids = get_node_location_details(graph, agent_id)['room_ids']
    if len(agent_room_ids) == 1:
        agent_room_node = id2node[agent_room_ids[0]]
        return (agent_room_node['class_name'], agent_room_node['id'])
    else:
        raise NotImplementedError()

def obs_room_items(partial_graph):
    room_items = ["coffeemaker", "dishwasher", "fridge", "microwave", "stove", "toaster", "washingmachine", "computer", "printer", "tv", "bathroomcabinet", "bathroomcounter", "bathtub", "bed", "bench", "bookshelf", "cabinet", "chair", "closet", "coatrack", "coffeetable", "desk", "faucet", "garbagecan", "hanger", "kitchencabinet", "kitchencounter", "kitchentable", "nightstand", "shelf", "sink", "sofa", "toilet", "towelrack", "ceilinglamp", "tablelamp", "walllamp"]
    obs_room_items = []
    for node in partial_graph['nodes']:
        if node['class_name'] in room_items:
            obs_room_items.append((node['class_name'], node['id']))
    return obs_room_items

def obs_close_objs(partial_graph, agent_id):
    obs_result = []
    id2node = {node['id']: node for node in partial_graph['nodes']}
    for edge in partial_graph['edges']:
        if edge['from_id'] == agent_id and edge['relation_type'] == 'CLOSE':
            target_node_id = edge['to_id']
            target_node = id2node[target_node_id]
            obs_result.append((target_node['class_name'], target_node['id']))
    return obs_result

def obs_partial_objs(partial_graph, agent_id):
    id2node = {node['id']: node for node in partial_graph['nodes']}
    partial_objs = []
    for node in partial_graph['nodes']:
        if node['id'] == agent_id or node['category'] == 'Rooms':
            continue
        else:
            partial_objs.append((node['class_name'], node['id']))
    return partial_objs

def obs_agent_grab(graph, agent_id):
    id2node = {node['id']: node for node in graph['nodes']}
    edges_from_agent = find_edges_connected_to_node(graph, agent_id)['edges_from_node']
    grab_ids =[edge['to_id'] for edge in edges_from_agent if edge['relation_type'] in ['HOLDS_RH', 'HOLDS_LH']]
    if len(grab_ids) == 0:
        return None
    else:
        return [(id2node[grab_id]['class_name'], id2node[grab_id]['id']) for grab_id in grab_ids]
    
##### Natural language transformation functions
def make_name_id_dict(graph, obj_dict_sim2nl):
    id2node = {node['id']: node for node in graph['nodes']}
    class_id_dict = defaultdict(list)
    for obj_id, node in id2node.items():
        class_name = node['class_name']
        class_id_dict[class_name].append(obj_id)
    
    transformed_ids = {}
    for class_name, ids in class_id_dict.items():
        ids.sort()
        for index, obj_id in enumerate(ids, start=1):
            transformed_ids[obj_id] = index
    
    name_id_dict_sim2nl = {}
    name_id_dict_nl2sim = {}
    for obj_id, node in id2node.items():
        class_name = node['class_name']
        name_id_sim = (class_name, obj_id)
        name_id_nl = (obj_dict_sim2nl[class_name], transformed_ids[obj_id])
        
        name_id_dict_sim2nl[name_id_sim] = name_id_nl
        name_id_dict_nl2sim[name_id_nl] = name_id_sim
    return name_id_dict_sim2nl, name_id_dict_nl2sim

def merge_obs_list(obs_sim_list, name_id_dict_sim2nl):
    obs_nl_summary = defaultdict(list)

    for sim_name, sim_id in obs_sim_list:
        if (sim_name, sim_id) in name_id_dict_sim2nl:
            natural_name, natural_id = name_id_dict_sim2nl[(sim_name, sim_id)]
            obs_nl_summary[natural_name].append(natural_id)
    
    result_str = []
    for name, ids in sorted(obs_nl_summary.items()):
        ids_sorted = sorted(set(ids))
        id_str = ', '.join(str(id) for id in ids_sorted)
        result_str.append(f'{name} ({id_str})')
        # result_str.append(f'{name} {id_str}')
    return ', '.join(result_str)

def decompose_nl_skill(nl_skill, name_id_dict_sim2nl, cur_recep_info):
    act_dict_nl2sim = {'go to': 'walk', 'pick up': 'grab', 'open': 'open', 'close': 'close', 'turn on': 'switchon'}
    if 'go to ' in nl_skill:
        sim_act = 'walk'
        nl_obj_info = split_nl_name_id(nl_skill.split('go to ')[1])
        sim_obj_info = name_id_dict_sim2nl[nl_obj_info]
        return {'sim_act': sim_act, 'sim_obj_info': sim_obj_info}
    elif 'pick up ' in nl_skill:
        sim_act = 'grab'
        nl_obj_info = split_nl_name_id(nl_skill.split('pick up ')[1])
        sim_obj_info = name_id_dict_sim2nl[nl_obj_info]
        return {'sim_act': sim_act, 'sim_obj_info': sim_obj_info}
    elif 'put down ' in nl_skill:
        surface_class_names = ['bathroomcabinet', 'bathroomcounter', 'bed', 'bench', 'boardgame', 'bookshelf', 'cabinet', 'chair', 'coffeetable', 'cuttingboard', 'desk', 'floor', 'fryingpan', 'kitchencabinet', 'kitchencounter', 'kitchentable', 'mousemat', 'nightstand', 'oventray', 'plate', 'radio', 'rug', 'sofa', 'stove', 'towelrack']
        container_class_names = ['bathroomcabinet', 'bookshelf', 'box', 'cabinet', 'closet', 'clothespile', 'coffeemaker', 'dishwasher', 'folder', 'fridge', 'fryingpan', 'garbagecan', 'kitchencabinet', 'microwave', 'nightstand', 'printer', 'sink', 'stove', 'toaster', 'toilet', 'washingmachine']
        nl_obj_info = split_nl_name_id(nl_skill.split('put down ')[1])
        sim_obj_info = name_id_dict_sim2nl[nl_obj_info]
        if cur_recep_info[0] == None:
            sim_act = 'putback'
            sim_recep_info = sim_obj_info
        elif cur_recep_info[0] in container_class_names:
            sim_act = 'putin'
            sim_recep_info = cur_recep_info
        elif cur_recep_info[0] in surface_class_names:
            sim_act = 'putback'
            sim_recep_info = cur_recep_info
        else:
            raise NotImplementedError()
        return {'sim_act': sim_act, 'sim_obj_info': sim_obj_info, 'sim_recep_info': sim_recep_info}
    elif 'open ' in nl_skill:
        sim_act = 'open'
        nl_obj_info = split_nl_name_id(nl_skill.split('open ')[1])
        sim_obj_info = name_id_dict_sim2nl[nl_obj_info]
        return {'sim_act': sim_act, 'sim_obj_info': sim_obj_info}
    elif 'close ' in nl_skill:
        sim_act = 'close'
        nl_obj_info = split_nl_name_id(nl_skill.split('close ')[1])
        sim_obj_info = name_id_dict_sim2nl[nl_obj_info]
        return {'sim_act': sim_act, 'sim_obj_info': sim_obj_info}
    elif 'turn on ' in nl_skill:
        sim_act = 'switchon'
        nl_obj_info = split_nl_name_id(nl_skill.split('turn on ')[1])
        sim_obj_info = name_id_dict_sim2nl[nl_obj_info]
        return {'sim_act': sim_act, 'sim_obj_info': sim_obj_info}

def split_nl_name_id(nl_name_id):
    match = re.match(r'^(.+?)\s(\d+)$', nl_name_id)
    if match:
        object_name, index = match.groups()
        index = int(index)
        return (object_name, index)
    else:
        return "Invalid input", -1
    
def make_script(sim_act, sim_obj_info, sim_recep_info):
    if sim_act in ['putin', 'putback']:
        script = f'<char0> [{sim_act}] <{sim_obj_info[0]}> ({sim_obj_info[1]}) <{sim_recep_info[0]}> ({sim_recep_info[1]})'
    elif sim_act in ['walk', 'grab', 'open', 'close', 'switchon']:
        script = f'<char0> [{sim_act}] <{sim_obj_info[0]}> ({sim_obj_info[1]})' 
    return script


##### Check step executability functions
def check_free_hand(graph, agent_id):
    edges_from_agent = find_edges_connected_to_node(graph, agent_id)['edges_from_node']
    grabbed_objs = [edge_from_agent['to_id'] for edge_from_agent in edges_from_agent if 'HOLDS' in edge_from_agent['relation_type']]
    return len(grabbed_objs) < 2

def check_obj_close_to_agent(graph, agent_id, obj_id):
    edges_from_agent = find_edges_connected_to_node(graph, agent_id)['edges_from_node']
    return obj_id in [edge['to_id'] for edge in edges_from_agent if edge['relation_type'] == 'CLOSE']

def check_obj_in_open_recep(graph, obj_id):
    id2node = {node['id']: node for node in graph['nodes']}
    obj_in_recep_ids = get_node_location_details(graph, obj_id)['in_receptacle_ids']

    if len(obj_in_recep_ids) == 0:
        return True, None
    elif len(obj_in_recep_ids) == 1:
        recep_id = obj_in_recep_ids[0]
        if 'CLOSED' in id2node[recep_id]['states']:
            return False, (id2node[recep_id]['class_name'], recep_id)
        else:
            return True, None 
    else:
        raise NotImplementedError()

def check_properties(graph, obj_id, property):
    id2node = {node['id']: node for node in graph['nodes']}
    return property in id2node[obj_id]['properties']

def check_states(graph, obj_id, state):
    id2node = {node['id']: node for node in graph['nodes']}
    return state in id2node[obj_id]['states']

def check_holding_obj(graph, agent_id, obj_id):
    edges_from_agent = find_edges_connected_to_node(graph, agent_id)['edges_from_node']
    x =  [edge_from_agent['relation_type'] for edge_from_agent in edges_from_agent if edge_from_agent['to_id']==obj_id]
    return 'HOLDS_RH' in x or 'HOLDS_LH' in x


from PIL import Image, ImageDraw, ImageFont

def add_text_to_np_img(np_img, text, font_path="UbuntuMono-B.ttf", font_size=35, padding=10):
    img = Image.fromarray(np_img[:, :, ::-1])
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, size=font_size)
    
    image_width = img.width - 2 * padding  
    text_lines = []
    
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        line_bbox = font.getbbox(test_line)
        line_width = line_bbox[2] - line_bbox[0]  
        
        if line_width <= image_width:
            line = test_line
        else:
            text_lines.append(line)  
            line = word              
        
    if line:
        text_lines.append(line)

    y_text = padding
    
    for line in text_lines:
        draw.text((padding, y_text), line, font=font, fill=(255, 255, 255, 255))
        y_text += font_size + 5  
    
    return img

def save_images_in_grid(img_list, text_list, title, file_name, grid_width=7, title_height=70, font_path="UbuntuMono-B.ttf", font_size=45):
    processed_imgs = [add_text_to_np_img(img, text) for img, text in zip(img_list, text_list)]
    image_width, image_height = processed_imgs[0].size
    grid_height = len(processed_imgs) // grid_width + (1 if len(processed_imgs) % grid_width else 0)

    total_height = grid_height * image_height + title_height
    total_width = grid_width * image_width
    grid_image = Image.new('RGB', (total_width, total_height), 'white')

    draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype(font_path, font_size)
    title_lines = textwrap.wrap(title, width=110)
    y_start = 10 if len(title_lines) > 1 else 15
    draw.multiline_text((10, y_start), '\n'.join(title_lines), font=font, fill='black')
    
    y_offset = title_height
    for idx, image in enumerate(processed_imgs):
        x_offset = (idx % grid_width) * image_width
        if idx % grid_width == 0 and idx != 0:
            y_offset += image_height
        grid_image.paste(image, (x_offset, y_offset))
    grid_image.save(file_name, 'PNG')

def save_vis_log(cfg, vis_log, task_id, nl_inst):
    base_path = cfg.out_dir
    img_list = [vis['images'][0] for vis in vis_log]
    text_list = [vis['action'] for vis in vis_log]
    file_path = os.path.join(base_path, f'{task_id}.png')
    save_images_in_grid(img_list, text_list, nl_inst, file_path)


##### Evaluation
def check_goal_condition(task_goal, graph, name_id_dict_sim2nl, name_id_dict_nl2sim):
    # task_goal keys -> 'inside_X_Y' 'on_X_Y' 'turnOn_X'
    id2node = {node['id']: node for node in graph['nodes']}
    task_goal_first_key = next(iter(task_goal))
    to_obj_name = task_goal_first_key.split('_')[-1]
    to_obj_ids = [node['id'] for node in graph['nodes'] if node['class_name']==to_obj_name]
    
    final_state_candi = {}
    for to_obj_id in to_obj_ids:
        final_state = {}
        for goal_key, goal_n in task_goal.items():
            if 'turnOn' in goal_key:
                relation, _ = goal_key.split('_')
                states = id2node[to_obj_id]['states']
                if 'ON' in states:
                    final_state[goal_key] = (1, goal_n)
                else:
                    final_state[goal_key] = (0, goal_n)
            elif 'on' in goal_key or 'inside' in goal_key:
                relation, from_obj_name, _ = goal_key.split('_')
                edges_to_node = find_edges_connected_to_node(graph, to_obj_id)['edges_to_node']
                
                count_satisfied = 0
                for edge in edges_to_node:
                    if id2node[edge['from_id']]['class_name'] == from_obj_name and relation == edge['relation_type'].lower():
                        count_satisfied += 1
                final_state[goal_key] = (count_satisfied, goal_n)
        final_state_candi[to_obj_id] = final_state

    subgoal_success_rates = {to_obj_id: get_subgoal_success_rate(final_state) for to_obj_id, final_state in final_state_candi.items()}
    max_key = max(subgoal_success_rates, key=subgoal_success_rates.get)
    subgoal_success_rate = subgoal_success_rates[max_key]
    return subgoal_success_rate

def get_subgoal_success_rate(final_state):
    subgoal_nums = 0
    success_subgoal_nums = 0
    for k, v in final_state.items():
        subgoal_nums += v[1]
        success_subgoal_nums += min(v[0], v[1])
    return success_subgoal_nums/subgoal_nums

def update_working_memory(working_memory, nl_target_obj_info, nl_room_info, nl_location_obj_info):
    ### working memory: {'class_name': [{'id': , 'room': , 'close_obj': }]}
    if nl_room_info:
        nl_room = f'{nl_room_info[0]} {nl_room_info[1]}'
    else:
        nl_room = None
    if nl_location_obj_info:
        nl_location_obj = f'{nl_location_obj_info[0]} {nl_location_obj_info[1]}'
    else:
        nl_location_obj = None    

    nl_target_obj_name, nl_target_obj_id = nl_target_obj_info[0], nl_target_obj_info[1] 

    if nl_target_obj_name not in working_memory:
        working_memory[nl_target_obj_name] = []
    updated = False
    for entry in working_memory[nl_target_obj_name]:
        if entry['id'] == nl_target_obj_id:
            entry['room'] = nl_room
            entry['location_obj'] = nl_location_obj
            updated = True
    if not updated:
        working_memory[nl_target_obj_name].append({'id': nl_target_obj_id, 'room': nl_room, 'location_obj': nl_location_obj})

def recall_working_memory(working_memory, target_obj):
    if target_obj in working_memory:
        location_infos = working_memory[target_obj]
        messages = []
        for location_info in location_infos:
            target_obj_id = location_info['id']
            nl_room = location_info['room']
            nl_location_obj = location_info['location_obj']
            if 'agent' in nl_room:
                message = f'You are holding {target_obj} {target_obj_id}.'
            else:
                if nl_location_obj:
                    message = f'You saw {target_obj} {target_obj_id} near {nl_location_obj} in {nl_room}.'
                else:
                    message = f'You saw {target_obj} {target_obj_id} in {nl_room}.'
            messages.append(message)
        obs_text = ' '.join(messages)
    else:
        obs_text = f'You have not seen {target_obj} before.'
    return obs_text

def sort_with_same_similarity(sorted_ic_ex_encode_list):
    final_list = []
    current_similarity = None
    same_similarity_list = []
    
    for experience in sorted_ic_ex_encode_list:
        similarity = experience["similarity"]

        if current_similarity is None:
            current_similarity = similarity

        if similarity == current_similarity:
            same_similarity_list.append(experience)
        else:
            final_list.extend(process_same_similarity_list(same_similarity_list))
            current_similarity = similarity
            same_similarity_list = [experience]
    if same_similarity_list:
        final_list.extend(process_same_similarity_list(same_similarity_list))
    return final_list

def process_same_similarity_list(experience_list, select_next='expand'):
    result_list = []
    success_list = []
    failure_list = []
    expand_list = []
    etc_list = []
    for exp in experience_list:
        if exp["text_trajectory"].endswith('done'):
            success_list.append(exp)
        elif exp["text_trajectory"].endswith('failure'):
            failure_list.append(exp)
        elif 'Expand:' in exp["text_trajectory"]:
            expand_list.append(exp)
        else:
            etc_list.append(exp)
    while expand_list or success_list or failure_list:
        if select_next == "expand" and expand_list:
            result_list.append(expand_list.pop(0))
            select_next = "success"
        elif select_next == "success" and success_list:
            result_list.append(success_list.pop(0))
            select_next = "failure"
        elif select_next == "failure" and failure_list:
            result_list.append(failure_list.pop(0))
            select_next = "expand"
        else:
            if select_next == "expand":
                select_next = "success" if success_list else "failure"
            elif select_next == "success":
                select_next = "failure" if failure_list else "expand"
            elif select_next == "failure":
                select_next = "expand" if expand_list else "success"
    result_list.extend(etc_list)
    return result_list