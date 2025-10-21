import os, math, re
import textwrap

import numpy as np
from scipy import spatial
from PIL import Image, ImageDraw, ImageFont
import logging

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '')
sys.path.insert(0, './alfred')

from alfred.env.thor_env import ThorEnv
from alfred.gen import constants
from alfred.gen.utils.game_util import get_objects_with_name_and_prop
from src.alfred.utils import natural_word_to_ithor_name
import src.alfred.utils as utils
import copy
from alfred.utils import ALFRED_RECEP, AFLRED_RECEP_MOVABLE
from alfred.utils import natural_word_to_ithor_name
from alfred.utils import name_id_dict_nl2sim

log = logging.getLogger(__name__)


class ThorConnector(ThorEnv):
    def __init__(self, cfg, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        super().__init__(x_display, player_screen_height, player_screen_width, quality, build_path)
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf", 24)
        self.agent_height = 0.9
        self.cur_receptacle = None
        self.reachable_positions, self.reachable_position_kdtree = None, None
        self.sliced = False
        self.is_working_memory = cfg.llm_agent.working_memory
        if self.is_working_memory:
            self.last_visit_recep  = None
            self.visible_obj = []

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        super().restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        self.cur_receptacle = None

    def get_reachable_positions(self):
        free_positions = super().step(dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree

    def write_step_on_img(self, instr):
        img = Image.fromarray(self.last_event.frame)
        lines = textwrap.wrap(instr, width=20)
        y_text = 6
        draw = ImageDraw.Draw(img)
        for line in lines:
            width, height = self.font.getsize(line)
            draw.text((6, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += height
        return img

    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    def llm_skill_interact(self, instruction: str = None, room=None):
        obj_name = []
        obs_obj = None
        put_obj = None      
        is_init_obs = False
        objects = self.last_event.metadata['objects']
        obj_list = [obj['name'] for obj in objects]
        name_dict = self.get_id_sim2nl()
        _, init_obs = utils.name_id_dict_sim2nl(obj_list, name_dict)

        if instruction is None:
            init_recep = utils.obs_partial_recep(objects)
            _, init_obs = utils.name_id_dict_sim2nl(init_recep, name_dict)
                    
            init_obs_str = ', '.join(init_obs)
            ret = f'You are in the house, and you arrive at the {self.get_num2room(room)}, Looking quickly around the room, you see {init_obs_str}.'
            obs_obj = init_obs
            is_init_obs = True
        else:
            is_init_obs = False
            if instruction.startswith("put down ") or instruction.startswith("open "):
                pass
            else:
                self.cur_receptacle = None
                
            if instruction.startswith("go to "):
                obj_sim_name = instruction.replace('go to a ', '').replace('go to an ', '').replace('go to ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                self.cur_receptacle = obj_sim_name
                ret, obs_obj = self.nav_obj(obj_sim_name, obj_name, self.sliced)
                # working memory updates 
                if self.is_working_memory:
                    self.last_visit_recep = obj_sim_name # lastly visited receptacle information
                    self.update_working_memory_obs("go to", obj_sim_name)
            elif instruction.startswith("pick up "):
                obj_sim_name = instruction.replace('pick up the ', '').replace('pick up ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.pick(obj_sim_name, obj_name)
                # working memory updates 
                if self.is_working_memory:
                    self.update_working_memory_obs("pick up", obj_sim_name)
            elif instruction.startswith("put down "):
                m = re.match(r'put down (.+)', instruction)
                obj = m.group(1).replace('the ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj, name_dict)
                receptacle = self.cur_receptacle
                if self.cur_receptacle is None:
                    ret, obs_obj = self.drop()
                    if self.is_working_memory:
                        self.update_working_memory_obs(
                            "drop", 
                            interact_obj=obj, 
                            interact_recep=receptacle
                        )
                else:
                    ret, obs_obj = self.put(obj, receptacle)
                    if not self.last_event.metadata['lastActionSuccess']:
                        # if put down failed, then drop the object
                        ret, obs_obj = self.drop()
                        self.last_event.metadata['lastActionSuccess'] = False
                        if self.is_working_memory:
                            self.update_working_memory_obs("drop", interact_obj=obj)
                    else: 
                        if self.is_working_memory:
                            self.update_working_memory_obs(
                                "put down", 
                                interact_obj=obj, 
                                interact_recep=receptacle
                            )
            elif instruction.startswith("open "):
                obj_sim_name = instruction.replace('open the ', '').replace('open ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.open(obj_sim_name, obj_name)
                # working memory updates 
                if self.is_working_memory:
                    self.last_visit_recep = obj_sim_name # lastly visited receptacle information
                    self.update_working_memory_obs("open", obj_sim_name)
            elif instruction.startswith("close "):
                obj_sim_name = instruction.replace('close the ', '').replace('close ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.close(obj_sim_name, obj_name)
                # working memory updates 
                if self.is_working_memory:
                    self.last_visit_recep = obj_sim_name # lastly visited receptacle information
                    self.update_working_memory_obs("close", obj_sim_name)
            elif instruction.startswith("turn on "):
                obj_sim_name = instruction.replace('turn on the ', '').replace('turn on ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.toggleon(obj_sim_name, obj_name)
                # working memory updates 
                if self.is_working_memory:
                    self.update_working_memory_obs("turn on", obj_sim_name)
            elif instruction.startswith("turn off "):
                obj_sim_name = instruction.replace('turn off the ', '').replace('turn off ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.toggleoff(obj_sim_name, obj_name)
                # working memory updates 
                if self.is_working_memory:
                    self.update_working_memory_obs("turn off", obj_sim_name)
            elif instruction.startswith("slice "):
                obj_sim_name = instruction.replace('slice the ', '').replace('slice ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.slice(obj_sim_name, obj_name)
                self.sliced = True
                # working memory updates 
                if self.is_working_memory:
                    self.update_working_memory_obs("slice", obj_sim_name)
            elif instruction.startswith("drop"):
                obj_sim_name = instruction.replace('drop the ', '').replace('drop ', '')
                _, obj_name = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
                ret, obs_obj = self.drop()
                # working memory updates 
                if self.is_working_memory:
                    self.update_working_memory_obs("drop", obj_sim_name)
            else:
                assert False, 'instruction not supported'

            if not self.last_event.metadata['lastActionSuccess']:
                log.warning(f"llm_skill_interact failed")
                log.warning(f"errorMessage: {self.last_event.metadata['errorMessage']}")
                log.warning(f"returned msg: {ret}")
            else:
                log.info(f"Last action succeeded")
            

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            pass
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            _, holding_obj = utils.name_id_dict_sim2nl(self.get_obj_id2name(holding_obj_id), name_dict)
            holding_obj_str = ', '.join(holding_obj)
            put_obj = holding_obj_str
            ret += f' I am holding {holding_obj_str} now.'

        ret_dict = {
            'action': instruction,
            'success': len(ret) <= 0,
            'message': ret,
            'nl_obs_partial_objs_info': obs_obj,  
            'put_obj': put_obj,
            'init_obs': is_init_obs
        }

        return ret_dict

    def get_object_prop(self, name, prop, metadata):
        for obj in metadata['objects']:
            if name in obj['objectId']:
                return obj[prop]
        return None
    
    def get_id_sim2nl(self):
        objects = self.last_event.metadata['objects']
        object_name = [obj['name'] for obj in objects]
        name_dict = utils.make_name_id_dict(object_name)

        return name_dict
    
    # object Id -> object name
    def get_obj_id2name(self, obj_id):
        objects = self.last_event.metadata['objects']
        obj_id2name = [obj['name'] for obj in objects if obj_id == obj['objectId']]

        return obj_id2name
    
    # object name -> object Id
    def get_obj_name2id(self, obj_name):
        objects = self.last_event.metadata['objects']
        obj_name2id = [obj['objectId'] for obj in objects if obj_name == obj['name']]

        return obj_name2id
    
    # object type -> object name
    def get_obj_type2name(self, obj_type):
        objects = self.last_event.metadata['objects']
        obj_type2name = [obj['name'] for obj in objects if obj_type == obj['objectType']]

        return obj_type2name
    
    def get_num2room(self, room):
        match = re.search(r'\d+', room)
        if match:
            num = int(match.group())
        else:
            return 'Invalid number'
    
        if 1 <= num <= 30:
            return 'kitchen'
        elif 201 <= num <= 230:
            return 'living room'
        elif 301 <= num <= 330:
            return 'bedroom'
        elif 401 <= num <= 430:
            return 'bathroom'
        else:
            return 'Invalid number'

    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))
    
    def init_reset(self, traj_data):
        scene_room = traj_data['scene']['floor_plan']
        instruction = None
        ret_msg = self.llm_skill_interact(instruction, scene_room) 
        self.init_event = copy.deepcopy(self.last_event) 
        init_obs = {
            'text': ret_msg['message'],
            'nl_obs_partial_objs_info': ret_msg['nl_obs_partial_objs_info'],
            'init_obs': ret_msg['init_obs'],
            'put_obj': ret_msg['put_obj']
        }
        return init_obs

    def nav_obj(self, target_obj, target_obj_name, prefer_sliced=False):
        objects = self.last_event.metadata['objects']
        action_name = 'object navigation'
        recepable = False
        openable = False
        toggleable = False
        isOpen = False
        isToggled = False
        already_visible = False
        visible_obj = []
        obs_obj = []
        ret_msg = ''
        
        name_dict = self.get_id_sim2nl()
        obj_name = target_obj.split(' ')[0]
        target_obj_name = ', '.join(target_obj_name)

        log.info(f'{action_name} ({target_obj})')

        # find object index from id
        obj_idx = -1
        for i, o in enumerate(objects):
            if o['name'] == target_obj_name:
                obj_idx = i
                if o['receptacle']:
                    recepable = True
                if o['openable']:
                    openable = True
                    if o['isOpen']:
                        isOpen = True 
                if o['toggleable']:
                    toggleable = True
                    if o['isToggled']:
                        isToggled = True
                break

        if obj_idx == -1:
            ret_msg = f'Cannot find {target_obj}'
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            max_attempts = 20
            teleport_success = False

            # get obj location
            loc = objects[obj_idx]['position']
            obj_rot = objects[obj_idx]['rotation']['y']

            # do not move if the object is already visible and close
            if objects[obj_idx]['visible'] and objects[obj_idx]['distance'] < 0.5 and not objects[obj_idx]['receptacle']:
                log.info('object is already visible.')
                already_visible = True
                teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (obj_name == 'Fridge' or obj_name == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position([loc['x'], loc['y'], loc['z']], reachable_pos_idx)

                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]), loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (obj_name == 'Fridge' or obj_name == 'Microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if obj_name == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if obj_name == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0], loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport
                cur_event = super().step(dict(action="TeleportFull",
                                  x=closest_loc[0], y=self.agent_height, z=closest_loc[2],
                                  rotation=rot_angle, horizon=-hor_angle))

                for o in cur_event.metadata['objects']:
                    if o['visible']:
                        visible_obj.append(o['name'])

                _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
                if self.is_working_memory:
                    self.visible_obj = _obs_obj

                visible_obj_str = ', '.join(obs_obj)
                visible_obj_str = 'You see ' + visible_obj_str

                if len(visible_obj) == 0:
                    visible_obj_str = 'There are no visible objects.'

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                else:
                    teleport_success = True
                    break
                
            obj_idx = -1

            if already_visible:
                for obj in objects:
                    if obj['visible']:
                        visible_obj.append(obj['name'])

                _, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
                visible_obj_str = ', '.join(obs_obj)
                visible_obj_str = 'You see ' + visible_obj_str

                ret_msg = f'You already arrive at {target_obj}. {visible_obj_str}. '
            elif recepable:
                ret_msg = f'You arrive at {target_obj}. {visible_obj_str}. '
            else:
                ret_msg = f'{visible_obj_str}. '

            if not teleport_success:
                ret_msg = f'Cannot move to {target_obj}. '

            if openable:
                if isOpen:
                    ret_msg += f'{target_obj} is open. '
                else:
                    ret_msg += f'{target_obj} is closed. '

            if toggleable:
                if isToggled:
                    ret_msg += f'{target_obj} is already turned on. '
                else:
                    pass
                
        return ret_msg, obs_obj

    def pick(self, obj_sim_name, obj_name):
        obs_obj = []
        name_dict = self.get_id_sim2nl()

        pick_obj_str = ', '.join(obj_name)
        obj_id = self.get_obj_name2id(pick_obj_str)
        obj_id = ''.join(obj_id)
        
        log.info(f'pick up {obj_sim_name}')

        for obj in self.last_event.metadata['objects']:
            if obj['name'] == pick_obj_str:
                obj_data = obj

        if obj_id is None:
            ret_msg = f'Cannot find {obj_sim_name} to pick up.'
        else:
            cur_event = super().step(dict(
                action="PickupObject",
                objectId=obj_id,
                forceAction=False
            ))
            if not self.last_event.metadata['lastActionSuccess']:
                if obj_data['distance'] > 2.5 or not obj_data['visible']:
                    ret_msg = f'The {obj_sim_name} is not close to you.'
                elif not obj_data['pickupable']:
                    ret_msg = f'The {obj_sim_name} cannot be pick up.'
                elif self.last_event.metadata['inventoryObjects']:
                    # check if the agent is holding the object
                    holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
                    
                    _, holding_obj = utils.name_id_dict_sim2nl(self.get_obj_id2name(holding_obj_id), name_dict)
                    holding_obj_str = ', '.join(holding_obj)

                    ret_msg = f'You are currently holding {holding_obj_str}.'
                else: ret_msg = f''
            else:
                visible_obj = []

                for o in cur_event.metadata['objects']:
                    if o['visible']:
                        visible_obj.append(o['name'])

                name_dict = self.get_id_sim2nl()    
                _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)

                if self.is_working_memory:
                    self.visible_obj = _obs_obj
                visible_obj_str = ', '.join(obs_obj)

                ret_msg = f'You pick up {obj_sim_name}. You see {visible_obj_str}.'
        return ret_msg, obs_obj

    def put(self, obj_sim_name, receptacle_name):
        # assume the agent always put the object currently holding
        ret_msg = ''
        obs_obj = []
        name_dict = self.get_id_sim2nl()

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = f'You do not holding any object.'
            return ret_msg, obs_obj
        elif obj_sim_name:
            _, holding_obj = utils.name_id_dict_nl2sim(obj_sim_name, name_dict)
            holding_obj_str = ', '.join(holding_obj)
            holding_obj_id = self.get_obj_name2id(holding_obj_str)
            holding_obj_id = ', '.join(holding_obj_id)
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']

        _, recep_name = utils.name_id_dict_nl2sim(receptacle_name, name_dict)
        recep_name_str = ', '.join(recep_name)
        recep_id = self.get_obj_name2id(recep_name_str)
        recep_id = ', '.join(recep_id)

        if not recep_id:
            ret_msg = f'Cannot find {receptacle_name} to put down.'

        log.info(f'put {obj_sim_name} on {receptacle_name}')   

        cur_event = super().step(dict(
            action="PutObject",
            objectId=holding_obj_id,
            receptacleObjectId=recep_id,
            forceAction=True,
            placeStationary=True
        ))

        if not self.last_event.metadata['lastActionSuccess']:
            log.warning(f"PutObject action failed: {self.last_event.metadata['errorMessage']}, trying again...")
            ret_msg = f'Putting the object on {recep_name_str} failed.'
        else:
            visible_obj = []

            for o in cur_event.metadata['objects']:
                if o['visible']:
                    visible_obj.append(o['name'])

            name_dict = self.get_id_sim2nl()    
            _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)

            if self.is_working_memory:
                self.visible_obj = _obs_obj
            visible_obj_str = ', '.join(obs_obj)

            ret_msg = f'You put {obj_sim_name} on {receptacle_name}. You see {visible_obj_str}.'

        return ret_msg, obs_obj

    def drop(self):
        log.info(f'drop')
        ret_msg = ''
        obs_obj = []

        cur_event = super().step(dict(
            action="DropHandObject",
            forceAction=True
        ))

        visible_obj = []

        for o in cur_event.metadata['objects']:
            if o['visible']:
                visible_obj.append(o['name'])

        name_dict = self.get_id_sim2nl()    
        _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
        if self.is_working_memory:
            self.visible_obj = _obs_obj
        visible_obj_str = ', '.join(obs_obj)

        if not self.last_event.metadata['lastActionSuccess']:
            if len(self.last_event.metadata['inventoryObjects']) == 0:
                ret_msg = f'You are not holding any object. You see {visible_obj_str}.'
            else:
                ret_msg = f"Drop action failed. You see {visible_obj_str}."
        else:
            ret_msg = f'You put down failed, drop it. You see {visible_obj_str}.'

        return ret_msg, obs_obj

    def open(self, obj_sim_name, obj_name):
        log.info(f'open {obj_sim_name}')
        ret_msg = ''
        obs_obj = []
        isOpen = False

        open_str = ', '.join(obj_name)
        obj_id = self.get_obj_name2id(open_str)
        obj_id = ', '.join(obj_id)

        for obj in self.last_event.metadata['objects']:
            if obj['name'] == open_str:
                obj_data = obj

        if obj_id is None:
            ret_msg = f'Cannot find {obj_sim_name} to open'
        else:
            # fall-back steps due to agent-recep collision problem
            if obj_data['objectType'] == 'Cabinet':
                for i in range(4):
                    _ = super().step(dict(action='MoveBack'))
            elif obj_data['objectType'] == 'Drawer':
                for i in range(2):
                    _ = super().step(dict(action='MoveBack'))
            elif obj_data['objectType'] == 'Microwave':
                for i in range(2):
                    _ = super().step(dict(action='MoveBack'))
            else:
                pass
            
            cur_event = super().step(dict(
                action="OpenObject",
                objectId=obj_id,
                forceAction=True,
                moveMagnitude = 1.0
            ))

            is_open_success = cur_event.metadata['lastActionSuccess']

            if obj_data['objectType'] == 'Cabinet':
                for i in range(4):
                    cur_event = super().step(dict(action='MoveAhead'))
            elif obj_data['objectType'] == 'Drawer':
                for i in range(2):
                    cur_event = super().step(dict(action='MoveAhead'))
            elif obj_data['objectType'] == 'Microwave':
                for i in range(3):
                    cur_event = super().step(dict(action='MoveAhead'))
            else:
                pass

            if is_open_success:
                self.last_event.metadata['lastActionSuccess'] = True

            visible_obj = []

            for o in cur_event.metadata['objects']:
                if o['visible']:
                    visible_obj.append(o['name'])
                if open_str == o['name']:
                    if o['receptacleObjectIds'] is None:
                        isOpen = True

            name_dict = self.get_id_sim2nl()    
            _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
            if self.is_working_memory:
                self.visible_obj = _obs_obj
            visible_obj_str = ', '.join(obs_obj)

            if not self.last_event.metadata['lastActionSuccess']:
                if obj_data['isOpen']:
                    ret_msg = f'The {obj_sim_name} is already open.'
                elif not obj_data['openable']:
                    ret_msg = f'The {obj_sim_name} cannot be opened.'
                elif not obj_data['visible']:
                    ret_msg = f'The {obj_sim_name} is not close to you.'
                else:
                    log.info(self.last_event.metadata['errorMessage'])
                    ret_msg = f'The {obj_sim_name} is open failed.'
            else:
                if isOpen:
                    ret_msg = f'You open {obj_sim_name}. The {obj_sim_name} is empty.'
                else:
                    ret_msg = f'You open {obj_sim_name}. You see {visible_obj_str}.'

        return ret_msg, obs_obj

    def close(self, obj_sim_name, obj_name):
        log.info(f'close {obj_sim_name}')
        ret_msg = ''
        obs_obj = []

        close_str = ', '.join(obj_name)
        obj_id = self.get_obj_name2id(close_str)
        obj_id = ', '.join(obj_id)

        for obj in self.last_event.metadata['objects']:
            if obj['name'] == close_str:
                obj_data = obj

        if obj_id is None:
            ret_msg = f'Cannot find {obj_sim_name} to close.'
        else:
            cur_event = super().step(dict(
                action="CloseObject",
                objectId=obj_id,
                forceAction=False
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                if not obj_data['isOpen']:
                    ret_msg = f'The {obj_sim_name} is already closed.'
                elif not obj_data['openable']:
                    ret_msg = f'The {obj_sim_name} cannot be closed.'
                elif not obj_data['visible']:
                    ret_msg = f'The {obj_sim_name} is not close to you.'
                else:
                    ret_msg = self.last_event.metadata['errorMessage']
            else:
                visible_obj = []

                for o in cur_event.metadata['objects']:
                    if o['visible']:
                        visible_obj.append(o['name'])

                name_dict = self.get_id_sim2nl()    
                _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
                if self.is_working_memory:
                    self.visible_obj = _obs_obj
                visible_obj_str = ', '.join(obs_obj)

                ret_msg = f"You close {obj_sim_name}. You see {visible_obj_str}."

        return ret_msg, obs_obj

    def toggleon(self, obj_sim_name, obj_name):
        log.info(f'turn on {obj_sim_name}')
        ret_msg = ''
        obs_obj = []

        turnon_str = ', '.join(obj_name)
        obj_id = self.get_obj_name2id(turnon_str)
        obj_id = ', '.join(obj_id)

        for obj in self.last_event.metadata['objects']:
            if obj['name'] == turnon_str:
                obj_data = obj

        if obj_id is None:
            ret_msg = f'Cannot find {obj_sim_name} to turn on.'
        else:
            cur_event = super().step(dict(
                action="ToggleObjectOn",
                objectId=obj_id,
                forceAction=False
            ))

        if not self.last_event.metadata['lastActionSuccess']:
            if obj_data['isToggled']:
                ret_msg = f'The {obj_sim_name} already turn on.'
            elif not obj_data['toggleable']:
                ret_msg = f'The {obj_sim_name} cannot be turn on.'
            elif not obj_data['visible']:
                ret_msg = f'The {obj_sim_name} is not close to you.'
        else:
            visible_obj = []

            for o in cur_event.metadata['objects']:
                if o['visible']:
                    visible_obj.append(o['name'])

            name_dict = self.get_id_sim2nl()    
            _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
            if self.is_working_memory:
                self.visible_obj = _obs_obj
            visible_obj_str = ', '.join(obs_obj)

            ret_msg = f'You turn on {obj_sim_name}. You see {visible_obj_str}.'

        return ret_msg, obs_obj

    def toggleoff(self, obj_sim_name, obj_name):
        log.info(f'turn off {obj_sim_name}')
        ret_msg = ''
        obs_obj = []

        turnoff_str = ', '.join(obj_name)
        obj_id = self.get_obj_name2id(turnoff_str)
        obj_id = ', '.join(obj_id)

        for obj in self.last_event.metadata['objects']:
            if obj['name'] == turnoff_str:
                obj_data = obj

        if obj_id is None:
            ret_msg = f'Cannot find {obj_sim_name} to turn off'
        else:
            cur_event = super().step(dict(
                action="ToggleObjectOff",
                objectId=obj_id,
                forceAction=False
            ))

        if not self.last_event.metadata['lastActionSuccess']:
            if not obj_data['isToggled']:
                ret_msg = f'The {obj_sim_name} already turn off.'
            elif not obj_data['toggleable']:
                ret_msg = f'The {obj_sim_name} cannot be turn off.'
            elif not obj_data['visible']:
                ret_msg = f'The {obj_sim_name} is not close to you.'
        else:
            visible_obj = []

            for o in cur_event.metadata['objects']:
                if o['visible']:
                    visible_obj.append(o['name'])

            name_dict = self.get_id_sim2nl()    
            _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
            if self.is_working_memory:
                self.visible_obj = _obs_obj
            visible_obj_str = ', '.join(obs_obj)

            ret_msg = f'You turn off {obj_sim_name}. You see {visible_obj_str}.'

        return ret_msg, obs_obj

    def slice(self, obj_sim_name, obj_name):
        log.info(f'slice {obj_sim_name}')
        ret_msg = ''
        obs_obj = []
        slice_obj = []  

        slice_obj_str = ', '.join(obj_name)
        obj_id = self.get_obj_name2id(slice_obj_str) # obj_id type: list
        obj_id = ''.join(obj_id)

        for obj in self.last_event.metadata['objects']:
            if obj['name'] == slice_obj_str:
                obj_data = obj

        if obj_id is None:
            obj_sim_name = obj_sim_name.spilt(' ')[0]
            ret_msg = f'Cannot find {obj_sim_name} to slice.'
        else:
            cur_event = super().step(dict(
                action="SliceObject",
                objectId=obj_id,
                forceAction=False
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                if obj_data['isSliced']:
                    ret_msg = f'The {obj_sim_name} already sliced.'
                elif not obj_data['sliceable']:
                    ret_msg = f'The {obj_sim_name} cannot be sliced.'
                elif not obj_data['visible']:
                    ret_msg = f'The {obj_sim_name} is not close to you.'
                else:
                    log.info(self.last_event.metadata['errorMessage'])
                    ret_msg = f'The {obj_sim_name} slice failed.'
            else:
                visible_obj = []
                slice_obj_list = []                 
                name_dict = self.get_id_sim2nl()

                for o in cur_event.metadata['objects']:
                    if o['visible']:
                        visible_obj.append(o['name'])
                    if '_Slice_' in o['name'] or '_Sliced' in o['name']:
                        slice_obj_list.append(o['name'])

                _, slice_obj = utils.name_id_dict_sim2nl(slice_obj_list, name_dict) 
                _obs_obj, obs_obj = utils.name_id_dict_sim2nl(visible_obj, name_dict)
                if self.is_working_memory:
                    self.visible_obj = _obs_obj
                visible_obj_str = ', '.join(obs_obj)
                slice_instance_str = ', '.join(slice_obj)   

                ret_msg = f'You slice {obj_sim_name}. You see {visible_obj_str}. {slice_instance_str} are sliced.'  

        return ret_msg, obs_obj
    
    def reset_working_memory(self):
        self.working_memory = dict()

    def get_parent_obj_ids_from_obj_name(self, event, target_obj_name):
        from alfred.utils import name_id_dict_sim2nl
        cur_scene_info = event['objects']
        parent_recep_id = None
        for obj_item in cur_scene_info:
            if obj_item['name']==target_obj_name:
                parent_recep_id = obj_item['parentReceptacles']
                # print(obj_item)
                break
        if parent_recep_id is None:
            return None
        parent_recep_simname = None
        parent_recep_id = parent_recep_id[0]
        # DiningTable|-00.52|+00.01|+00.95
        init_scene_info = self.init_event.metadata['objects']
        for obj_item in init_scene_info:
            if obj_item['objectId']==parent_recep_id:
                parent_recep_simname = obj_item['name']
        if parent_recep_simname is None:
            return None
        # DiningTable_dc543948
        _, parent_recep_nlname = name_id_dict_sim2nl(
            [parent_recep_simname], 
            self.get_id_sim2nl()
        ) # ['DiningTable (2)']
        return parent_recep_nlname[0]

    def update_working_memory_obs(self, sim_act, interact_obj=None, interact_recep=None):
        metadata = self.last_event.metadata

        if sim_act == 'go to':
            is_loc_is_obj = False
            last_loc_class_name = self.last_visit_recep.split("(")[0]
            last_loc_class_name = natural_word_to_ithor_name(last_loc_class_name)
            if last_loc_class_name in ALFRED_RECEP:
                is_loc_is_obj = False
                if last_loc_class_name in AFLRED_RECEP_MOVABLE:
                    is_loc_is_obj = True
            else:
                is_loc_is_obj = True
            parent_recep_nl_name = None
            if is_loc_is_obj:
                cur_scene_info = metadata['objects']
                name_dict = self.get_id_sim2nl()
                _, nl2sim_loc_name = name_id_dict_nl2sim(self.last_visit_recep, name_dict)
                nl2sim_loc_name = nl2sim_loc_name[0] # Bread (1) --> Bread_42314d78
                parent_recep_nl_name = self.get_parent_obj_ids_from_obj_name(
                    metadata, 
                    nl2sim_loc_name
                ) # Bread (1) --> DiningTable (2) 
            #### 
            obs_obj = self.visible_obj # updated when go to, open, close , etc.. 
            name_dict = self.get_id_sim2nl()
            for obj_name in obs_obj: 
                name_dict = self.get_id_sim2nl()
                is_ignore=False
                if metadata['inventoryObjects']:
                    # check if the agent is holding the object
                    holding_obj_id = metadata['inventoryObjects'][0]['objectId']
                    holding_obj_list, _ = utils.name_id_dict_sim2nl(self.get_obj_id2name(holding_obj_id), name_dict)
                    for o in holding_obj_list:
                        if obj_name == o:
                            is_ignore=True
                            break
                if is_ignore: # ignore holding objects 
                    continue
                if not parent_recep_nl_name is None:
                    self.working_memory = utils.update_working_memory(
                        self.working_memory, 
                        obj_name, 
                        parent_recep_nl_name
                    )
                else:
                    self.working_memory = utils.update_working_memory(
                        self.working_memory, 
                        obj_name, 
                        self.last_visit_recep
                    )
        elif sim_act == 'pick up':
            for k in self.working_memory.keys():
                for entry in self.working_memory[k]:
                    if entry['id'] == interact_obj:
                        self.working_memory = utils.update_working_memory(
                            self.working_memory, 
                            interact_obj,
                            'hand' # agent is holding the object
                        )
                        break
        elif sim_act == 'open':
            obs_obj = self.visible_obj 
            name_dict = self.get_id_sim2nl()
            for obj_name in obs_obj:
                name_dict = self.get_id_sim2nl()
                is_ignore=False
                if metadata['inventoryObjects']:
                    # check if the agent is holding the object
                    holding_obj_id = metadata['inventoryObjects'][0]['objectId']
                    holding_obj_list, _ = utils.name_id_dict_sim2nl(self.get_obj_id2name(holding_obj_id), name_dict)
                    for o in holding_obj_list:
                        if obj_name == o:
                            is_ignore=True
                            break
                if is_ignore: # ignore holding objects 
                    continue
                # update only newly appeared objects to receptacle 
                # (Drawer, Cabinet, Microwave, Fridge)
                # 1. check last visit receptacle is object or not
                is_loc_is_obj = False
                last_loc_class_name = self.last_visit_recep.split("(")[0]
                last_loc_class_name = natural_word_to_ithor_name(last_loc_class_name)
                if last_loc_class_name in ALFRED_RECEP:
                    is_loc_is_obj = False
                    if last_loc_class_name in AFLRED_RECEP_MOVABLE:
                        is_loc_is_obj = True
                else:
                    is_loc_is_obj = True
                parent_recep_nl_name = None
                if is_loc_is_obj:
                    cur_scene_info = metadata['objects']
                    name_dict = self.get_id_sim2nl()
                    _, nl2sim_loc_name = name_id_dict_nl2sim(self.last_visit_recep, name_dict)
                    nl2sim_loc_name = nl2sim_loc_name[0] # Bread (1) --> Bread_42314d78
                    parent_recep_nl_name = self.get_parent_obj_ids_from_obj_name(
                        metadata, 
                        nl2sim_loc_name
                    ) # Bread (1) --> DiningTable (2) 
                if not parent_recep_nl_name is None:
                    self.working_memory = utils.update_working_memory(
                        self.working_memory, 
                        obj_name, 
                        parent_recep_nl_name
                    )
                else:
                    self.working_memory = utils.update_working_memory(
                        self.working_memory, 
                        obj_name, 
                        self.last_visit_recep
                    )
        elif sim_act == 'put down': 
            self.working_memory = utils.update_working_memory(
                self.working_memory, 
                interact_obj,  # object 
                interact_recep # location 
            )
        elif sim_act == 'drop':
            for k in self.working_memory.keys():
                for entry in self.working_memory[k]:
                    if entry['id'] == interact_obj:
                        self.working_memory = utils.update_working_memory(
                            self.working_memory, 
                            interact_obj, # object
                            'drop'        # location  
                        )
                        break
        elif sim_act == 'slice': 
            #### [0] check last visit location is object or not 
            # and get parent recep name 
            is_loc_is_obj = False
            last_loc_class_name = self.last_visit_recep.split("(")[0]
            last_loc_class_name = natural_word_to_ithor_name(last_loc_class_name)
            if last_loc_class_name in ALFRED_RECEP:
                is_loc_is_obj = False
                if last_loc_class_name in AFLRED_RECEP_MOVABLE:
                    is_loc_is_obj = True
            else:
                is_loc_is_obj = True
            parent_recep_nl_name = None
            if is_loc_is_obj:
                cur_scene_info = metadata['objects']
                name_dict = self.get_id_sim2nl()
                _, nl2sim_loc_name = name_id_dict_nl2sim(self.last_visit_recep, name_dict)
                nl2sim_loc_name = nl2sim_loc_name[0] # Bread (1) --> Bread_42314d78
                parent_recep_nl_name = self.get_parent_obj_ids_from_obj_name(
                    metadata, 
                    nl2sim_loc_name
                ) # Bread (1) --> DiningTable (2) 

            #### [1] slice target object delete from working memory 
            for k in self.working_memory.keys():
                for entry in self.working_memory[k]:
                    if entry['id'] == interact_obj:
                        self.working_memory = utils.delete_obj_from_working_memory(
                            self.working_memory, 
                            interact_obj,         # object
                            self.last_visit_recep # location
                        )
                        break
            # empty key list delete
            delete_keys = []
            for k in self.working_memory.keys():
                if len(self.working_memory[k]) == 0:
                    # empty key list delete
                    delete_keys.append(k)
            for delete_key in delete_keys:
                del self.working_memory[delete_key]

            #### [2] Visible object update 
            obs_obj = self.visible_obj # updated when go to, open, close , etc.. 
            name_dict = self.get_id_sim2nl()
            for obj_name in obs_obj: 
                name_dict = self.get_id_sim2nl()
                is_ignore=False
                if metadata['inventoryObjects']:
                    # check if the agent is holding the object
                    holding_obj_id = metadata['inventoryObjects'][0]['objectId']
                    holding_obj_list, _ = utils.name_id_dict_sim2nl(self.get_obj_id2name(holding_obj_id), name_dict)
                    for o in holding_obj_list:
                        if obj_name == o:
                            is_ignore=True
                            break
                if is_ignore: # ignore holding objects 
                    continue
                
                # Update Working memory 
                if not parent_recep_nl_name is None:
                    self.working_memory = utils.update_working_memory(
                        self.working_memory, 
                        obj_name, 
                        parent_recep_nl_name
                    )
                else:
                    self.working_memory = utils.update_working_memory(
                        self.working_memory, 
                        obj_name, 
                        self.last_visit_recep
                    )

        elif sim_act == 'close': 
            pass
        elif sim_act == 'turn on': 
            pass
        elif sim_act == 'turn off':
            pass
        else:
            raise NotImplementedError()