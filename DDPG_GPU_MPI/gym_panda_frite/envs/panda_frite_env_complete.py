"""
Laurent LEQUIEVRE
Research Engineer, CNRS (France)
ISPR - MACCS Team
Institut Pascal UMR6602
laurent.lequievre@uca.fr

Melodie DANIEL
Post-doctorate
ISPR - MACCS Team
Institut Pascal UMR6602
melodie.daniel@sigma-clermont.fr
"""

import os, inspect
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

import math
from numpy import linalg as LA

import threading
from datetime import datetime
import time

from gym_panda_frite.envs.debug_gui import Debug_Gui
from gym_panda_frite.envs.panda_robot import PandaArm
from gym_panda_frite.envs.ik_dh import IK_DH

from ddpg import DDPGagent

# FOR ROS ONLY ==============
"""
import rospy
import rospkg
import tf
from geometry_msgs.msg import PoseArray, Point, Quaternion
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from std_msgs.msg import Float64
from std_msgs.msg import Float64MultiArray

from visualization_msgs.msg import MarkerArray, Marker
"""
# ===========================


class PandaFriteEnvComplete(gym.Env):
	
	def __init__(self, database = None, json_decoder = None, env_pybullet = None, gui = None, env_rank=None):
		
		if json_decoder==None:
			raise RuntimeError("=> PandaFriteEnvComplete class need a JSON Decoder, to get some parameters !!!")
			return
	
		print("****** PandaFriteEnvComplete !!!! ************")
		
		# init public class properties
		self.rank = env_rank
		self.json_decoder = json_decoder
		self.gui = gui
		self.database = database
		self.env_pybullet = env_pybullet
		
		# open log file
		self.env_file_log = open(self.json_decoder.config_dir_name + 'log_env_rank'+ str(self.rank) + '.txt', "w+")
		self.env_file_log.write("rank = {}\n".format(self.rank))
		
		# read JSON env properties
		self.is_graphic_mode = self.json_decoder.config_data["env"]["is_graphic_mode"]
		self.env_random_seed = self.json_decoder.config_data["env"]["random_seed"]
		self.do_reset_env = json_decoder.config_data["env"]["do_reset_env"]
		self.E = self.json_decoder.config_data["env"]["frite_parameters"]["E"]
		self.NU = self.json_decoder.config_data["env"]["frite_parameters"]["NU"]
		self.env_file_log.write("env_random_see = {}, do_reset_env = {}, E = {}, NU = {}\n".format(self.env_random_seed, self.do_reset_env, self.E, self.NU))
		
		self.dt_factor = self.json_decoder.config_data["env"]["panda_parameters"]["dt_factor"]
		self.joint_motor_control_force = self.json_decoder.config_data["env"]["panda_parameters"]["joint_motor_control_force"]
		self.time_set_action = self.json_decoder.config_data["env"]["time_set_action"]
		self.distance_threshold = self.json_decoder.config_data["env"]["reward_parameters"]["distance_threshold"]
		self.env_file_log.write("dt_factor = {}, joint_motor_control_force = {}, time_set_action = {}, distance_threshold={}\n".format(self.dt_factor, self.joint_motor_control_force, self.time_set_action,self.distance_threshold))
		self.env_file_log.flush()
		
		# Points (on the front side) from bottom to up
		# [5, 2]
		#   ||
		#   vv
		# [31, 15]
		#   ||
		#   vv
		# [47, 33]
		#   ||
		#   vv
		# [13, 10]
		#   ||
		#   vv
		# [18, 14]
		#   ||
		#   vv
		# [28, 53]
		#   ||
		#   vv
		# [9, 6] (TIP)
		#self.id_frite_to_follow = [ [31, 15], [13, 10], [18, 14], [9, 6] ]  # left then right  [left, right], [left,right] ...
		# -> self.id_frite_to_follow = [ [31, 15], [13, 10], [18, 14], [28, 53] ]  # left then right  [left, right], [left,right] ...
		#self.id_frite_to_follow = [ [31, 15], [47, 33], [18, 14], [28, 53] ]  # left then right  [left, right], [left,right] ...
		
		self.id_frite_to_follow = self.json_decoder.config_data["env"]["frite_parameters"]["id_frite_to_follow"]
		
		# Points from bottom to up, on the same plane of id_frite_to _follow, one level under ((on the front side)
		# [63, 38] (under)
		# [31, 15]
		#   ||
		#   vv
		# [64, 45] (under)
		# [47, 33]
		#   ||
		#   vv
		# [58, 54] (under)
		# [13, 10]
		#   ||
		#   vv
		# [42, 37] (under)
		# [18, 14]
		#   ||
		#   vv
		# [23, 32] (under)
		# [28, 53] (under)
		# [9, 6] (TIP)
		# -> self.under_id_frite_to_follow = [ [63, 38], [58, 54], [42, 37], [23, 32] ]  # left then right  [left, right], [left,right] ...
		#self.under_id_frite_to_follow = [ [63, 38], [64, 45], [42, 37], [23, 32] ]  # left then right  [left, right], [left,right] ...
		
		self.under_id_frite_to_follow = self.json_decoder.config_data["env"]["frite_parameters"]["under_id_frite_to_follow"]
		
		# set arrays of meshes
		# array containing the upper mean point shifted by a normalized normal vector
		self.position_mesh_to_follow = [None]*len(self.id_frite_to_follow)
		
		# array containing the upper mean points (between left and right upper points)
		self.mean_position_to_follow = [None]*len(self.id_frite_to_follow)
		
		# set env to database and load datas
		self.database.set_env(self)
		self.database.load()
		
		# pybullet gripper position properties
		self.factor_dt_factor = 1.0
		self.dt = self.env_pybullet.time_step*self.env_pybullet.n_substeps*self.dt_factor*self.factor_dt_factor
		self.max_vel = 1
		
		# random properties
		self.seed(self.env_random_seed)
		
		# set initial pos and orien gripper
		self.initial_pos_gripper = np.array([0.554, 0.0, 0.521])
		self.initial_orien_gripper = np.array([1.0, 0.0, 0.0, 0.0])
		self.env_file_log.write("init : initial_pos_gripper = {}, initial_orien_gripper = {}\n".format(self.initial_pos_gripper, self.initial_orien_gripper))
		self.env_file_log.flush()
			
		# create agent rotation gripper
		self.create_agent_rotation_gripper()
		
		# create debug gui tools
		self.debug_gui = Debug_Gui(env = self)
		
		# read gripper orientation
		self.read_gripper_orientation()
		
		# read all spaces from json config file
		self.read_all_spaces()
		
		# create IK DH
		self.create_IK_DH()
		
		# init gripper orientation
		self.gripper_orientation_to_add = [0.0,0.0,0.0]
		
		# init desired gripper orientation (for agent rotation gripper)
		self.desired_initial_gripper_orientation = [0.0,0.0,0.0]
		
		# read from json config file
		self.is_ros_version = self.json_decoder.config_data["env"]["is_ros_version"]
		if self.is_ros_version:
			print("*** ROS VERSION !!!!!!!!! *********")
			self.env_file_log.write("ROS VERSION !\n".format(self.rank))
			self.env_file_log.flush()
			self.init_ros()
		else:
			print("*** NO ROS VERSION !!!!!!!!!!!! *********")
			self.env_file_log.write("NO ROS VERSION !\n".format(self.rank))
			self.env_file_log.flush()
			# reset env bullet
			self.reset_env_bullet()
		
		
		# For Graphic Cross Update
		self.mutex_get_mesh_data = threading.Lock()
		self.update_cross_is_running = True
		
		if self.gui == True and self.is_graphic_mode == True:
			print('START GRAPHIC THREAD TO UPDATE CROSS !')
			self.draw_cross_thread = threading.Thread(target=self.loop_update_cross)
			self.draw_cross_thread.start()
		
	def loop_update_cross(self):
		time.sleep(5)
		print("START THREAD TO UPDATE CROSS !")
		while True:
			if self.update_cross_is_running == True:
				self.compute_mesh_pos_to_follow(draw_normal=False)
				self.draw_id_to_follow()
				time.sleep(0.5)
	
	
	def create_agent_rotation_gripper(self):
		# Create and load DDPG Agent of Rotation Gripper
		ddpg_cuda = self.json_decoder.config_data["ddpg"]["cuda"]
		ddpg_max_memory_size = self.json_decoder.config_data["ddpg"]["max_memory_size"]
		
		num_states_rotation_gripper = self.json_decoder.config_data["env"]["rotation_gripper_parameters"]["num_states"]
		num_actions_rotation_gripper = self.json_decoder.config_data["env"]["rotation_gripper_parameters"]["num_actions"]
		
		self.agent_rotation_gripper = DDPGagent(ddpg_cuda, num_states=num_states_rotation_gripper, num_actions=num_actions_rotation_gripper, max_memory_size=ddpg_max_memory_size, directory=self.json_decoder.config_dir_name+'env_rotation_gripper/')
		self.agent_rotation_gripper.load()
		
	def reset_env_bullet(self, use_frite=True):
		self.update_cross_is_running = False
		
		self.debug_gui.reset()
		
		self.env_pybullet.reset()
		
		# load plane
		self.load_plane()

		#load panda
		self.load_panda()
		
		# set panda joints to initial positions
		self.set_panda_initial_joints_positions()
		
		#self.draw_gripper_position()
		
		# load cube
		self.load_cube()
	
		# set gym spaces
		self.set_gym_spaces()
		
		if use_frite:
			# load frite
			self.load_frite()
			
			# anchor frite to gripper
			self.create_anchor_panda()
			
			# close gripper
			#self.close_gripper()
	
		self.update_cross_is_running = True
	
	def update_gripper_orientation_bullet(self):
		if self.is_gripper_orien_from_initial():
			self.gripper_orientation_to_add = [0.0,0.0,0.0]
			
		if self.is_gripper_orien_from_agent():
			state_rotation_gripper = np.concatenate((np.array([0.0,0.0,0.0]), self.goal.flatten(), self.desired_initial_gripper_orientation))
			self.gripper_orientation_to_add = self.agent_rotation_gripper.get_action(state_rotation_gripper)
			print("rotation gripper from agent = {}".format(self.gripper_orientation_to_add))
		
		self.go_to_gripper_orientation_bullet()
		
	def reset_bullet(self):
		
		self.goal = self.sample_goal_from_database()
		
		# if not reset env and mode from db or from agent
		if self.do_reset_env == False:
			if self.is_gripper_orien_from_db() == True or self.is_gripper_orien_from_agent() == True:
				# clip current gripper pos to pos_space (in case the gripper is out of current pose_space)
				# go to that clipped position
				current_gripper_pos, current_gripper_orien = self.panda_arm.ee_pose(to_euler=False)
				if self.is_inside_pos_space(current_gripper_pos) == False:
					print("Clip current gripper pos to current pos space !")
					self.env_file_log.write("rank: {}, CLIP current gripper pos [{:.5f}, {:.5f}, {:.5f}] to pos space low={}, high={} !\n".format(self.rank,current_gripper_pos[0],current_gripper_pos[1],current_gripper_pos[2], self.array_low_pos_space, self.array_high_pos_space))
					self.env_file_log.flush()
					clip_pos = np.clip(current_gripper_pos, self.pos_space.low, self.pos_space.high)
					self.env_file_log.write("rank: {}, NEW CLIP gripper pos [{:.5f}, {:.5f}, {:.5f}] !\n".format(self.rank,clip_pos[0],clip_pos[1],clip_pos[2]))
					self.env_file_log.flush()
					self.go_to_cartesian_bullet(clip_pos, current_gripper_orien)
					max_distance, time_elapsed = self.wait_until_frite_deform_ended()
				
		self.update_gripper_orientation_bullet()
		
		if self.gui:
			# draw goal
			self.draw_goal()
		
		#self.previous_reward = -1
				
		return self.get_obs_bullet()
	
	def render(self):
		print("PandaFriteEnvComplete -> render !")
	
	
	def seed(self, seed=None):
		seed_seq = np.random.SeedSequence(seed)
		np_seed = seed_seq.entropy
		self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
		return [np_seed]
	
	# MATH TOOLS ***************************************************************************************************
	
	def truncate_array(self, v, n):
		return np.array([self.truncate(v[i], n) for i in range(len(v))])
	
	def truncate(self, f, n):
		return math.floor(f * 10 ** n) / 10 ** n
	
	# **************************************************************************************************************
	
	# IKDH **********************************************************************************************************
	def create_IK_DH(self):
		# Create and init IK DH
		print("constructor IK_DH")
		start = datetime.now()
		self._ik_dh = IK_DH()
		print("constructor IK_DH time elapsed = {}\n".format(datetime.now()-start))
		self.env_file_log.write("constructor IK_DH time elapsed = {}\n".format(datetime.now()-start))
		self.env_file_log.flush()
		
		# Init IK_DH
		self._ik_dh.init_IK_DH()
	# ***************************************************************************************************************
	
	# RL BULLET *************************************************************************************************************
	
	def is_success(self, d):
		return (d < self.distance_threshold).astype(np.float32)
	
	def step_bullet(self, action, rank=None, episode=None, step=None):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		self.set_action_bullet(action, rank, episode, step)
		
		obs = self.get_obs_bullet()
		
		reward = -1
		done = False
		
		info = {
			'is_success': False,
			'distance_error' : -1,
			'exploded': False,
		}
		
		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
	
	def go_to_position_bullet(self, a_position):
		_, cur_orien = self.panda_arm.ee_pose(to_euler=False)
		new_pos = a_position
		
		self.go_to_cartesian_bullet(new_pos, cur_orien)
		max_distance, time_elapsed = self.wait_until_frite_deform_ended()
	
	def is_inside_pos_space(self, pos):
		x = pos[0]
		y = pos[1]
		z = pos[2]

		low = self.pos_space.low
		high = self.pos_space.high

		x_low = low[0]
		y_low = low[1]
		z_low = low[2]

		x_high = high[0]
		y_high = high[1]
		z_high = high[2]

		if x>=x_low and x<=x_high and y>=y_low and y<=y_high and z>=z_low and z<=z_high:
			b = True
		else:
			b = False

		return b
	
	def go_to_cartesian_bullet(self, pos, orien):
		pos_truncated = self.truncate_array(pos,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(pos_truncated, orien)
		
		pos_fk = self._ik_dh.forward_K(q_values)
		is_inside_box = self.is_inside_pos_space(pos_fk)
		d =  np.linalg.norm(pos_truncated - pos_fk, axis=-1)
		
		if d > 0.01:
			self.env_file_log.write("rank: {}, IK PRECISION: pos desired={}, pos fk={}\n".format(self.rank, pos_truncated, pos_fk))
			self.env_file_log.flush()
		
		if is_inside_limits:
			if is_inside_box:
				if d > 0.01:
					self.env_file_log.write("rank: {}, OUT OF DISTANCE: d={}\n".format(self.rank, d))
					self.env_file_log.flush()
				else:
					self.panda_arm.execute_qvalues(q_values)
			else:
				self.env_file_log.write("rank: {}, OUT OF BOX !\n".format(self.rank))
				self.env_file_log.flush()
		else:
			self.env_file_log.write("rank: {}, OUT OF LIMITS: pos={}, orien={}, q_values={}\n".format(self.rank,pos_truncated,orien,q_values))
			self.env_file_log.flush()
	
	def set_action_bullet(self, action, rank=None, episode=None, step=None):
		assert action.shape == (self.nb_action_values,), 'action shape error'
		
		cur_pos, cur_orien = self.panda_arm.ee_pose(to_euler=False)
		cur_orien_euler = p.getEulerFromQuaternion(cur_orien)
		
		new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		if self.is_action_3D():
			new_orien_quaternion = cur_orien
			
		if self.is_action_6D():
			new_orien_euler = cur_orien_euler + np.array(action[3:]) * self.max_vel * self.dt
			#print("6D : before orien clip : {}".format(new_orien_euler))
			# clip new euler orientation x,y,z from initial [-3.1415,0,0] added with a max range [-0.5..0.5] on each angle.
			new_orien_euler = np.clip(new_orien_euler, [-3.1415-0.5, -0.5, -0.5], [-3.1415+0.5, 0.5, 0.5])
			#print("6D : after orien clip : {}".format(new_orien_euler))
			new_orien_quaternion = p.getQuaternionFromEuler(new_orien_euler)
			
		self.go_to_cartesian_bullet(new_pos, new_orien_quaternion)
		max_distance, time_elapsed = self.wait_until_frite_deform_ended()
		print("*** cartesian pos reached : rank={}, episode={}, step={} wait time elapsed={}".format(rank, episode, step, time_elapsed))
	
	def get_obs_bullet(self):
		gripper_link_pos, gripper_link_orien, gripper_link_orien_euler, gripper_link_vel, gripper_link_vel_orien = self.panda_arm.ee_pose_with_velocity()
		
		self.compute_mesh_pos_to_follow(draw_normal=False)
		mesh_to_follow_pos = np.array(self.position_mesh_to_follow).flatten()
		
		# self.goal = len(self.id_frite_to_follow = [53, 129, 101, 179]) x 3 values (x,y,z) cartesian world position = 12 floats
		# observation = 
		#  3 floats (x,y,z) gripper link cartesian world position  [0,1,2]
		# + 3 float (theta_x,theta_y,theta_z) gripper link cartesian world orientation [3,4,5]
		# + 3 float (vx, vy, vz) gripper link cartesian world velocity [6,7,8]
		# + 3 float (theta__dot_x,theta__dot_y,theta__dot_z) gripper link cartesian world angular velocity [9,10,11]
		# + current cartesian world position of id frite to follow (12 floats) [12,13,14,15,16,17,18,19,20,21,22,23]
		# + self.goal cartesian world position of id frite to reach (12 floats) [24,25,26,27,28,29,30,31,32,33,34,35]
		# observation = 36 floats
		
		#print("goal = {}".format(self.goal))
		#print("goal flat = {}, id pos flat = {}".format(self.goal.flatten(), id_frite_to_follow_pos))
		
		if self.is_action_3D():
			obs = np.concatenate((gripper_link_pos, gripper_link_vel, mesh_to_follow_pos, self.goal.flatten()))
		
		if self.is_action_6D():
			obs = np.concatenate((
							gripper_link_pos, gripper_link_orien_euler, gripper_link_vel, gripper_link_vel_orien, mesh_to_follow_pos, self.goal.flatten())
							)	
		
		if self.add_frite_parameters_to_observation:
			# with frite parameters
			obs = np.concatenate(( obs, np.array([float(self.E), float(self.NU)]) ))
			
		return obs
	
	# ****************************************************************************************************************
	
	# SPACES AND GRIPPER ORIENTATION *********************************************************************************
	
	def read_gripper_orientation(self):
		# Gripper orientation
		orientation_gripper_index = self.json_decoder.config_data["env"]["panda_parameters"]["orientation_gripper"]["index"]
		orientation_gripper_name = self.json_decoder.config_data["env"]["panda_parameters"]["orientation_gripper"]["orientation_gripper_array"][orientation_gripper_index]["name"]
		self.type_orientation_gripper = self.json_decoder.config_data["env"]["panda_parameters"]["orientation_gripper"]["orientation_gripper_array"][orientation_gripper_index]["value"]
		print("** Gripper Orientation *******************")
		print("index = {}, name = {}, type = {}".format(orientation_gripper_index, orientation_gripper_name, self.type_orientation_gripper))
		print("*********************************")
		self.env_file_log.write("Gripper orientation : index = {}, name = {}, type = {}\n".format(orientation_gripper_index, orientation_gripper_name, self.type_orientation_gripper))
		self.env_file_log.flush()
		
	def read_all_spaces(self):
		self.env_file_log.write("*** ALL SPACES ***\n")
		
		# Goal space
		goal_index = self.json_decoder.config_data["all_spaces"]["goal_space"]["index"]
		
		goal_name = self.json_decoder.config_data["all_spaces"]["goal_space"]["goal_array"][goal_index]["name"]
		goal_x_up = self.json_decoder.config_data["all_spaces"]["goal_space"]["goal_array"][goal_index]["x_up"]
		goal_x_down = self.json_decoder.config_data["all_spaces"]["goal_space"]["goal_array"][goal_index]["x_down"]
		goal_y_up = self.json_decoder.config_data["all_spaces"]["goal_space"]["goal_array"][goal_index]["y_up"]
		goal_y_down = self.json_decoder.config_data["all_spaces"]["goal_space"]["goal_array"][goal_index]["y_down"]
		goal_z_down = self.json_decoder.config_data["all_spaces"]["goal_space"]["goal_array"][goal_index]["z_down"]
		
		self.goal_dim_space = np.array([goal_x_up, goal_x_down, goal_y_up, goal_y_down, goal_z_down])
		
		print("** Goal Space *******************")
		print("index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}".format(goal_index, goal_name, goal_x_up, goal_x_down, goal_y_up, goal_y_down, goal_z_down))
		print("*********************************")
		self.env_file_log.write("Goal Space : index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}\n".format(goal_index, goal_name, goal_x_up, goal_x_down, goal_y_up, goal_y_down, goal_z_down))
		
		# Goal space graphic
		goal_index_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["index"]
		
		goal_name_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["goal_array"][goal_index_graphic]["name"]
		goal_x_up_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["goal_array"][goal_index_graphic]["x_up"]
		goal_x_down_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["goal_array"][goal_index_graphic]["x_down"]
		goal_y_up_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["goal_array"][goal_index_graphic]["y_up"]
		goal_y_down_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["goal_array"][goal_index_graphic]["y_down"]
		goal_z_down_graphic = self.json_decoder.config_data["all_spaces"]["goal_space_graphic"]["goal_array"][goal_index_graphic]["z_down"]
		
		self.goal_dim_space_graphic = np.array([goal_x_up_graphic, goal_x_down_graphic, goal_y_up_graphic, goal_y_down_graphic, goal_z_down_graphic])
		
		print("** Goal Space Graphic *******************")
		print("index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}".format(goal_index_graphic, goal_name_graphic, goal_x_up_graphic, goal_x_down_graphic, goal_y_up_graphic, goal_y_down_graphic, goal_z_down_graphic))
		print("*********************************")
		self.env_file_log.write("Goal Space Graphic : index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}\n".format(goal_index_graphic, goal_name_graphic, goal_x_up_graphic, goal_x_down_graphic, goal_y_up_graphic, goal_y_down_graphic, goal_z_down_graphic))
	
		# Pose space
		pos_index = self.json_decoder.config_data["all_spaces"]["pose_space"]["index"]
		
		pos_name = self.json_decoder.config_data["all_spaces"]["pose_space"]["pose_array"][pos_index]["name"]
		pos_x_up = self.json_decoder.config_data["all_spaces"]["pose_space"]["pose_array"][pos_index]["x_up"]
		pos_x_down = self.json_decoder.config_data["all_spaces"]["pose_space"]["pose_array"][pos_index]["x_down"]
		pos_y_up = self.json_decoder.config_data["all_spaces"]["pose_space"]["pose_array"][pos_index]["y_up"]
		pos_y_down = self.json_decoder.config_data["all_spaces"]["pose_space"]["pose_array"][pos_index]["y_down"]
		pos_z_down = self.json_decoder.config_data["all_spaces"]["pose_space"]["pose_array"][pos_index]["z_down"]
		
		self.pos_dim_space = np.array([pos_x_up, pos_x_down, pos_y_up, pos_y_down, pos_z_down])
			
		print("** Pose Space *******************")
		print("index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}".format(pos_index, pos_name, pos_x_up, pos_x_down, pos_y_up, pos_y_down, pos_z_down))
		print("*********************************")
		self.env_file_log.write("Pose Space : index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}\n".format(pos_index, pos_name, pos_x_up, pos_x_down, pos_y_up, pos_y_down, pos_z_down))
		
		# Pose space gaphic
		pos_index_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["index"]
		
		pos_name_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["pose_array"][pos_index_graphic]["name"]
		pos_x_up_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["pose_array"][pos_index_graphic]["x_up"]
		pos_x_down_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["pose_array"][pos_index_graphic]["x_down"]
		pos_y_up_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["pose_array"][pos_index_graphic]["y_up"]
		pos_y_down_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["pose_array"][pos_index_graphic]["y_down"]
		pos_z_down_graphic = self.json_decoder.config_data["all_spaces"]["pose_space_graphic"]["pose_array"][pos_index_graphic]["z_down"]
		
		self.pos_dim_space_graphic = np.array([pos_x_up_graphic, pos_x_down_graphic, pos_y_up_graphic, pos_y_down_graphic, pos_z_down_graphic])
			
		print("** Pose Space Graphic *******************")
		print("index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}".format(pos_index_graphic, pos_name_graphic, pos_x_up_graphic, pos_x_down_graphic, pos_y_up_graphic, pos_y_down_graphic, pos_z_down_graphic))
		print("*********************************")
		self.env_file_log.write("Pose Space Graphic : index = {}, name = {}, x_up= {}, x_down = {}, y_up = {}, y_down = {}, z_down = {}\n".format(pos_index_graphic, pos_name_graphic, pos_x_up_graphic, pos_x_down_graphic, pos_y_up_graphic, pos_y_down_graphic, pos_z_down_graphic))
		
		# Observation space
		self.add_frite_parameters_to_observation = self.json_decoder.config_data["all_spaces"]["observation_space"]["add_frite_parameters"]
		print("** Observation Space *******************")
		print("add_frite_parameters = {}".format(self.add_frite_parameters_to_observation))
		print("*********************************")
		self.env_file_log.write("Observation Space : add_frite_parameters = {}\n".format(self.add_frite_parameters_to_observation))
		
		# Action space
		action_index = self.json_decoder.config_data["all_spaces"]["action_space"]["index"]
		self.type_action = self.json_decoder.config_data["all_spaces"]["action_space"]["action_array"][action_index]["value"]
		action_name = self.json_decoder.config_data["all_spaces"]["action_space"]["action_array"][action_index]["name"]
		
		print("** Action Space *******************")
		print("index = {}, name = {}, type = {}".format(action_index, action_name, self.type_action))
		print("*********************************")
		self.env_file_log.write("Action Space : index = {}, name = {}, type = {}\n".format(action_index, action_name, self.type_action))
		
		self.env_file_log.write("***************\n")
		self.env_file_log.flush()
	
	def is_action_3D(self):
		if self.type_action == 0:
			return True
		else:
			return False
			
	def is_action_6D(self):
		if self.type_action == 1:
			return True
		else:
			return False
	
	def is_gripper_orien_from_initial(self):
		if self.type_orientation_gripper == 0:
			return True
		else:
			return False
	
	def is_gripper_orien_from_db(self):
		if self.type_orientation_gripper == 1:
			return True
		else:
			return False
	
	def is_gripper_orien_from_agent(self):
		if self.type_orientation_gripper == 2:
			return True
		else:
			return False
	
	def test_limits_pos(self):
		#print("Start Test of Pos corners limits !")

		nb_out_limits = 0

		w_lp = self.array_low_pos_space
		w_hp = self.array_high_pos_space
		
		#print("test_limits_pos, w_lp = {}, w_hp = {}".format(w_lp,w_hp))
		#print("test_limits_pos, self.array_low_pos_space={}, self.array_high_pos_space={}".format(self.array_low_pos_space,self.array_high_pos_space))

		x_d = w_lp[0]
		x_u = w_hp[0]
		y_d = w_lp[1]
		y_u = w_hp[1]
		z_d = w_lp[2]
		z_u = w_hp[2]
		
		#print("*** Box pos size = [{},{},{},{},{}]".format(x_u,x_d,y_u,y_d,z_d))
		
		p1 = [x_d,y_d,z_u]
		self.debug_gui.draw_cross("corner_p1", a_pos = p1, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p1", a_text = "p1", a_pos = p1, a_size = 1.5, a_color = [1, 0, 0])
	
		p1_truncated = self.truncate_array(p1,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p1_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p1 into limits !")
		else:
			print("p1 NOT into limits !")
			nb_out_limits+=1
			
		p2 = [x_d,y_u,z_u]
		self.debug_gui.draw_cross("corner_p2", a_pos = p2, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p2", a_text = "p2", a_pos = p2, a_size = 1.5, a_color = [1, 0, 0])

		p2_truncated = self.truncate_array(p2,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p2_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p2 into limits !")
		else:
			print("p2 NOT into limits !")
			nb_out_limits+=1


		p3 = [x_u,y_d,z_u]
		self.debug_gui.draw_cross("corner_p3", a_pos = p3, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p3", a_text = "p3", a_pos = p3, a_size = 1.5, a_color = [1, 0, 0])

		p3_truncated = self.truncate_array(p3,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p3_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p3 into limits !")
		else:
			print("p3 NOT into limits !")
			nb_out_limits+=1


		p4 = [x_u,y_u,z_u]
		self.debug_gui.draw_cross("corner_p4", a_pos = p4, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p4", a_text = "p4", a_pos = p4, a_size = 1.5, a_color = [1, 0, 0])

		p4_truncated = self.truncate_array(p4,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p4_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p4 into limits !")
		else:
			print("p4 NOT into limits !")
			nb_out_limits+=1

		p5 = [x_d,y_d,z_d]
		self.debug_gui.draw_cross("corner_p5", a_pos = p5, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p5", a_text = "p5", a_pos = p5, a_size = 1.5, a_color = [1, 0, 0])

		p5_truncated = self.truncate_array(p5,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p5_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p5 into limits !")
		else:
			print("p5 NOT into limits !")
			nb_out_limits+=1

		p6 = [x_d,y_u,z_d]
		self.debug_gui.draw_cross("corner_p6", a_pos = p6, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p6", a_text = "p6", a_pos = p6, a_size = 1.5, a_color = [1, 0, 0])

		p6_truncated = self.truncate_array(p6,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p6_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p6 into limits !")
		else:
			print("p6 NOT into limits !")
			nb_out_limits+=1

		p7 = [x_u,y_d,z_d]
		self.debug_gui.draw_cross("corner_p7", a_pos = p7, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p7", a_text = "p7", a_pos = p7, a_size = 1.5, a_color = [1, 0, 0])

		p7_truncated = self.truncate_array(p7,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p7_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p7 into limits !")
		else:
			print("p7 NOT into limits !")
			nb_out_limits+=1


		p8 = [x_u,y_u,z_d]
		self.debug_gui.draw_cross("corner_p8", a_pos = p8, a_color = [1, 0, 0] )
		self.debug_gui.draw_text("corner_p8", a_text = "p8", a_pos = p8, a_size = 1.5, a_color = [1, 0, 0])

		p8_truncated = self.truncate_array(p8,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(p8_truncated, self.initial_orien_gripper)
		
		if (is_inside_limits):
			print("p8 into limits !")
		else:
			print("p8 NOT into limits !")
			nb_out_limits+=1
		
		
		print("nb out limits = {}".format(nb_out_limits))
		#print("End Test of corners limits !")
	
	
	def update_goal_space(self, goal_dims):
		self.goal_dim_space = goal_dims
		
		pos_ee = self.initial_pos_gripper
		
		x_down_goal = self.truncate(pos_ee[0]-self.goal_dim_space[1],3)
		y_down_goal = self.truncate(pos_ee[1]-self.goal_dim_space[3],3)
		z_down_goal = self.truncate(pos_ee[2]-self.goal_dim_space[4],3)
		
		x_up_goal = self.truncate(pos_ee[0]+self.goal_dim_space[0],3)
		y_up_goal = self.truncate(pos_ee[1]+self.goal_dim_space[2],3)
		z_up_goal = self.truncate(pos_ee[2],3)
		
		self.array_low_goal_space = np.array([x_down_goal, y_down_goal ,z_down_goal])
		self.array_high_goal_space = np.array([x_up_goal, y_up_goal ,z_up_goal])
		
		self.goal_space = spaces.Box(low=np.array([x_down_goal, y_down_goal ,z_down_goal]), high=np.array([x_up_goal, y_up_goal ,z_up_goal]))
		
	
	def update_pose_space(self, pos_dims):
		self.pos_dim_space = pos_dims
		
		pos_ee = self.initial_pos_gripper
		
		x_down_pos = self.truncate(pos_ee[0]-self.pos_dim_space[1],3)
		y_down_pos = self.truncate(pos_ee[1]-self.pos_dim_space[3],3)
		z_down_pos = self.truncate(pos_ee[2]-self.pos_dim_space[4],3)
		
		x_up_pos = self.truncate(pos_ee[0]+self.pos_dim_space[0],3)
		y_up_pos = self.truncate(pos_ee[1]+self.pos_dim_space[2],3)
		z_up_pos = self.truncate(pos_ee[2],3)
		
		self.array_low_pos_space = np.array([x_down_pos, y_down_pos ,z_down_pos])
		self.array_high_pos_space = np.array([x_up_pos, y_up_pos ,z_up_pos])
		
		self.pos_space = spaces.Box(low=np.array([x_down_pos, y_down_pos ,z_down_pos]), high=np.array([x_up_pos, y_up_pos ,z_up_pos]))
		

	def update_gym_spaces(self, goal_dims, pos_dims):
		self.goal_dim_space = goal_dims
		self.pos_dim_space = pos_dims
		
		pos_ee = self.initial_pos_gripper
		
		x_down_goal = self.truncate(pos_ee[0]-self.goal_dim_space[1],3)
		y_down_goal = self.truncate(pos_ee[1]-self.goal_dim_space[3],3)
		z_down_goal = self.truncate(pos_ee[2]-self.goal_dim_space[4],3)
		
		x_up_goal = self.truncate(pos_ee[0]+self.goal_dim_space[0],3)
		y_up_goal = self.truncate(pos_ee[1]+self.goal_dim_space[2],3)
		z_up_goal = self.truncate(pos_ee[2],3)
		
		self.array_low_goal_space = np.array([x_down_goal, y_down_goal ,z_down_goal])
		self.array_high_goal_space = np.array([x_up_goal, y_up_goal ,z_up_goal])
		
		self.goal_space = spaces.Box(low=np.array([x_down_goal, y_down_goal ,z_down_goal]), high=np.array([x_up_goal, y_up_goal ,z_up_goal]))
		self.env_file_log.write("rank: {}, UPDATE ENV BOX GOAL SPACE : low=[{},{},{}], high=[{},{},{}]\n".format(self.rank,x_down_goal,y_down_goal,z_down_goal,x_up_goal,y_up_goal,z_up_goal))
		self.env_file_log.flush()
		
		#print("rank: {}, UPDATE ENV BOX GOAL SPACE : low=[{},{},{}], high=[{},{},{}]".format(self.rank,x_down_goal,y_down_goal,z_down_goal,x_up_goal,y_up_goal,z_up_goal))
		
		x_down_pos = self.truncate(pos_ee[0]-self.pos_dim_space[1],3)
		y_down_pos = self.truncate(pos_ee[1]-self.pos_dim_space[3],3)
		z_down_pos = self.truncate(pos_ee[2]-self.pos_dim_space[4],3)
		
		x_up_pos = self.truncate(pos_ee[0]+self.pos_dim_space[0],3)
		y_up_pos = self.truncate(pos_ee[1]+self.pos_dim_space[2],3)
		z_up_pos = self.truncate(pos_ee[2],3)
		
		self.array_low_pos_space = np.array([x_down_pos, y_down_pos ,z_down_pos])
		self.array_high_pos_space = np.array([x_up_pos, y_up_pos ,z_up_pos])
		
		self.pos_space = spaces.Box(low=np.array([x_down_pos, y_down_pos ,z_down_pos]), high=np.array([x_up_pos, y_up_pos ,z_up_pos]))
	
		self.env_file_log.write("rank: {}, UPDATE ENV BOX POS SPACE : low=[{},{},{}], high=[{},{},{}]\n".format(self.rank,x_down_pos,y_down_pos,z_down_pos,x_up_pos,y_up_pos,z_up_pos))
		self.env_file_log.flush()
		
		#print("rank: {}, UPDATE ENV BOX POS SPACE : low=[{},{},{}], high=[{},{},{}]".format(self.rank,x_down_pos,y_down_pos,z_down_pos,x_up_pos,y_up_pos,z_up_pos))
		#print("self.array_low_pos_space = {}, self.array_high_pos_space = {}".format(self.array_low_pos_space,self.array_high_pos_space))
		
	
	def get_nb_action_values(self):
		return self.nb_action_values
		
	def get_nb_obs_values(self):
		return self.nb_obs_values
	
	def get_action_space_low(self):
		return self.action_space.low
		
	def get_action_space_high(self):
		return self.action_space.high
	
	def set_gym_spaces(self):
		pos_ee = self.initial_pos_gripper
		
		x_down_goal = self.truncate(pos_ee[0]-self.goal_dim_space[1],3)
		y_down_goal = self.truncate(pos_ee[1]-self.goal_dim_space[3],3)
		z_down_goal = self.truncate(pos_ee[2]-self.goal_dim_space[4],3)
		
		x_up_goal = self.truncate(pos_ee[0]+self.goal_dim_space[0],3)
		y_up_goal = self.truncate(pos_ee[1]+self.goal_dim_space[2],3)
		z_up_goal = self.truncate(pos_ee[2],3)
		
		self.array_low_goal_space = np.float32(np.array([x_down_goal, y_down_goal ,z_down_goal]))
		self.array_high_goal_space = np.float32(np.array([x_up_goal, y_up_goal ,z_up_goal]))
		
		self.goal_space = spaces.Box(low=np.array([x_down_goal, y_down_goal ,z_down_goal]), high=np.array([x_up_goal, y_up_goal ,z_up_goal]))
		self.env_file_log.write("rank: {}, BOX GOAL SPACE : low=[{},{},{}], high=[{},{},{}]\n".format(self.rank,x_down_goal,y_down_goal,z_down_goal,x_up_goal,y_up_goal,z_up_goal))
		self.env_file_log.flush()
		
		# Goal Graphic =============
		
		x_down_goal_graphic = self.truncate(pos_ee[0]-self.goal_dim_space_graphic[1],3)
		y_down_goal_graphic = self.truncate(pos_ee[1]-self.goal_dim_space_graphic[3],3)
		z_down_goal_graphic = self.truncate(pos_ee[2]-self.goal_dim_space_graphic[4],3)
		
		x_up_goal_graphic = self.truncate(pos_ee[0]+self.goal_dim_space_graphic[0],3)
		y_up_goal_graphic = self.truncate(pos_ee[1]+self.goal_dim_space_graphic[2],3)
		z_up_goal_graphic = self.truncate(pos_ee[2],3)
		
		self.array_low_goal_space_graphic = np.float32(np.array([x_down_goal_graphic, y_down_goal_graphic ,z_down_goal_graphic]))
		self.array_high_goal_space_graphic = np.float32(np.array([x_up_goal_graphic, y_up_goal_graphic ,z_up_goal_graphic]))
		
		self.goal_space_graphic = spaces.Box(low=np.array([x_down_goal_graphic, y_down_goal_graphic ,z_down_goal_graphic]), high=np.array([x_up_goal_graphic, y_up_goal_graphic ,z_up_goal_graphic]))
		self.env_file_log.write("rank: {}, BOX GOAL SPACE GRAPHIC : low=[{},{},{}], high=[{},{},{}]\n".format(self.rank,x_down_goal_graphic,y_down_goal_graphic,z_down_goal_graphic,x_up_goal_graphic,y_up_goal_graphic,z_up_goal_graphic))
		self.env_file_log.flush()
		
		# ==========================
		
		x_down_pos = self.truncate(pos_ee[0]-self.pos_dim_space[1],3)
		y_down_pos = self.truncate(pos_ee[1]-self.pos_dim_space[3],3)
		z_down_pos = self.truncate(pos_ee[2]-self.pos_dim_space[4],3)
		
		x_up_pos = self.truncate(pos_ee[0]+self.pos_dim_space[0],3)
		y_up_pos = self.truncate(pos_ee[1]+self.pos_dim_space[2],3)
		z_up_pos = self.truncate(pos_ee[2],3)
		
		self.array_low_pos_space = np.float32(np.array([x_down_pos, y_down_pos ,z_down_pos]))
		self.array_high_pos_space = np.float32(np.array([x_up_pos, y_up_pos ,z_up_pos]))
		
		self.pos_space = spaces.Box(low=np.array([x_down_pos, y_down_pos ,z_down_pos]), high=np.array([x_up_pos, y_up_pos ,z_up_pos]))
	
		self.env_file_log.write("rank: {}, BOX POS SPACE : low=[{},{},{}], high=[{},{},{}]\n".format(self.rank,x_down_pos,y_down_pos,z_down_pos,x_up_pos,y_up_pos,z_up_pos))
		self.env_file_log.flush()
		
		# Pose Graphic ========================
		
		x_down_pos_graphic = self.truncate(pos_ee[0]-self.pos_dim_space_graphic[1],3)
		y_down_pos_graphic = self.truncate(pos_ee[1]-self.pos_dim_space_graphic[3],3)
		z_down_pos_graphic = self.truncate(pos_ee[2]-self.pos_dim_space_graphic[4],3)
		
		x_up_pos_graphic = self.truncate(pos_ee[0]+self.pos_dim_space_graphic[0],3)
		y_up_pos_graphic = self.truncate(pos_ee[1]+self.pos_dim_space_graphic[2],3)
		z_up_pos_graphic = self.truncate(pos_ee[2],3)
		
		self.array_low_pos_space_graphic = np.float32(np.array([x_down_pos_graphic, y_down_pos_graphic ,z_down_pos_graphic]))
		self.array_high_pos_space_graphic = np.float32(np.array([x_up_pos_graphic, y_up_pos_graphic ,z_up_pos_graphic]))
		
		self.pos_space_graphic = spaces.Box(low=np.array([x_down_pos_graphic, y_down_pos_graphic ,z_down_pos_graphic]), high=np.array([x_up_pos_graphic, y_up_pos_graphic ,z_up_pos_graphic]))
	
		self.env_file_log.write("rank: {}, BOX POS SPACE GRAPHIC : low=[{},{},{}], high=[{},{},{}]\n".format(self.rank,x_down_pos_graphic,y_down_pos_graphic,z_down_pos_graphic,x_up_pos_graphic,y_up_pos_graphic,z_up_pos_graphic))
		self.env_file_log.flush()
		
		# =====================================
		
		if self.is_action_3D():
			self.nb_action_values = 3
			
		if self.is_action_6D():
			self.nb_action_values = 6
		
		self.env_file_log.write("rank: {}, nb actions = {}\n".format(self.rank, self.nb_action_values))
		self.env_file_log.flush()
		
		print("** NB Action values = {}".format(self.nb_action_values))
		
		self.action_space = spaces.Box(-1., 1., shape=(self.nb_action_values,), dtype=np.float32)
		
		self.nb_obs_values = 0
		
		# add gripper link cartesian world position (x,y,z)
		self.nb_obs_values += 3
		
		# add gripper link cartesian world velocity (vx, vy, vz)
		self.nb_obs_values += 3
		
		# add gripper link cartesian world orientation (theta_x,theta_y,theta_z)
		self.nb_obs_values += 3

		# add gripper link cartesian world angular velocity (theta__dot_x,theta__dot_y,theta__dot_z)
		self.nb_obs_values += 3
		
		# add current cartesian world position of id frite to follow ((x,y,z) x 4)
		self.nb_obs_values += len(self.id_frite_to_follow) * 3
		
		# add goal cartesian world position of id frite to reach ((x,y,z) x 4)
		self.nb_obs_values += len(self.id_frite_to_follow) * 3
		
		self.pos_of_mesh_in_obs = 12
		
		if self.is_action_3D():
			# remove gripper link world orientation and angular velocity
			self.nb_obs_values -= 6
			self.pos_of_mesh_in_obs = 6
		
		if self.add_frite_parameters_to_observation:
			# add E and NU
			self.nb_obs_values += 2
		
		
		print("*** NB Observation values = {}".format(self.nb_obs_values))
		
		# observation = X float -> see function _get_obs 
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(self.nb_obs_values,), dtype=np.float32)
			
	#*****************************************************************************************************************************
	
	# LOAD AND SET ENV PROPERTIES *******************************************************************************************************************
	
	def set_env_with_frite_parameters(self, values_frite_parameters):
		E = values_frite_parameters[0]
		NU = values_frite_parameters[1]
		time_step = values_frite_parameters[2]
		factor_dt_factor = values_frite_parameters[3]
		x_rot = values_frite_parameters[4]
		y_rot = values_frite_parameters[5]
		z_rot = values_frite_parameters[6]
		
		self.set_E(E)
		self.set_NU(NU)
		self.set_factor_dt_factor(factor_dt_factor)
		self.set_time_step(time_step)
		self.set_gripper_orientation_to_add(x_rot, y_rot, z_rot)
		self.set_desired_initial_gripper_orientation(x_rot, y_rot, z_rot)
		
		x_up_goal = float(values_frite_parameters[7])
		x_down_goal = float(values_frite_parameters[8])
		y_up_goal = float(values_frite_parameters[9])
		y_down_goal = float(values_frite_parameters[10])
		z_down_goal = float(values_frite_parameters[11])
			
		goal_dims = np.array([x_up_goal, x_down_goal, y_up_goal, y_down_goal, z_down_goal])
			
		x_up_pos = float(values_frite_parameters[12])
		x_down_pos = float(values_frite_parameters[13])
		y_up_pos = float(values_frite_parameters[14])
		y_down_pos = float(values_frite_parameters[15])
		z_down_pos = float(values_frite_parameters[16])
			
		pos_dims = np.array([x_up_pos, x_down_pos, y_up_pos, y_down_pos, z_down_pos])
		
		self.update_gym_spaces(goal_dims, pos_dims)
	
	def set_E(self, value):
		self.E = float(value)
		
	def set_NU(self, value):
		self.NU = float(value)
		
	def set_factor_dt_factor(self,value):
		self.factor_dt_factor = float(value)
		
	def set_time_step(self, value):
		self.env_pybullet.time_step=float(value)
		self.dt = self.env_pybullet.time_step*self.env_pybullet.n_substeps*self.dt_factor*self.factor_dt_factor
	
	
	def set_gripper_orientation_to_add(self, x_rot, y_rot, z_rot):
		self.gripper_orientation_to_add = [float(x_rot), float(y_rot), float(z_rot)]
	
	def set_desired_initial_gripper_orientation(self, x_rot, y_rot, z_rot):
		self.desired_initial_gripper_orientation = [float(x_rot), float(y_rot), float(z_rot)]

	def load_plane(self):
		self.plane_height = -0.85
		self.plane_id = p.loadURDF("urdf/plane.urdf", basePosition=[0,0,self.plane_height], useFixedBase=True)
		
	def load_cube(self):
		gripper_pos, _ = self.panda_arm.ee_pose(to_euler=False)
		self.cube_height = 0.365
		cube_z_position = self.plane_height + (self.cube_height / 2.0)
		# load cube
		self.cube_startPos = [gripper_pos[0], gripper_pos[1], cube_z_position]
		self.cube_id = p.loadURDF("urdf/my_cube.urdf", self.cube_startPos, useFixedBase=True)
	
	def load_panda(self):
		# Create a panda arm robot
		self.panda_arm = PandaArm(a_debug_gui=self.debug_gui, a_ik_dh=self._ik_dh)
		self.panda_id = self.panda_arm._id
		self.panda_end_eff_idx = self.panda_arm._end_eff_idx
		
	def load_frite(self):
		gripper_pos, _ = self.panda_arm.ee_pose(to_euler=False)
		self.frite_startOrientation = p.getQuaternionFromEuler([0,0,math.pi/4])

		frite_z_position = self.plane_height + self.cube_height
		self.frite_startPos = [gripper_pos[0], gripper_pos[1], frite_z_position]
		
		
		# plage E -> 0.1 à 40
		# frite blanche :
		# E = 0.1*pow(10,6)  NU = 0.49
		
		# frite noire :  E = 35 , NU = 0.46
		# E = 40*pow(10,6)  NU = 0.49
		
		E = self.E*pow(10,6)
		NU = self.NU
		print("Use JSON E={}, NU={}".format(E,NU))
		
		(a_lambda,a_mu) = self.conv_module_d_young_to_lame(E,NU)
		
		#print("frite a_lambda={}, a_mu={}".format(a_lambda,a_mu))
		
		vtk_file_name = self.json_decoder.config_dir_name + self.json_decoder.config_data["env"]["vtk_file_name"]
		
		# frite : 103 cm with 0.1 cell size
		self.frite_id = p.loadSoftBody(vtk_file_name, basePosition = self.frite_startPos, baseOrientation=self.frite_startOrientation, mass = 0.2, useNeoHookean = 1, NeoHookeanMu = a_mu, NeoHookeanLambda = a_lambda, NeoHookeanDamping = 0.01, useSelfCollision = 1, collisionMargin = 0.001, frictionCoeff = 0.5, scale=1.0)
		#p.changeVisualShape(self.frite_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
		
	# *******************************************************************************************************************
	
	# GOAL **************************************************************************************************************
	def sample_goal_from_goal_space(self):
		# sample a goal np.array[x,y,z] from the goal_space
		goal = np.array(self.goal_space.sample())
		return goal.copy()
	
	def sample_goal_from_database(self):
		# db random with frite parameters
		self.goal_with_frite_parameters = self.database.get_random_targets_with_frite_parameters()
		self.set_env_with_frite_parameters(self.goal_with_frite_parameters[0])
		#print("goal_with_frite_parameters = {}".format(self.goal_with_frite_parameters[0]))
		goal = self.goal_with_frite_parameters[1]
		#print("goal = {}".format(goal))
		
		return goal
	
	# set copy of goal	
	def set_goal(self, a_goal):
		self.goal = np.copy(a_goal)
	
	# *******************************************************************************************************************
	
	# DRAW **************************************************************************************************************
	
	def draw_frite_parameters(self):
		str_to_print = "E=" + str(self.E) + ", NU=" + str(self.NU) + ", timeStep=" + str(self.env_pybullet.time_step) + ", factor_dt_factor=" + str(self.factor_dt_factor)
		str_to_print += ", orien=" + str(self.gripper_orientation_to_add)
		self.debug_gui.draw_text("frite_parameters", a_text=str_to_print, a_pos=[1,1,1], a_size=1.0)
		
	def draw_id_to_follow(self):
		for i in range(len(self.id_frite_to_follow)):
			self.debug_gui.draw_cross("id_frite_"+str(i), a_pos = self.position_mesh_to_follow[i], a_color = [0, 0, 1])
			#self.debug_gui.draw_text("text_id_frite_"+str(i), a_text = str(i), a_pos = self.position_mesh_to_follow[i], a_color = [0, 0, 1])

	def draw_normal_plane(self, index, data, a_normal_pt):
		# self.id_frite_to_follow[index][0] -> upper left
		# self.id_frite_to_follow[index][1] -> upper right
		# self.under_id_frite_to_follow[index][0] -> under left
		# self.under_id_frite_to_follow[index][1] -> under right
		
		# Draw a square by using upper (left/right) and under (left/right) points
		self.debug_gui.draw_line(name="l_"+str(index)+"_up",a_pos_from = data[1][self.id_frite_to_follow[index][0]], a_pos_to = data[1][self.id_frite_to_follow[index][1]])
		self.debug_gui.draw_line(name="l_"+str(index)+"_bottom",a_pos_from = data[1][self.under_id_frite_to_follow[index][0]], a_pos_to = data[1][self.under_id_frite_to_follow[index][1]])
		self.debug_gui.draw_line(name="l_"+str(index)+"_left",a_pos_from = data[1][self.id_frite_to_follow[index][0]], a_pos_to = data[1][self.under_id_frite_to_follow[index][0]])
		self.debug_gui.draw_line(name="l "+str(index)+"_right",a_pos_from = data[1][self.id_frite_to_follow[index][1]], a_pos_to = data[1][self.under_id_frite_to_follow[index][1]])
		
		# Draw a line for the normal vector from the mean upper point
		self.debug_gui.draw_line(name="normal_"+str(index),a_pos_from = self.mean_position_to_follow[index], a_pos_to = a_normal_pt, a_color = [1, 1, 0])

	def draw_gripper_position(self):
		self.panda_arm.draw_cross_ee()
		
	def draw_env_box(self):
		if self.is_graphic_mode == True:
			self.debug_gui.draw_box("pos_graphic", self.pos_space_graphic.low, self.pos_space_graphic.high, [0, 0, 1])
			self.debug_gui.draw_box("goal_graphic", self.goal_space_graphic.low, self.goal_space_graphic.high, [1, 0, 0])
		else:
			self.debug_gui.draw_box("pos", self.pos_space.low, self.pos_space.high, [0, 0, 1])
			self.debug_gui.draw_box("goal", self.goal_space.low, self.goal_space.high, [1, 0, 0])

	def draw_env_box_pose(self):
		self.debug_gui.draw_box("pos", self.pos_space.low, self.pos_space.high, [0, 0, 1])
	
	def remove_env_box_pose(self):
		self.debug_gui.remove_box("pos")
			
	def draw_env_box_goal(self):
		self.debug_gui.draw_box("goal", self.goal_space.low, self.goal_space.high, [1, 0, 0])
	
	def remove_env_box_goal(self):
		self.debug_gui.remove_box("goal")
	
	def draw_goal(self):
		for i in range(self.goal.shape[0]):
			#print("draw_goal[{}]={}".format(i,self.goal[i]))
			self.debug_gui.draw_cross("goal_"+str(i) , a_pos = self.goal[i])
			#self.debug_gui.draw_text("text_goal_"+str(i), a_text = str(i), a_pos = self.goal[i])


	# ***********************************************************************************************************************
	
	# FRITE *****************************************************************************************************************
	
	# Shift 'pt_mean' point with a distance 'a_distance' 
	# using a normal vector calculated from 3 points : 'pt_left', 'pt_right', 'pt_mean'. 
	def shift_point_in_normal_direction(self, pt_left, pt_right, pt_mean, a_distance = 0.2):
		# under left and right points
		vleft = np.array([pt_left[0], pt_left[1], pt_left[2]])
		vright = np.array([pt_right[0], pt_right[1], pt_right[2]])
		
		# upper mean point
		vmean = np.array([pt_mean[0], pt_mean[1], pt_mean[2]])
		
		# calculate the normal vector using the cross product of two (arrays of) vectors.
		vnormal = np.cross(vleft-vmean, vright-vmean)
		
		# calculate the norm of the normal vector
		norm_of_vnormal = np.linalg.norm(vnormal)
		
		# Normalize the normal vector 
		vnormal_normalized = vnormal / norm_of_vnormal
		
		# Shift the upper mean point of a distance by using the normal vector normalized 
		vmean_shifted = vmean + vnormal_normalized * a_distance
		
		return vmean_shifted
	
	def compute_mesh_pos_to_follow(self, draw_normal=False):
		self.mutex_get_mesh_data.acquire()
		try:
			data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		finally:
			self.mutex_get_mesh_data.release()
		
		# For all id to follow except the TIP
		for i in range(len(self.id_frite_to_follow)):
			
			# get left and right upper points 
			a_pt_left = np.array(data[1][self.id_frite_to_follow[i][0]])
			a_pt_right = np.array(data[1][self.id_frite_to_follow[i][1]])
			
			# calculate the upper mean (front side) between left and right upper points
			self.mean_position_to_follow[i] = (a_pt_left + a_pt_right)/2.0
			
			# get left and right under points
			a_pt_left_under = np.array(data[1][self.under_id_frite_to_follow[i][0]])
			a_pt_right_under = np.array(data[1][self.under_id_frite_to_follow[i][1]])
			
			# calculate the upper mean point shifted by a normalized normal vector.
			# The normal vector is calculated from a triangle defined by left+right under points and upper mean point.
			# 0.007 is equal to the half of the marker thickness
			self.position_mesh_to_follow[i] = self.shift_point_in_normal_direction(pt_left=a_pt_left_under, pt_right=a_pt_right_under, pt_mean=self.mean_position_to_follow[i], a_distance = 0.007)
			
			if draw_normal:
				a_normal_pt = self.shift_point_in_normal_direction(pt_left=a_pt_left_under, pt_right=a_pt_right_under, pt_mean=self.mean_position_to_follow[i], a_distance = 0.1)
				self.draw_normal_plane(i, data, a_normal_pt)
	
	
	def conv_module_d_young_to_lame(self, E, NU):
		a_lambda = (E * NU)/((1+NU)*(1-2*NU))
		a_mu = E/(2*(1+NU))
		
		return (a_lambda,a_mu)
	
	def get_position_id_frite_to_follow(self):
		data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		positions = []
		for i in range(len(self.id_frite_to_follow)):
			positions.append(data[1][self.id_frite_to_follow[i][0]])
		return positions
	
	def wait_until_frite_deform_ended(self):
		#print("*** wait_until_soft_deform_ended ***")
		finish = False
		 
		distances = np.array([0.0,0.0,0.0,0.0])
		positions_previous = self.get_position_id_frite_to_follow()
		positions_initial = positions_previous
		
		start=datetime.now()
		
		while not finish:
			
			positions_current = self.get_position_id_frite_to_follow()
			
			for i in range(len(positions_current)):
				distance = np.linalg.norm(np.array(positions_previous[i]) - np.array(positions_current[i]), axis=-1)
				distances[i] = distance
			
			max_distance = np.amax(distances)
			#print("max_distance = {}".format(max_distance))
			
			if max_distance != 0.0 and max_distance <= 0.0001:
				finish = True
			
			#print("total seconds = {}".format((datetime.now()-start).total_seconds()))
			if ((datetime.now()-start).total_seconds() > 60):
				finish = True
			
			positions_previous = positions_current
		
		time_elapsed = datetime.now()-start
		
		for i in range(len(positions_current)):
				distance = np.linalg.norm(np.array(positions_initial[i]) - np.array(positions_current[i]), axis=-1)
				distances[i] = distance
			
		max_distance = np.amax(distances)
		
		return max_distance,time_elapsed
	
	
	# *****************************************************************************************************************
	
		
	# PANDA ************************************************************************************************************
	
	def get_gripper_position(self):
		pos_ee, _ = self.panda_arm.ee_pose(to_euler=False)
		return pos_ee

	def go_to_gripper_orientation_bullet(self):
		self.compute_mesh_pos_to_follow(draw_normal=False)
		pos_ee, _ = self.panda_arm.ee_pose(to_euler=False)
		q_values, _, _, _ = self.panda_arm.calculateInverseKinematics(pos_ee, self.initial_orien_gripper, self.gripper_orientation_to_add)
		self.panda_arm.execute_qvalues(q_values)
		max_distance, time_elapsed = self.wait_until_frite_deform_ended()
			
	def set_panda_initial_joints_positions(self, init_gripper = True):
		self.panda_arm.set_initial_joints_positions(init_gripper)
		gripper_pos, gripper_orien = self.panda_arm.ee_pose(to_euler=False)
		print("*** PANDA Initial pos={}, orien={}".format(gripper_pos,gripper_orien))

	def close_gripper(self):
		self.panda_arm.close_gripper()
								
	def create_anchor_panda(self):
		
		# mesh id = 0 -> down
		# -1, -1 -> means anchor to the cube
		p.createSoftBodyAnchor(self.frite_id, 0, self.cube_id , -1, [0,0,0])
		p.createSoftBodyAnchor(self.frite_id, 5, self.cube_id , -1, [0,0,0])
		p.createSoftBodyAnchor(self.frite_id, 2, self.cube_id , -1, [0,0,0])
		
		p.createSoftBodyAnchor(self.frite_id, 4, self.cube_id , -1, [0,0,0])
		p.createSoftBodyAnchor(self.frite_id, 3, self.cube_id , -1, [0,0,0])
		#p.createSoftBodyAnchor(self.frite_id, 0, -1 , -1)
		
		# mesh id = 1 -> up
		p.createSoftBodyAnchor(self.frite_id, 1, self.panda_id , self.panda_end_eff_idx, [0,0,0])
		
		# panda finger joint 1 = 10
		# panda finger joint 2 = 11
		"""pos_10 = p.getLinkState(self.panda_id, 10)[0]
		new_pos_10  = [pos_10[0]+0.025, pos_10[1]-0.01, pos_10[2]-0.02]
		pos_11 = p.getLinkState(self.panda_id, 11)[0]
		new_pos_11  = [pos_11[0]+0.025, pos_11[1]+0.01, pos_11[2]-0.02]"""
		p.createSoftBodyAnchor(self.frite_id, 6, self.panda_id , 10, [0.025,-0.01,-0.02])
		p.createSoftBodyAnchor(self.frite_id, 9, self.panda_id , 11, [0.025,0.01,-0.02])
	
	
	# ********************************************************************************************************************
		
	# ROS ****************************************************************************************************************
	
	def to_rt_matrix(self,Q, T):
		# Q = geometry_msgs/Quaternion -> x,y,z,w
		# T = geometry_msgs/Point -> x,y,z
		# Extract the values from Q
		qw = Q.w
		qx = Q.x
		qy = Q.y
		qz = Q.z

		d = qx*qx + qy*qy + qz*qz + qw*qw
		s = 2.0 / d

		xs = qx * s
		ys = qy * s
		zs = qz * s
		wx = qw * xs
		wy = qw * ys
		wz = qw * zs
		xx = qx * xs
		xy = qx * ys
		xz = qx * zs
		yy = qy * ys
		yz = qy * zs
		zz = qz * zs


		r00 = 1.0 - (yy + zz)
		r01 = xy - wz
		r02 = xz + wy

		r10 = xy + wz
		r11 = 1.0 - (xx + zz)
		r12 = yz - wx

		r20 = xz - wy
		r21 = yz + wx
		r22 = 1.0 - (xx + yy)


		# 4x4 RT matrix
		rt_matrix = np.array([[r00, r01, r02, T.x],
							   [r10, r11, r12, T.y],
							   [r20, r21, r22, T.z],
							   [0, 0, 0, 1]])
								
		return rt_matrix
	
	def publish_initial_mesh_pos(self):
		intial_mesh_pos_array = np.array([[0.586, 0.000, -0.225],[0.586, 0.000, 0.032],[0.586, 0.000, 0.288],[0.586, 0.000, 0.481]])
		
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []
		
		for i in range(len(intial_mesh_pos_array)):
			simple_marker_msg = Marker()
			simple_marker_msg.header.frame_id = "panda_link0"
			simple_marker_msg.header.stamp = rospy.get_rostime()
			simple_marker_msg.ns = "intial_mesh_pos_in_arm_frame"
			simple_marker_msg.action = simple_marker_msg.ADD

			simple_marker_msg.type = simple_marker_msg.SPHERE
			simple_marker_msg.scale.x = self.marker_sphere_scale_x
			simple_marker_msg.scale.y = self.marker_sphere_scale_y
			simple_marker_msg.scale.z = self.marker_sphere_scale_z
			simple_marker_msg.color.g = 1.0
			simple_marker_msg.color.a = 1.0
			simple_marker_msg.id = i+40
			
			simple_marker_msg.pose.position.x = intial_mesh_pos_array[i][0]
			simple_marker_msg.pose.position.y = intial_mesh_pos_array[i][1]
			simple_marker_msg.pose.position.z = intial_mesh_pos_array[i][2]
			simple_marker_msg.pose.orientation.x = 0
			simple_marker_msg.pose.orientation.y = 0
			simple_marker_msg.pose.orientation.z = 0
			simple_marker_msg.pose.orientation.w = 1
			
			marker_array_msg.markers.append(simple_marker_msg)
		
		self.publisher_intial_mesh_pos_in_arm_frame.publish(marker_array_msg)
	
	def publish_deformation_status_text(self, text_to_publish):
		simple_marker_msg = Marker()
		
		simple_marker_msg.header.frame_id = "panda_link0"
		simple_marker_msg.header.stamp = rospy.get_rostime()
		simple_marker_msg.ns = "deformation_status_text"
		simple_marker_msg.action = simple_marker_msg.ADD

		simple_marker_msg.type = simple_marker_msg.TEXT_VIEW_FACING
		simple_marker_msg.scale.x = 0.1
		simple_marker_msg.scale.y = 0.1
		simple_marker_msg.scale.z = 0.1
		simple_marker_msg.color.g = 1.0
		simple_marker_msg.color.a = 1.0
		simple_marker_msg.id = 100
		
		simple_marker_msg.pose.position.x = 0.85
		simple_marker_msg.pose.position.y = 0.5
		simple_marker_msg.pose.position.z = 0.6
		simple_marker_msg.pose.orientation.x = 0
		simple_marker_msg.pose.orientation.y = 0
		simple_marker_msg.pose.orientation.z = 0
		simple_marker_msg.pose.orientation.w = 1
		
		simple_marker_msg.text = text_to_publish
			
		self.publisher_deformation_status_text.publish(simple_marker_msg)
	
	def publish_mocap_mesh(self):
		# publish mocap meshes into arm frame
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []
		
		for i in range(1,len(self.poses_meshes_in_arm_frame),1):
			simple_marker_msg = Marker()
			simple_marker_msg.header.frame_id = "panda_link0"
			simple_marker_msg.header.stamp = rospy.get_rostime()
			simple_marker_msg.ns = "points_and_lines_in_arm_frame"
			simple_marker_msg.action = simple_marker_msg.ADD

			simple_marker_msg.type = simple_marker_msg.SPHERE
			simple_marker_msg.scale.x = self.marker_sphere_scale_x
			simple_marker_msg.scale.y = self.marker_sphere_scale_y
			simple_marker_msg.scale.z = self.marker_sphere_scale_z
			simple_marker_msg.color.b = 1.0
			simple_marker_msg.color.g = 1.0
			simple_marker_msg.color.a = 1.0
			simple_marker_msg.id = i
			
			simple_marker_msg.pose.position.x = self.poses_meshes_in_arm_frame[i][0]
			simple_marker_msg.pose.position.y = self.poses_meshes_in_arm_frame[i][1]
			simple_marker_msg.pose.position.z = self.poses_meshes_in_arm_frame[i][2]
			simple_marker_msg.pose.orientation.x = 0
			simple_marker_msg.pose.orientation.y = 0
			simple_marker_msg.pose.orientation.z = 0
			simple_marker_msg.pose.orientation.w = 1
			
			marker_array_msg.markers.append(simple_marker_msg)
			
		self.publisher_poses_meshes_in_arm_frame.publish(marker_array_msg)
		
	def publish_goal(self):
		simple_marker_msg = Marker()
		marker_array_msg = MarkerArray()
		marker_array_msg.markers = []

		for i in range(len(self.goal)):
			simple_marker_msg = Marker()
			simple_marker_msg.header.frame_id = "panda_link0"
			simple_marker_msg.header.stamp = rospy.get_rostime()
			simple_marker_msg.ns = "goal_pos_in_arm_frame"
			simple_marker_msg.action = simple_marker_msg.ADD

			simple_marker_msg.type = simple_marker_msg.SPHERE
			simple_marker_msg.scale.x = self.marker_sphere_scale_x
			simple_marker_msg.scale.y = self.marker_sphere_scale_y
			simple_marker_msg.scale.z = self.marker_sphere_scale_z
			simple_marker_msg.color.g = 1.0
			simple_marker_msg.color.a = 1.0
			simple_marker_msg.id = i+50
			
			simple_marker_msg.pose.position.x = self.goal[i][0]
			simple_marker_msg.pose.position.y = self.goal[i][1]
			simple_marker_msg.pose.position.z = self.goal[i][2]
			simple_marker_msg.pose.orientation.x = 0
			simple_marker_msg.pose.orientation.y = 0
			simple_marker_msg.pose.orientation.z = 0
			simple_marker_msg.pose.orientation.w = 1
			
			marker_array_msg.markers.append(simple_marker_msg)

		self.publisher_goal_in_arm_frame.publish(marker_array_msg)
	
	def joint_states_callback(self, msg):
		# msg = sensor_msgs::JointState
		self.mutex_joint_states.acquire()
		try:
			self.array_joint_states = np.array(msg.position)
		finally:
			self.mutex_joint_states.release()

	def mocap_callback(self, msg):
		# msg = geometry_msgs::PoseArray
		self.mutex_array_mocap.acquire()
		try:
			self.array_mocap_poses_base_frame = np.array(msg.poses)
		finally:
			self.mutex_array_mocap.release()
		
		# index 0 contain the base
		pos_base_frame = msg.poses[0].position
		orien_base_frame = msg.poses[0].orientation

		self.matrix_base_frame_in_mocap_frame = self.to_rt_matrix(orien_base_frame, pos_base_frame)

		self.matrix_mocap_frame_in_arm_frame = np.dot(self.matrix_base_frame_in_arm_frame, LA.inv(self.matrix_base_frame_in_mocap_frame))

		self.mutex_array_mocap_in_arm_frame.acquire()
		try:
			for i in range(len(self.id_frite_to_follow)+1):
				pos_mesh_in_mocap_frame = np.array([msg.poses[i].position.x,msg.poses[i].position.y,msg.poses[i].position.z,1])
				self.poses_meshes_in_arm_frame[i] = np.dot(self.matrix_mocap_frame_in_arm_frame, pos_mesh_in_mocap_frame)
		finally:
			self.mutex_array_mocap_in_arm_frame.release()	
		
		# publish current goal
		self.publish_goal()
				
		# publish mocap meshes into arm frame
		self.publish_mocap_mesh()
		
		if self.publish_init_pos_mesh:
			self.publish_initial_mesh_pos()
	
	def observation_callback(self, msg):
		self.mutex_observation.acquire()
		try:
			
			tip_position_list = [msg.data[0], msg.data[1], msg.data[2]]
			self.tip_position = np.array(tip_position_list)
			
			tip_velocity_list = [msg.data[3], msg.data[4], msg.data[5]]
			self.tip_velocity = np.array(tip_velocity_list)
			
			tip_orientation_list = p.getEulerFromQuaternion([msg.data[6], msg.data[7], msg.data[8], msg.data[9]])
			self.tip_orientation = np.array(tip_orientation_list)
			
			tip_velocity_orientation_list = [msg.data[10], msg.data[11], msg.data[12]]
			self.tip_velocity_orientation = np.array(tip_velocity_orientation_list)
			
			
		finally:
			self.mutex_observation.release()	

	
	def convertToJointStateMsg(self,positions):
		msg = Float64MultiArray()
		#msg.header = Header()
		#msg.name = ['joint1','joint2','joint3','joint4','joint5','joint6','joint7']
		msg.data = [positions[0],positions[1],positions[2],positions[3],positions[4],positions[5],positions[6]]
		return msg
	
	def publish_joint_position(self, positions):
		position_msg = self.convertToJointStateMsg(positions)
		self.publisher_joint_position.publish(position_msg)

	def wait_until_q_values_reached_ros(self, q_values_desired):
		# wait until reached q_values desired positions
		start=datetime.now()
		
		while True:
			self.mutex_joint_states.acquire()
			try:
				current_array_joint_states = self.array_joint_states[0:7].copy()
			finally:
				self.mutex_joint_states.release()
			
			#print("current_array_joint_states={}, q_values_desired={}".format(current_array_joint_states,q_values_desired))
			
			if np.linalg.norm(current_array_joint_states - q_values_desired.flatten(), axis=-1) <= 0.01:
				print("Exit with precision = {} !".format(np.linalg.norm(current_array_joint_states - q_values_desired.flatten(), axis=-1)))
				print("With {} seconds".format((datetime.now()-start).total_seconds()))
				break
			
			if (datetime.now()-start).total_seconds() >= 100:
				print("Exit with time !")
				print("With {} seconds".format((datetime.now()-start).total_seconds()))
				break
				
		time_elapsed = datetime.now()-start
		precision = np.linalg.norm(current_array_joint_states - q_values_desired.flatten(), axis=-1)
		
		return precision,time_elapsed
	
	def go_to_home_position_ros(self):
		q_values, _, _, is_inside_limits = self.panda_arm.calculateInverseKinematics(self.initial_pos_gripper, self.initial_orien_gripper, None)
		if is_inside_limits:
			self.publish_joint_position(q_values)
			self.wait_until_q_values_reached_ros(q_values)
		else:
			print("### OUT OF PANDA LIMITS : pos={}, orien={}".format(pos_ee, self.initial_orien_gripper))
			
		
	def go_to_gripper_orientation_ros(self):
		pos_ee = self.initial_pos_gripper
		q_values, _, _, is_inside_limits = self.panda_arm.calculateInverseKinematics(pos_ee, self.initial_orien_gripper, self.gripper_orientation_to_add)
		if is_inside_limits:
			self.publish_joint_position(q_values)
			self.wait_until_q_values_reached_ros(q_values)
		else:
			print("### OUT OF PANDA LIMITS : pos={}, orien={}, orien_to_add={}".format(pos_ee, self.initial_orien_gripper, self.gripper_orientation_to_add))
			self.env_file_log.write("rank: {}, OUT OF POS BOX: pos={}, orien={}, orien_to_add={}, q_values={}\n".format(self.rank,pos_ee,self.initial_orien_gripper, self.gripper_orientation_to_add,q_values))
			self.env_file_log.flush()
		
	def update_gripper_orientation_ros(self):
		if self.is_gripper_orien_from_initial():
			self.gripper_orientation_to_add = [0.0,0.0,0.0]
			
		if self.is_gripper_orien_from_agent():
			state_rotation_gripper = np.concatenate((np.array([0.0,0.0,0.0]), self.goal.flatten(), self.desired_initial_gripper_orientation))
			self.gripper_orientation_to_add = self.agent_rotation_gripper.get_action(state_rotation_gripper)
			print("rotation gripper from agent = {}".format(self.gripper_orientation_to_add))
		
		self.go_to_gripper_orientation_ros()

	def go_to_cartesian_ros(self, pos, orien, euler_angle_to_add):
		pos_truncated = self.truncate_array(pos,3)
		q_values, _, _, is_inside_limits =  self.panda_arm.calculateInverseKinematics(pos_truncated, orien, euler_angle_to_add)
		
		pos_fk = self._ik_dh.forward_K(q_values)
		is_inside_box = self.is_inside_pos_space(pos_fk)
		d =  np.linalg.norm(pos_truncated - pos_fk, axis=-1)
		
		if d > 0.01:
			print("### IK PRECISION: pos desired={}, pos fk={}\n".format(pos_truncated, pos_fk))
			self.env_file_log.write("rank: {}, IK PRECISION: pos desired={}, pos fk={}\n".format(self.rank, pos_truncated, pos_fk))
			self.env_file_log.flush()
		
		if is_inside_limits:
			if is_inside_box:
				if d > 0.01:
					print("### OUT OF DISTANCE: d={}\n".format(d))
					self.env_file_log.write("rank: {}, OUT OF DISTANCE: d={}\n".format(self.rank, d))
					self.env_file_log.flush()
				else:
					self.publish_joint_position(q_values)
					precision, time_elapsed = self.wait_until_q_values_reached_ros(q_values)
					print("*** cartesian pos reached : time elapsed={}, precision={}".format(time_elapsed, precision))
			else:
				print("### OUT OF BOX !\n")
				self.env_file_log.write("rank: {}, OUT OF BOX !\n".format(self.rank))
				self.env_file_log.flush()
		else:
			print("### OUT OF LIMITS: pos={}, orien={}, q_values={}\n".format(pos_truncated,orien,q_values))
			self.env_file_log.write("rank: {}, OUT OF LIMITS: pos={}, orien={}, q_values={}\n".format(self.rank,pos_truncated,orien,q_values))
			self.env_file_log.flush()
			
		return q_values

	def set_action_ros(self, action):
		assert action.shape == (self.nb_action_values,), 'action shape error'
		
		self.mutex_observation.acquire()
		try:
			
			tip_position = self.tip_position.copy()
			
			tip_orientation = self.tip_orientation.copy()  # Euler gripper orientation
			
		finally:
			self.mutex_observation.release()
			
		new_pos = tip_position + np.array(action[:3]) * self.max_vel * self.dt
		new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
		
		if self.is_action_3D():
			new_orien_quaternion = self.initial_orien_gripper
		
		if self.is_action_6D():
			new_orien_euler = tip_orientation + np.array(action[3:]) * self.max_vel * self.dt
			# clip new euler orientation x,y,z from initial [-3.1415,0,0] added with a max range [-0.5..0.5] on each angle.
			new_orien_euler = np.clip(new_orien_euler, [-3.1415-0.5, -0.5, -0.5], [-3.1415+0.5, 0.5, 0.5])
			new_orien_quaternion = p.getQuaternionFromEuler(new_orien_euler)

		self.go_to_cartesian_ros(new_pos, new_orien_quaternion, self.gripper_orientation_to_add)
		

	def get_position_orientation_gripper_ros(self):
		self.mutex_observation.acquire()
		try:
			
			tip_position = self.tip_position.copy()
			
			tip_orientation = self.tip_orientation.copy()
			
		finally:
			self.mutex_observation.release()
			
		obs = np.concatenate((tip_position, tip_orientation))
		
		return obs
		

	def get_obs_ros(self):
		
		self.mutex_observation.acquire()
		try:
			
			tip_position = self.tip_position.copy()
			
			tip_velocity = self.tip_velocity.copy()
			
			tip_orientation = self.tip_orientation.copy()
			
			tip_velocity_orientation = self.tip_velocity_orientation.copy()
			
		finally:
			self.mutex_observation.release()
			
			
		self.mutex_array_mocap_in_arm_frame.acquire()
		try:
			poses_meshes_in_arm_frame = self.poses_meshes_in_arm_frame[1:5,0:3].copy()
			
		finally:
			self.mutex_array_mocap_in_arm_frame.release()	
		
		#print("----->", poses_meshes_in_arm_frame.flatten())
		
		if self.is_action_3D():
			obs = np.concatenate((tip_position, tip_velocity, poses_meshes_in_arm_frame.flatten().astype('float64'), self.goal.flatten()))
		
		if self.is_action_6D():
			obs = np.concatenate((
							tip_position, tip_orientation, tip_velocity, tip_velocity_orientation, poses_meshes_in_arm_frame.flatten().astype('float64'), self.goal.flatten())
							)	
		
		if self.add_frite_parameters_to_observation:
			# with frite parameters
			obs = np.concatenate(( obs, np.array([float(self.E), float(self.NU)]) ))
			
		return obs

	def step_ros(self, action, rank=None, episode=None, step=None):
		action = np.clip(action, self.action_space.low, self.action_space.high)
		self.set_action_ros(action)
		
		time.sleep(self.time_set_action)
		
		obs = self.get_obs_ros()

		reward = -1
		done = False
		
		info = {
			'is_success': False,
			'distance_error' : -1,
		}
		
		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info


	def reset_ros(self):
		
		# sample a goal from database, set env frite parameters, update gym spaces
		self.goal = self.sample_goal_from_database()
			
		self.update_gripper_orientation_ros()
				
		return self.get_obs_ros()
	
	def set_marker_sphere_scale(self, a_size):
		self.marker_sphere_scale_x = a_size
		self.marker_sphere_scale_y = a_size
		self.marker_sphere_scale_z = a_size
			
	def init_ros(self):
		# init ros node
		print("*** INIT ROS !!!!!!!!!!!!")
		rospy.init_node('rl_melodie_node')
		
		# set size of sphere marker
		self.set_marker_sphere_scale(0.05)
		
		# set gym spaces, calculate pos_of_mesh_in_obs, nb_obs_values, goal_space, pos_space, action_space
		self.set_gym_spaces()
		
		self.publish_init_pos_mesh = self.json_decoder.config_data["env_test"]["real"]["publish_init_pos_mesh"]
		
		# init mutexes for threading
		self.mutex_observation = threading.Lock()
		self.mutex_array_mocap = threading.Lock()
		self.mutex_array_mocap_in_arm_frame = threading.Lock()
		self.mutex_joint_states = threading.Lock()
		
		#load panda, needed for IK
		self.load_panda()

		# set panda joints to initial positions
		self.set_panda_initial_joints_positions()

		# Numpy array that content the Poses and orientation of each rigid body defined with the mocap
		# [[mesh0 geometry_msgs/Pose ], [mesh1 geometry_msgs/Pose] ..., [mesh n geometry_msgs/Pose]]
		self.array_mocap_poses_base_frame = None
		
		self.matrix_base_frame_in_mocap_frame = None
		self.matrix_mocap_frame_in_arm_frame = None
		
		# array of panda robot joint_states
		self.array_joint_states = None
										
		# x = 0.186 cm, y = 0.27 cm  z = 0.013 cm								
		self.matrix_base_frame_in_arm_frame = np.array(
											[[1, 0, 0, 0.186],
											[0, 1, 0, 0.27],
											[0, 0, 1, 0.013],
											[0, 0, 0, 1]]
										)
										
		# 4*[None] -> X,Y,Z,1
		self.poses_meshes_in_arm_frame = np.array((len(self.id_frite_to_follow)+1) * [[None]*4])

		# sample a goal from database, set env frite parameters, update gym spaces
		self.goal = self.sample_goal_from_database()
		
		# topics :
		# =======
		# Panda Arm is loaded into (0,0,0) of the world
		# (published from mocap) '/PoseAllBodies' -> positions and orientations of real frite markers
		# (published from env) '/VisualizationGoalArrayMarkersInArmFrame' -> msgs Marker that contains SPHERE of GOAL positions in arm frame
		# (published from env) '/VisualizationPoseArrayMarkersInArmFrame' -> msgs Marker that contains SPHERE of frite markers in arm frame
		# key json : ["env_test"]["real"]["publish_init_pos_mesh"] -> publish init poses of meshes
		# (published from env) ''/VisualizationInitialPoseArrayMarkersInArmFrame'' -> msgs Marker that contains SPHERE of hard coded 'initial' positions of frite markers
		# (published from franka controller) '/joint_position_controller_ip/current_observation_orientation' 
		# End Effector (tip) : [pos_x, pos_y, pos_z, pos_vel_x, pos_vel_y, pos_vel_z, orien_x, orien_y, orien_z, orien_w, orien_vel_x, orien_vel_y, orien_vel_z]      
		# (published from franka) '/joint_states' -> joints values of real robot
		# subcribe to mocap topic named '/PoseAllBodies'
		rospy.Subscriber('/PoseAllBodies', PoseArray, self.mocap_callback, queue_size=10)
		
		# subscribe to topic '/joint_states'
		rospy.Subscriber('/joint_states', JointState, self.joint_states_callback, queue_size=10)
		
		# publisher mocap meshes into arm frame
		self.publisher_poses_meshes_in_arm_frame = rospy.Publisher('/VisualizationPoseArrayMarkersInArmFrame', MarkerArray, queue_size=10)
		
		# publisher initial pose of frite marker into arm frame
		self.publisher_intial_mesh_pos_in_arm_frame = rospy.Publisher('/VisualizationInitialPoseArrayMarkersInArmFrame', MarkerArray, queue_size=10)

		# publisher goal into arm frame
		self.publisher_goal_in_arm_frame = rospy.Publisher('/VisualizationGoalArrayMarkersInArmFrame', MarkerArray, queue_size=10)

		# publisher of panda robot position controller command 
		self.publisher_joint_position = rospy.Publisher('/joint_position_controller_ip/command', Float64MultiArray, queue_size=1)

		# publisher of deformation status text
		self.publisher_deformation_status_text = rospy.Publisher('/VisualizationDeformationStatusText', Marker, queue_size=10)
		
		# subscriber of panda robot position controller observation topic
		rospy.Subscriber('/joint_position_controller_ip/current_observation_orientation', Float64MultiArray, self.observation_callback, queue_size=10)

	# ROS *****************************************************************************************************************


