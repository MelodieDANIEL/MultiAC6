import os, inspect
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

import math
from numpy import linalg as LA

class PandaFriteEnvRotationGripper(gym.Env):
	
	def __init__(self, database = None, json_decoder = None, env_rank=None):
		
		if json_decoder==None:
			raise RuntimeError("=> PandaFriteEnvComplete class need a JSON Decoder, to get some parameters !!!")
			return
	
		print("****** PandaFriteEnvComplete !!!! ************")
		
		# init public class properties
		self.rank = env_rank
		self.json_decoder = json_decoder
		self.database = database
		
		# open log file
		self.env_file_log = open(self.json_decoder.config_dir_name + 'log_env_rank'+ str(self.rank) + '.txt', "w+")
		self.env_file_log.write("rank = {}\n".format(self.rank))
	
		# read JSON env properties
		self.env_random_seed = self.json_decoder.config_data["env"]["random_seed"]
		self.env_file_log.write("env_random_seed = {}\n".format(self.env_random_seed))
		
		self.distance_threshold = self.json_decoder.config_data["env"]["reward_parameters"]["distance_threshold"]
		self.env_file_log.write("distance_threshold={}\n".format(self.distance_threshold))
		self.env_file_log.flush()
		
		# id frite to follow
		self.id_frite_to_follow = self.json_decoder.config_data["env"]["frite_parameters"]["id_frite_to_follow"]
		
		# nb frite to follow
		self.nb_frite_to_follow = len(json_decoder.config_data["env"]["frite_parameters"]["id_frite_to_follow"])
	
		# set env to database and load datas
		self.database.set_env(self)
		self.database.load()
		
		# random properties
		self.seed(self.env_random_seed)
		
		# reset env bullet
		self.reset_env_bullet()
		
	def reset_env_bullet(self):
		# set gym spaces
		self.set_gym_spaces()
		
	def get_nb_action_values(self):
		return self.nb_action_values
		
	def get_nb_obs_values(self):
		return self.nb_observation_values
		
	def get_action_space_low(self):
		return self.action_space.low
		
	def get_action_space_high(self):
		return self.action_space.high
		
	def set_gym_spaces(self):
		
		# action = 3 floats (theta_x, theta_y, theta_z) = euler_angles_to_add	
		# action_space of gripper : 3 actions (theta_x, theta_y, theta_z) = euler_angles_to_add
		self.nb_action_values = 3
		
		# action space
		self.action_space = spaces.Box(-1., 1., shape=(self.nb_action_values,), dtype=np.float32)
		
		# observation = 
		#  3 floats (theta_x,theta_y,theta_z) current euler angles to add to the gripper tip orientation [0,1,2]
		# + self.goal cartesian world position of id frite to reach (12 floats in case of controlling 4 mesh nodes) [3,4,5,6,7,8,9,10,11,12,13,14]
		# + 3 floats (theta_x,theta_y,theta_z) desired euler angles to add to the gripper tip orientation [15,16,17]
		# observation = 18 floats (in case of controlling 4 mesh nodes)
		
		current_nb_euler_angles = 3
		desired_nb_euler_angles = 3
		
		self.nb_observation_values = current_nb_euler_angles + 3 * self.nb_frite_to_follow + desired_nb_euler_angles
		
		# observation space
		self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(self.nb_observation_values,), dtype=np.float32)

	def sample_goal_from_database(self):
		# db random with frite parameters
		self.goal_with_frite_parameters = self.database.get_random_targets_with_frite_parameters()
		deformation_parameters = self.goal_with_frite_parameters[0]
		#print("goal_with_frite_parameters = {}".format(self.goal_with_frite_parameters[0]))
		goal = self.goal_with_frite_parameters[1]
		#print("goal = {}".format(goal))
		desired_euler_angles_to_add = np.array([float(deformation_parameters[4]), float(deformation_parameters[5]), float(deformation_parameters[6])])
		
		return goal, desired_euler_angles_to_add

	def get_obs_bullet(self, current_euler_angles_to_add):
		
		# observation = 
		#  3 floats (theta_x,theta_y,theta_z) current euler angles to add to the gripper tip orientation [0,1,2]
		# + self.goal cartesian world position of id frite to reach (12 floats in case of controlling 4 mesh nodes) [3,4,5,6,7,8,9,10,11,12,13,14]
		# + 3 floats (theta_x,theta_y,theta_z) desired euler angles to add to the gripper tip orientation [15,16,17]
		# observation = 18 floats (in case of controlling 4 mesh nodes)
		
		obs = np.concatenate((current_euler_angles_to_add, self.goal.flatten(), self.desired_euler_angles_to_add))
		
		return obs
	
	def reset_bullet(self):
		# sample a new goal
		self.goal, self.desired_euler_angles_to_add = self.sample_goal_from_database()
		current_euler_angles_to_add = np.array([0,0,0])
		
		return self.get_obs_bullet(current_euler_angles_to_add)

	def is_success(self, d):
		return (d < self.distance_threshold).astype(np.float32)
		
	def step_bullet(self, action, rank=None, episode=None, step=None):
		current_euler_angles_to_add = np.clip(action, self.action_space.low, self.action_space.high)
		
		obs = self.get_obs_bullet(current_euler_angles_to_add)

		done = True
		
		d = np.linalg.norm(current_euler_angles_to_add - self.desired_euler_angles_to_add, axis=-1)

		info = {
			'is_success': self.is_success(d),
			'distance_error' : d,
		}

		reward = -d
		if (d > self.distance_threshold):
			done = False

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info

	def render(self):
		print("PandaFriteEnvRotationGripper -> render !")
		
	def seed(self, seed=None):
		seed_seq = np.random.SeedSequence(seed)
		np_seed = seed_seq.entropy
		self.np_random = np.random.Generator(np.random.PCG64(seed_seq))
		return [np_seed]
		
