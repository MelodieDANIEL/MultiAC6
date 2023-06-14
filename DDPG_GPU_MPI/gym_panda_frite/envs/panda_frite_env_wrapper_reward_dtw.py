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


from gym import Wrapper
import numpy as np

class PandaFriteEnvWrapperRewardDTW(Wrapper):
	
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		print("$$$$$ CONSTRUCTOR PandaFriteEnvWrapperRewardDTW $$$$$")
		
	def get_action_space_low(self):
		return self.env.get_action_space_low()
		
	def get_action_space_high(self):
		return self.env.get_action_space_high()
	
	def get_nb_action_values(self):
		return self.env.get_nb_action_values()
		
	def get_nb_obs_values(self):
		return self.env.get_nb_obs_values()
	
	def draw_env_box(self):
		self.env.draw_env_box()
	
	def draw_frite_parameters(self):
		self.env.draw_frite_parameters()
		
	def draw_id_to_follow(self):
		self.env.draw_id_to_follow()
		
	def seed(self, seed=None):
		self.env.seed(seed)
		
	def reset_env_bullet(self, use_frite=True):
		self.env.reset_env_bullet(use_frite)
		
	def reset_bullet(self):
		return self.env.reset_bullet()
		
	def step_bullet(self, action, rank=None, episode=None, step=None):
		obs, reward, done, info = self.env.step_bullet(action,rank,episode,step)
		
		done = True
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		sum_d = 0
		exploded = False
		index_exploded = -1
		value_exploded = 0.0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.env.pos_of_mesh_in_obs+(i*3)):(self.env.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			
			if (d > 2.0):
				index_exploded = i
				value_exploded = d
				exploded = True
				break	
			
			sum_d+=d
		
		sum_d = np.float32(sum_d)
		
		info = {
			'is_success': self.env.is_success(sum_d),
			'distance_error' : sum_d,
			'exploded' : exploded,
			'index_exploded' : index_exploded,
			'value_exploded' : value_exploded,
			
		}

		reward = -sum_d
		
		"""
		if exploded:
			reward = self.env.previous_reward
		else:
			self.env.previous_reward = reward
		"""
		
		if (sum_d > self.env.distance_threshold):
			done = False
			
		print("$$$$$ step_bullet PandaFriteEnvWrapperRewardDTW $$$$$")
		print("rank={}, episode={}, step={}, exploded={}, index_exploded={}, value_exploded={}".format(rank,episode,step,exploded,index_exploded,value_exploded))
		print("obs={}".format(obs))
		print("reward={}, done={}, info={}".format(reward,done,info))
		print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

		# p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
		return obs, reward, done, info
	
	# ROS
	
	def reset_ros():
		return self.env.reset_ros()
		
	def step_ros(self, action, rank=None, episode=None, step=None):
		obs, reward, done, info = self.env.step_ros(action,rank,episode,step)
		
		done = True
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		sum_d = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.env.pos_of_mesh_in_obs+(i*3)):(self.env.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			sum_d+=d
		
		sum_d = np.float32(sum_d)
		
		info = {
			'is_success': self.env.is_success(sum_d),
			'distance_error' : sum_d,
		}

		reward = -sum_d
		
		if (sum_d > self.env.distance_threshold):
			done = False
			
		print("$$$$$ step_ros PandaFriteEnvWrapperRewardDTW $$$$$")
		print("obs={}".format(obs))
		print("reward={}, done={}, info={}".format(reward,done,info))
		print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

		return obs, reward, done, info
