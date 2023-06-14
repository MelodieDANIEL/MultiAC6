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

import sys
import pybullet as p
import numpy as np
from numpy import random
import math
import time
from datetime import datetime

class Database_Frite:
	def __init__(self, json_decoder=None):
		
		if json_decoder==None:
			raise RuntimeError("=> Database_Frite class need a JSON Decoder, to get some parameters !!!")
			return
			
		self.load_name = json_decoder.config_data["database"]["name"]
		self.generate_name = 'generate_' + json_decoder.config_data["database"]["name"]
		self.path_load = json_decoder.config_dir_name
		self.path_generate = json_decoder.config_dir_name
		
		self.parameters_array = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"]
		
		self.nb_lines = 0
		self.nb_deformations = 0
		self.nb_frite_parameters = 0
		
		self.env = None
		self.data = None
		self.dico_data = None
		self.data_gripper = None
		self.dico_data_gripper = None
		self.nb_points = None
		
		self.print_config_db()
		
	def set_env(self, env):
		self.env = env
		self.nb_points = len(self.env.id_frite_to_follow)
	
	def debug_point(self, pt, offset = 0.1, width = 3.0, color = [1, 0, 0]):
		
		p.addUserDebugLine(lineFromXYZ          = [pt[0]+offset, pt[1], pt[2]]  ,
						   lineToXYZ            = [pt[0]-offset, pt[1], pt[2]],
						   lineColorRGB         = color  ,
						   lineWidth            = width        ,
						   lifeTime             = 0          )
						   
		p.addUserDebugLine(lineFromXYZ          = [pt[0], pt[1]+offset, pt[2]]  ,
						   lineToXYZ            = [pt[0], pt[1]-offset, pt[2]],
						   lineColorRGB         = color  ,
						   lineWidth            = width        ,
						   lifeTime             = 0          )
						   
	def debug_all_points(self):
		for i in range(self.nb_deformations):
			for j in range(self.nb_points):
				self.debug_point(pt=self.data[i,j], offset=0.01)
			
	def debug_all_random_points(self, nb):
		for i in range(nb):
			a_pt = self.get_random_targets()
			for j in range(self.nb_points):
				self.debug_point(pt = a_pt[j], offset =0.01, color = [0, 0, 1])
	
	def print_config_db(self):
		print("GENERATE DB FROM JSON 'database/frite_parameters/parameters_array' !")
		print("load name = {}, generate name = {}".format(self.load_name,self.generate_name))
		print("path load = {}, path_generate = {}".format(self.path_load, self.path_generate))
		
	def get_random_targets_with_frite_parameters(self):
		if self.dico_data is not None:
			"""
			OLD METHOD WITH RANDINT
			index_frite_parameters = random.randint(len(list(self.dico_data))-1)
			index_data = random.randint(self.nb_deformations-1)
			#print("get_random_targets -> index_frite_parameters={}, index_data={}".format(index_frite_parameters,index_data))
			#print("key random={}, data key random={}".format(list(self.dico_data)[index_frite_parameters],self.dico_data[list(self.dico_data)[index_frite_parameters]][index_data]))
			"""
			
			index_frite_parameters = self.env.np_random.integers(0,len(list(self.dico_data)))
			index_data = self.env.np_random.integers(0,self.nb_deformations)
			#print("get_random_targets -> index_frite_parameters={}, index_data={}".format(index_frite_parameters,index_data))
			#print("key random={}, data key random={}".format(list(self.dico_data)[index_frite_parameters],self.dico_data[list(self.dico_data)[index_frite_parameters]][index_data]))
			
			
			return (list(self.dico_data)[index_frite_parameters], self.dico_data[list(self.dico_data)[index_frite_parameters]][index_data])
		else:
			return None
			
	def get_random_targets(self):
		if self.data is not None:
			index = random.randint(self.nb_deformations-1)
			return self.data[index]
		else:
			return None
	
	def load(self):
		self.load_random_from_frite_parameters()
		
	def load_random_from_frite_parameters(self):
		print("=> LOAD DATABASE Name = {}".format(self.path_load + self.load_name))
		f = open(self.path_load + self.load_name)
		
		self.dico_data = {}
		self.dico_data_gripper = {}
		self.nb_frite_parameters = 0
		line = f.readline()
		while line:
			line_split = line.split()
			#print("line_split = {}".format(line_split))
			E = line_split[0]
			NU = line_split[1]
			time_step = line_split[2]
			factor_dt_factor = line_split[3]
			
			x_rot_gripper = line_split[4]
			y_rot_gripper = line_split[5]
			z_rot_gripper = line_split[6]
			
			#x_up_goal,x_down_goal,y_up_goal,y_down_goal,z_down_goal
			x_up_goal = line_split[7]
			x_down_goal = line_split[8]
			y_up_goal = line_split[9]
			y_down_goal = line_split[10]
			z_down_goal = line_split[11]
			
			#x_up_pos,x_down_pos,y_up_pos,y_down_pos,z_down_pos	
			x_up_pos = line_split[12]
			x_down_pos = line_split[13]
			y_up_pos = line_split[14]
			y_down_pos = line_split[15]
			z_down_pos = line_split[16]
			
			nb_random_goal = line_split[17]
			
			self.nb_frite_parameters+=1
			print("E={}, NU={}, TIMESTEP={}, factor_dt_factor={}".format(E,NU,time_step,factor_dt_factor))
			print("Gripper rot x={}, y={}, z={}".format(x_rot_gripper,y_rot_gripper,z_rot_gripper))
			print("Goal space x_up={}, x_down={}, y_up={}, y_down={}, z_down={}".format(x_up_goal,x_down_goal,y_up_goal,y_down_goal,z_down_goal))
			print("Pos space x_up={}, x_down={}, y_up={}, y_down={}, z_down={}".format(x_up_pos,x_down_pos,y_up_pos,y_down_pos,z_down_pos))
			print("Nb Goal random={}".format(nb_random_goal))
			
			line = f.readline()
			self.nb_lines = 0
			self.nb_deformations = 0
			total_list = []
			total_list_gripper = []

			while len(line.split()) == 9:
				line_split = line.split()
				self.nb_lines+=1
				# 0 = X mean, 1 = Y mean, 2 = Z mean
				total_list.append(float(line_split[3])) #3 = x shifted
				total_list.append(float(line_split[4])) #4= y shifted
				total_list.append(float(line_split[5]))	#5= z shifted
				
				total_list_gripper.append(float(line_split[6])) #6= gripper x
				total_list_gripper.append(float(line_split[7])) #7= gripper y
				total_list_gripper.append(float(line_split[8]))	#8= gripper z
				
				line = f.readline()
				
			self.nb_deformations = int(self.nb_lines/self.nb_points)
			print("nb lines = {}, nb points = {}, nb_deformations = {}".format(self.nb_lines, self.nb_points,self.nb_deformations))
			data = np.array(total_list).reshape(self.nb_deformations, self.nb_points, 3)
			data_gripper = np.array(total_list_gripper).reshape(self.nb_deformations, self.nb_points, 3)
			
			self.dico_data[(E,NU,time_step,factor_dt_factor,x_rot_gripper,y_rot_gripper,z_rot_gripper,x_up_goal,x_down_goal,y_up_goal,y_down_goal,z_down_goal,x_up_pos,x_down_pos,y_up_pos,y_down_pos,z_down_pos,nb_random_goal)] = data
			self.dico_data_gripper[(E,NU,time_step,factor_dt_factor,x_rot_gripper,y_rot_gripper,z_rot_gripper,x_up_goal,x_down_goal,y_up_goal,y_down_goal,z_down_goal,x_up_pos,x_down_pos,y_up_pos,y_down_pos,z_down_pos,nb_random_goal)] = data_gripper
				
		#print("dico data={}".format(self.dico_data))
		#print("dico_data_gripper={}".format(self.dico_data_gripper))
		
	def write_floats(self, f):
		self.env.compute_mesh_pos_to_follow(draw_normal=False)
		for k in range(len(self.env.position_mesh_to_follow)):
			f.write("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(self.env.mean_position_to_follow[k][0], self.env.mean_position_to_follow[k][1], self.env.mean_position_to_follow[k][2], self.env.position_mesh_to_follow[k][0],  self.env.position_mesh_to_follow[k][1], self.env.position_mesh_to_follow[k][2]))
		f.flush()
			
	def write_floats_with_gripper_pos(self, gripper_pos, f):
		self.env.compute_mesh_pos_to_follow(draw_normal=False)
		for k in range(len(self.env.position_mesh_to_follow)):
			f.write("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(self.env.mean_position_to_follow[k][0], self.env.mean_position_to_follow[k][1], self.env.mean_position_to_follow[k][2], self.env.position_mesh_to_follow[k][0],  self.env.position_mesh_to_follow[k][1], self.env.position_mesh_to_follow[k][2], gripper_pos[0], gripper_pos[1], gripper_pos[2]))
		f.flush()

	def write_gripper_floats(self, gripper_pos, f):
		f.write("{:.5f} {:.5f} {:.5f}\n".format(gripper_pos[0], gripper_pos[1], gripper_pos[2]))
		f.flush()			

	def generate(self):
		self.generate_random_from_frite_parameters()
		
	def generate_random_from_frite_parameters(self):
		start_time_generate = datetime.now()
		
		print("*** Create database file : ", self.path_generate + self.generate_name , " !")
		f = open(self.path_generate + self.generate_name, "w+")
		
		for parameter in self.parameters_array:
			E = float(parameter["E"])
			NU = float(parameter["NU"])
			print("E = {}, NU = {}".format(E, NU))
			
			time_step = float(parameter["time_step"])
			factor_dt_factor = float(parameter["factor_dt_factor"])
			
			x_rot = float(parameter["x_rot"])
			y_rot = float(parameter["y_rot"])
			z_rot = float(parameter["z_rot"])
			
			x_up_goal = float(parameter["x_up_goal"])
			x_down_goal = float(parameter["x_down_goal"])
			y_up_goal = float(parameter["y_up_goal"])
			y_down_goal = float(parameter["y_down_goal"])
			z_down_goal = float(parameter["z_down_goal"])
			
			goal_dims = np.array([x_up_goal, x_down_goal, y_up_goal, y_down_goal, z_down_goal])
			
			x_up_pos = float(parameter["x_up_pos"])
			x_down_pos = float(parameter["x_down_pos"])
			y_up_pos = float(parameter["y_up_pos"])
			y_down_pos = float(parameter["y_down_pos"])
			z_down_pos = float(parameter["z_down_pos"])
			
			pos_dims = np.array([x_up_pos, x_down_pos, y_up_pos, y_down_pos, z_down_pos])
			
			self.env.update_gym_spaces(goal_dims, pos_dims)
			
			nb_random_goal = int(parameter["nb_random_goal"])
			
			self.env.set_E(E)
			self.env.set_NU(NU)
			self.env.set_factor_dt_factor(factor_dt_factor)
			self.env.set_time_step(time_step)
			self.env.set_gripper_orientation_to_add(x_rot,y_rot,z_rot)
			
			print("**** CHANGE-> E={}, NU={}, time_step={}, factor_dt_factor={} *****************".format(E,NU,time_step,factor_dt_factor))
			print("**** Initial gripper orientation x_rot={}, y_rot={}, z_rot={}".format(x_rot, y_rot, z_rot))
			
			f.write("{:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} \n".format(E, NU, time_step, factor_dt_factor, x_rot, y_rot, z_rot,x_up_goal, x_down_goal, y_up_goal, y_down_goal, z_down_goal, x_up_pos, x_down_pos, y_up_pos, y_down_pos, z_down_pos, nb_random_goal))
			f.flush()
			
			self.env.reset_env_bullet()
			self.env.update_gripper_orientation_bullet()
			if self.env.gui:
				self.env.draw_env_box()
				self.env.draw_frite_parameters()
			
			for i in range(nb_random_goal):
				# get a random goal from goal space = numpy array [x,y,z]
				a_random_goal = self.env.sample_goal_from_goal_space()
				print("*** {} : Go to GOAL : {} , gripper orien : x_rot={}, y_rot={}, z_rot={}!".format(i,a_random_goal,x_rot, y_rot, z_rot))
				start_time_goal = datetime.now()
				self.env.go_to_position_bullet(a_random_goal)
				print("*** Goal Reached with time elapsed = {} !".format(datetime.now()-start_time_goal))
				self.write_floats_with_gripper_pos(a_random_goal, f)
				print("***  Save meshes and gripper pos values !")
			
		print("*** Close db generate file !")
		f.close()
		
		print("End generate db with time elapsed = {}".format(datetime.now()-start_time_generate))
