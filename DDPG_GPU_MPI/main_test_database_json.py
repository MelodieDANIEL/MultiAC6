import sys
import gym
import numpy as np
import random
import torch
import os

import pybullet as p
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

import argparse
import os
import time
from datetime import datetime

from mpi4py import MPI
import pybullet as p

import gym_panda_frite
from database_frite import Database_Frite

from gym_panda_frite.envs.environment import Environment
from gym_panda_frite.envs.panda_frite_env_wrapper_reward_max import PandaFriteEnvWrapperRewardMax
from gym_panda_frite.envs.panda_frite_env_wrapper_reward_mean import PandaFriteEnvWrapperRewardMean
from gym_panda_frite.envs.panda_frite_env_wrapper_reward_dtw import PandaFriteEnvWrapperRewardDTW

from json_decoder import JsonDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test' or 'debug_cartesian' or 'debug_articular'
parser.add_argument('--gui', default=False, type=bool) # use gui to see graphic results
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)

args = parser.parse_args()

def main():
	
	#p.connect(p.GUI)
	
	if not os.path.isfile(args.config_file):
			raise RuntimeError("=> Config file JSON to load does not exit : " + args.config_file)
			return

	
	json_decoder = JsonDecoder(args.config_file)

	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'
	
	env_name = json_decoder.config_data["env"]["name"]
	env_random_seed = json_decoder.config_data["env"]["random_seed"]
	env_time_set_action = json_decoder.config_data["env"]["time_set_action"]
	env_reward_type = json_decoder.config_data["env"]["reward_parameters"]["reward_index"]
	
	print("** ENV PARAMETERS **")
	print("env_name = {}".format(env_name))
	print("env_random_seed = {}".format(env_random_seed))
	print("env_time_set_action = {}".format(env_time_set_action))
	
	ddpg_cuda = json_decoder.config_data["ddpg"]["cuda"]
	ddpg_max_memory_size = json_decoder.config_data["ddpg"]["max_memory_size"]
	ddpg_batch_size = json_decoder.config_data["ddpg"]["batch_size"]
	ddpg_log_interval = json_decoder.config_data["ddpg"]["log_interval"]
	
	print("** DDPG PARAMETERS **")
	print("ddpg_cuda = {}".format(ddpg_cuda))
	print("ddpg_batch_size = {}".format(ddpg_batch_size))
	print("ddpg_max_memory_size = {}".format(ddpg_max_memory_size))
	print("ddpg_log_interval = {}".format(ddpg_log_interval))
	
	log_name = json_decoder.config_data["log"]["name"]
	database_name = json_decoder.config_data["database"]["name"]
	
	print("** LOG PARAMETERS **")
	print("log_name = {}".format(log_name))
	
	rank = MPI.COMM_WORLD.Get_rank()
    
	if not os.path.isfile(json_decoder.config_dir_name  + database_name):
		raise RuntimeError("=> Database file to load does not exit : " + json_decoder.config_dir_name  + database_name)
		return
		
	if rank == 0:
		file_log = open(json_decoder.config_dir_name + log_name, "w+")
		goal_log = open(json_decoder.config_dir_name + "log_goal.txt", "w+")
		ids_frite_log = open(json_decoder.config_dir_name + "log_ids_frite.txt", "w+")
		db_selected = open(json_decoder.config_dir_name + "db_selected.txt", "w+")
		
	env_pybullet = Environment(json_decoder=json_decoder, gui=args.gui)
	#env_pybullet.startThread()
	
	db = Database_Frite(json_decoder=json_decoder)
	
	# what kind of reward ? ( 0 = Mean, 1 = Max, 2 = DTW)
	if env_reward_type == 0:
			env = PandaFriteEnvWrapperRewardMean(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	elif env_reward_type == 1:
			env = PandaFriteEnvWrapperRewardMax(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	
	elif env_reward_type == 2:
			env = PandaFriteEnvWrapperRewardDTW(gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank))
	else:
			raise RuntimeError("=> NO REWARD TYPE SPECIFIED INTO CONFIG FILE !!!!")
			return
	
	# env is a ros version
	# env into init class function, call read_all_spaces() and init_ros()
	# init_ros() call also set_gym_spaces()
	
	#env = gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=args.gui, env_rank=rank)

	env.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	#np.random.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	if (ddpg_cuda):
		torch.cuda.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	agent = DDPGagent(ddpg_cuda, num_states=env.get_nb_obs_values(), num_actions=env.get_nb_action_values(), max_memory_size=ddpg_max_memory_size, directory=json_decoder.config_dir_name+'env_panda/')
	noise = OUNoise(env.get_nb_action_values(), env.get_action_space_low() , env.get_action_space_high())
    
	list_global_rewards = []
	
	button_sample_deformation = p.addUserDebugParameter("Sample deformation", 1, 0, 1)
	previous_value_button_sample_deformation = p.readUserDebugParameter(button_sample_deformation)
	
	button_save_goal = p.addUserDebugParameter("Save Goal", 1, 0, 1)
	previous_value_button_save_goal = p.readUserDebugParameter(button_save_goal)
	
	button_test_rl = p.addUserDebugParameter("Test RL", 1, 0, 1)
	previous_value_button_test_rl = p.readUserDebugParameter(button_test_rl)
	
	button_save_ids_frite = p.addUserDebugParameter("Save ids frite", 1, 0, 1)
	previous_value_button_save_ids_frite = p.readUserDebugParameter(button_save_ids_frite)
	
	button_test_db_selected = p.addUserDebugParameter("Test DB selected", 1, 0, 1)
	previous_value_button_test_db_selected = p.readUserDebugParameter(button_test_db_selected)
	
	button_create_db = p.addUserDebugParameter("Create DB", 1, 0, 1)
	previous_value_button_create_db = p.readUserDebugParameter(button_create_db)
	
	while True:
		
		current_value_button_sample_deformation = p.readUserDebugParameter(button_sample_deformation)
		current_value_button_save_goal = p.readUserDebugParameter(button_save_goal)
		current_value_button_test_rl = p.readUserDebugParameter(button_test_rl)
		current_value_button_save_ids_frite = p.readUserDebugParameter(button_save_ids_frite)
		current_value_button_test_db_selected = p.readUserDebugParameter(button_test_db_selected)
		current_value_button_create_db = p.readUserDebugParameter(button_create_db)
		
		
		if current_value_button_create_db != previous_value_button_create_db:
			previous_value_button_create_db = current_value_button_create_db
			
			trajectory_log = open(json_decoder.config_dir_name + "db_trajectory.txt", "r")
			db_selected_traj = open(json_decoder.config_dir_name + "db_selected_trajectory.txt", "w+")
			
			parameters_array = np.empty([18])
			
			line = trajectory_log.readline()
			while line:
				# frite parameters
				line_split_parameters = line.split()
				
				for param in range(len(line_split_parameters)):
					parameters_array[param] = float(line_split_parameters[param])
					db_selected_traj.write("{} ".format(line_split_parameters[param]))
				
				# set new env frite parameters
				env.env.set_env_with_frite_parameters(parameters_array)
				
				db_selected_traj.write("\n")
				db_selected_traj.flush()
				
				line = trajectory_log.readline()
				
				print("Go home position !")
				
				env.env.go_to_home_position_ros()
				time.sleep(1)
				
				while len(line.split()) == 6:
					line_split = line.split()
					
					# move arm to pos and orien
					pos = np.array([float(line_split[0]), float(line_split[1]), float(line_split[2])])
					orien = np.array([float(line_split[3]), float(line_split[4]), float(line_split[5])])
					orien_quaternion = p.getQuaternionFromEuler(orien)
					
					print("Go cartesian pos={}, orien={} !".format(pos, orien))
					env.env.go_to_cartesian_ros(pos, orien_quaternion, None)
					
					line = trajectory_log.readline()
					print("len line read split = {}".format(len(line.split())))
					
				# save pos meshes
				
				# nb_meshes = 4 + 1
				nb_meshes = len(env.env.poses_meshes_in_arm_frame)

				for m in range(nb_meshes-1):
					db_selected_traj.write("{} {} {} ".format(env.env.poses_meshes_in_arm_frame[m+1][0], env.env.poses_meshes_in_arm_frame[m+1][1], env.env.poses_meshes_in_arm_frame[m+1][2]))

				db_selected_traj.write("\n")
				db_selected_traj.flush() 
					
			trajectory_log.close()
			db_selected_traj.close()
			

		if current_value_button_test_db_selected != previous_value_button_test_db_selected:
			 
			previous_value_button_test_db_selected = current_value_button_test_db_selected
			# TEST DB SELECTED
			
			print("Read DB selected")
			
			file_log_test_db_selected = open(json_decoder.config_dir_name + "log_test_db_selected.txt", "w+")
			
			for cpt_timer in range(10):
				env.env.publish_deformation_status_text("{}".format(cpt_timer))
				time.sleep(1)
			
			env.env.publish_deformation_status_text("")
			
			n_steps = json_decoder.config_data["env_test"]["n_steps"]
			
			parameters_array = np.empty([18])
			goal_array = np.empty([4,3])
			
			db_selected_open = open(json_decoder.config_dir_name + "db_selected_save.txt", "r")
			
			line = db_selected_open.readline()
			
			
			nb_line_parameters = 0
			num_episode = 0
			
			while line:
				line_split_parameters = line.split()
				nb_line_parameters+=2
				
				env.env.debug_gui.draw_text("nb_line_parameters", a_text=str(nb_line_parameters), a_pos=[1,1,1], a_size=1.0)
				
				for param in range(len(line_split_parameters)):
					parameters_array[param] = float(line_split_parameters[param])
				
				# set new env frite parameters
				env.env.set_env_with_frite_parameters(parameters_array)
				
				line = db_selected_open.readline()
				line_meshes = line.split()
				
				for id_g in range(4):
					for id_pos in range(3):
						goal_array[id_g][id_pos] = line_meshes[id_g*3+id_pos]
				
				# set new env goal
				env.env.set_goal(goal_array)
				
				print("EPISODE INFOS :")
				print("goal = {}, frite parameters = {}".format(goal_array, parameters_array))
				print("n_steps = {}".format(n_steps))
				
				#input("hit return to launch a new episode !")
				
				num_episode+=1
				print("num_episode = {}".format(num_episode))
				
				print("Move robot to home position with initial orientation !")
				
				# Need to do env.reset_ros() but WITHOUT SAMPLE A NEW GOAL
				# SO : 
				# 1-> MOVE TO HOME POSITION WITH GOOD ORIENTATION
				# 2-> GET FIRST OBSERVATION
				
				# move robot to home position and set initial orientation
				env.env.update_gripper_orientation_ros()
				
				# get the first observation
				state = env.env.get_obs_ros()
				
				# initial error
				nb_mesh_to_follow = len(env.env.position_mesh_to_follow)
		
				max_d = 0
				initial_error = 0

				for i in range(nb_mesh_to_follow):
					current_pos_mesh = state[(env.env.pos_of_mesh_in_obs+(i*3)):(env.env.pos_of_mesh_in_obs+(i*3)+3)]
					goal_pos_id_frite = env.env.goal[i]
					d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
						
					if (d > max_d):
						max_d = d
						
				max_d = np.float32(max_d)

				initial_error = -max_d
				
				print("episode={}, initial error={}".format(num_episode,initial_error))
				file_log_test_db_selected.write("episode={}, initial error={}\n".format(num_episode,initial_error))
				file_log_test_db_selected.flush()
				
				# EXECUTE RL
				
				start=datetime.now()
			
				file_log.write("mode test db selected !\n")
				file_log_test_db_selected.write("mode test db selected !\n")
				file_log_test_db_selected.flush()

				print("mode test db selected !")

				agent.load()
			
				current_distance_error = 0
			
				for step in range(n_steps):
					
					action = agent.get_action(state)
				
					print("action={}".format(action))
					file_log.write("action = {}\n".format(action))
					file_log.flush()
					
					file_log_test_db_selected.write("action = {}\n".format(action))
					file_log_test_db_selected.flush()
					

					new_state, reward, done, info = env.step_ros(action)
					current_distance_error = info['distance_error']

					print("step={}, distance_error={}\n".format(step,info['distance_error']))
					file_log.write("step={}, distance_error={}\n".format(step,info['distance_error']))
					file_log.flush()
					
					file_log_test_db_selected.write("step={}, distance_error={}\n".format(step,info['distance_error']))
					file_log_test_db_selected.flush()

					state = new_state
			   
					if done:
						env.env.publish_deformation_status_text("DONE")
						print("DONE episode={}, step={}  !".format(num_episode,step))
						file_log.write("DONE episode={}, step={}  !\n".format(num_episode,step))
						file_log.flush()
						
						file_log_test_db_selected.write("DONE episode={}, step={}  !\n".format(num_episode,step))
						file_log_test_db_selected.flush()
						
						time.sleep(2.0)
						env.env.publish_deformation_status_text("")
						break
			
				if not done:
					print("FAILED episode={} !".format(num_episode))
					file_log.write("FAILED episode={} !\n".format(num_episode))
					file_log.flush()
					
					file_log_test_db_selected.write("FAILED episode={} !\n".format(num_episode))
					file_log_test_db_selected.flush()
					
					env.env.publish_deformation_status_text("FAILED")
					time.sleep(2.0)
					env.env.publish_deformation_status_text("")
							
				file_log.write("time elapsed = {}\n".format(datetime.now()-start))
				file_log.flush()
				
				file_log_test_db_selected.write("time elapsed = {}\n".format(datetime.now()-start))
				file_log_test_db_selected.flush()
						
				# read a new episode
				line = db_selected_open.readline()
			
			db_selected_open.close()
			file_log_test_db_selected.close()
			
			print("end test DB selected")
		
		if current_value_button_sample_deformation != previous_value_button_sample_deformation:
			 
			previous_value_button_sample_deformation = current_value_button_sample_deformation
			# SAMPLE DEFORMATION
			
			# sample a new goal, move to home position and good orientation, get first observation
			state = env.reset_ros()
			
		if current_value_button_save_goal != previous_value_button_save_goal:
			 
			previous_value_button_save_goal = current_value_button_save_goal
			# SAVE GOAL
			
			print("save goal : {}".format(env.env.goal))
			
			all_frite_parameters = env.env.goal_with_frite_parameters
			frite_parameters_only = all_frite_parameters[0]
			goal = all_frite_parameters[1]
			
			nb_goal = len(goal)
			print("nb_goal = {}, goal = {}".format(nb_goal,goal))
			nb_frite_parameters = len(frite_parameters_only)
			
			for f in range(nb_frite_parameters):
				goal_log.write("{} ".format(frite_parameters_only[f]))
			goal_log.write("\n")
			goal_log.flush()
			
			for g in range(nb_goal):
				for v in range(3):
					goal_log.write("{} ".format(env.env.goal[g][v]))
				
			goal_log.write("\n")
			goal_log.flush()
			
			print("end save goal !")
			
		if current_value_button_test_rl != previous_value_button_test_rl:
			 
			previous_value_button_test_rl = current_value_button_test_rl
			# TEST RL
			
			trajectory_log = open(json_decoder.config_dir_name + "db_trajectory.txt", "a")
			
			start=datetime.now()
			
			n_steps = json_decoder.config_data["env_test"]["n_steps"]

			file_log.write("mode test real !\n")

			print("mode test real !")

			agent.load()
			
			current_distance_error = 0
			
			continue_step = True
			
			# write frite parameters of sampled current deformation
			all_frite_parameters = env.env.goal_with_frite_parameters
			frite_parameters_only = all_frite_parameters[0]
			
			nb_frite_parameters = len(frite_parameters_only)
			
			for f in range(nb_frite_parameters):
				trajectory_log.write("{} ".format(frite_parameters_only[f]))
				
			trajectory_log.write("\n")
			trajectory_log.flush()
			
			for step in range(n_steps):
				response = input("launch a step ? [y/N] ")
				response = response.strip().lower()
				if response.startswith('y'):
					continue_step = True
				elif response.startswith('n') or response == '':
					continue_step = False
				else:
					print("answer 'y' or 'n'")
					continue_step = False

				if continue_step == False:
					break
						
				action = agent.get_action(state)
				
				print("action={}".format(action))
				file_log.write("action = {}\n".format(action))
			   
				new_state, reward, done, info = env.step_ros(action)
				current_distance_error = info['distance_error']
				
				obs_traj = env.env.get_position_orientation_gripper_ros()
				
				for i_traj in range(len(obs_traj)):
					trajectory_log.write("{} ".format(obs_traj[i_traj]))
				trajectory_log.write("\n")
				trajectory_log.flush()
				
				print("step={}, distance_error={}\n".format(step,info['distance_error']))
				file_log.write("step={}, distance_error={}\n".format(step,info['distance_error']))
				
				state = new_state
			   
				if done:
				   print("done with step={}  !".format(step))
				   file_log.write("done with step={}  !\n".format(step))
				   break
			
			file_log.write("time elapsed = {}\n".format(datetime.now()-start))
			trajectory_log.close()
				   
			print("End test real !")
			
		if current_value_button_save_ids_frite != previous_value_button_save_ids_frite:
			 
			previous_value_button_save_ids_frite = current_value_button_save_ids_frite
			# SAVE IDS FRITE
			
			print("save ids frite, save DB selected")
			
			all_frite_parameters = env.env.goal_with_frite_parameters
			frite_parameters_only = all_frite_parameters[0]
			
			nb_frite_parameters = len(frite_parameters_only)
			
			# nb_meshes = 4 + 1
			nb_meshes = len(env.env.poses_meshes_in_arm_frame)
			
			for f in range(nb_frite_parameters):
				ids_frite_log.write("{} ".format(frite_parameters_only[f]))
				db_selected.write("{} ".format(frite_parameters_only[f]))
				
			ids_frite_log.write("\n")
			db_selected.write("\n")
			
			ids_frite_log.flush()
			db_selected.flush()
			
			for m in range(nb_meshes-1):
				ids_frite_log.write("{} {} {} ".format(env.env.poses_meshes_in_arm_frame[m+1][0], env.env.poses_meshes_in_arm_frame[m+1][1], env.env.poses_meshes_in_arm_frame[m+1][2]))
				db_selected.write("{} {} {} ".format(env.env.poses_meshes_in_arm_frame[m+1][0], env.env.poses_meshes_in_arm_frame[m+1][1], env.env.poses_meshes_in_arm_frame[m+1][2]))
			
			ids_frite_log.write("\n")
			db_selected.write("\n")
			
			ids_frite_log.flush()
			db_selected.flush()
			
			print("end save ids frite !")
			    
if __name__ == '__main__':
	main()

