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
import gym
import numpy as np
import random
import torch
import os

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

from json_decoder import JsonDecoder

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test' or 'debug_cartesian' or 'debug_articular'
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)

args = parser.parse_args()

def main():
	
	if not os.path.isfile(args.config_file):
			raise RuntimeError("=> Config file JSON to load does not exit : " + args.config_file)
			return

	json_decoder = JsonDecoder(args.config_file)

	os.environ['OMP_NUM_THREADS'] = '1'
	os.environ['MKL_NUM_THREADS'] = '1'
	os.environ['IN_MPI'] = '1'
	
	env_name = json_decoder.config_data["env"]["rotation_gripper_parameters"]["name"]
	env_random_seed = json_decoder.config_data["env"]["random_seed"]
	
	print("** ENV PARAMETERS **")
	print("env_name = {}".format(env_name))
	print("env_random_seed = {}".format(env_random_seed))
	
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

	db = Database_Frite(json_decoder=json_decoder)
	env = gym.make(env_name, database=db, json_decoder = json_decoder, env_rank=rank)

	env.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	torch.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
	#np.random.seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	if (ddpg_cuda):
		torch.cuda.manual_seed(env_random_seed + MPI.COMM_WORLD.Get_rank())
    
	agent = DDPGagent(ddpg_cuda, num_states=env.get_nb_obs_values(), num_actions=env.get_nb_action_values(), max_memory_size=ddpg_max_memory_size, directory=json_decoder.config_dir_name+'env_rotation_gripper/')
	noise = OUNoise(env.get_nb_action_values(), env.get_action_space_low() , env.get_action_space_high())
    
	list_global_rewards = []
    
	if args.mode == 'test':
		print("TEST MODE !")
		
		start=datetime.now()
		
		file_log.write("mode test !\n")
		
		agent.load()
		n_episodes = json_decoder.config_data["env_test"]["n_episodes"]
		n_steps = json_decoder.config_data["env_test"]["n_steps"]
		
		
		file_log.write("** ENV MODE TEST **\n")
		file_log.write("config_file = {}\n".format(args.config_file))
		file_log.write("n_episodes = {}\n".format(n_episodes))
		file_log.write("n_steps = {}\n".format(n_steps))
		file_log.flush()
		
		print("** ENV MODE TEST **")
		print("n_episodes = {}".format(n_episodes))
		print("n_steps = {}".format(n_steps))
				
		nb_dones = 0
		sum_distance_error = 0
		list_episode_error = []
		for episode in range(n_episodes):
			print("Episode : {}".format(episode))
			file_log.write("Episode : {}\n".format(episode))
		   
			print("RESET !")
			file_log.write("RESET !\n")
			state = env.reset_bullet()
				   
			current_distance_error = 0
			
			for step in range(n_steps):
				action = agent.get_action(state)
				
				print("action={}".format(action))
				file_log.write("action = {}\n".format(action))
				file_log.flush()
			   
				new_state, reward, done, info = env.step_bullet(action, rank=rank, episode=episode, step=step)
				current_distance_error = info['distance_error']
				
				print("step={}, distance_error={}\n".format(step,info['distance_error']))
				file_log.write("step={}, distance_error={}\n".format(step,info['distance_error']))
				file_log.flush()
				
				#print("step={}, action={}, reward={}, done={}, info={}".format(step,action,reward, done, info))
				state = new_state
			   
				if done:
				   print("done with step={}  !".format(step))
				   file_log.write("done with step={}  !\n".format(step))
				   
				   nb_dones+=1
				   break
		   
			sum_distance_error += current_distance_error
			list_episode_error.append(current_distance_error)

		print("nb dones = {}".format(nb_dones))
		print("mean distance error = {}".format(sum_distance_error/n_episodes))
		print("sum distance error = {}".format(sum_distance_error))
		print("std error episode = {}, min error episode = {}, max error episode = {}".format(np.std(np.array(list_episode_error)),min(list_episode_error),max(list_episode_error)))
		print("time elapsed = {}".format(datetime.now()-start))
		
		file_log.write("nb dones = {}\n".format(nb_dones))
		file_log.write("mean distance error = {}\n".format(sum_distance_error/n_episodes))
		file_log.write("sum distance error = {}\n".format(sum_distance_error))
		file_log.write("std error episode = {}, min error episode = {}, max error episode = {}".format(np.std(np.array(list_episode_error)),min(list_episode_error),max(list_episode_error)))
		file_log.write("time elapsed = {}\n".format(datetime.now()-start))
		file_log.flush()
		
		file_log.close()
		
		input("hit return to exit !")
			
	elif args.mode == 'train':
		print("TRAIN MODE !")
		start=datetime.now()
		
		n_episodes = json_decoder.config_data["env_train"]["n_episodes"]
		n_steps = json_decoder.config_data["env_train"]["n_steps"]
		ddpg_load = json_decoder.config_data["ddpg"]["load"]
		
		print("** ENV MODE TRAIN OPTIONS **")
		print("n_episodes = {}".format(n_episodes))
		print("n_steps = {}".format(n_steps))
		print("ddpg_load = {}".format(ddpg_load))
		
		if rank == 0:
			f_mean_rewards = open(json_decoder.config_dir_name + "mean_rewards.txt", "w+")
			f_max_rewards = open(json_decoder.config_dir_name + "max_rewards.txt", "w+")
			f_min_rewards = open(json_decoder.config_dir_name + "min_rewards.txt", "w+")
			file_log.write("** ENV MODE TRAIN OPTIONS **\n")
			file_log.write("config_file = {}\n".format(args.config_file))
			file_log.write("n_episodes = {}\n".format(n_episodes))
			file_log.write("n_steps = {}\n".format(n_steps))
			file_log.write("ddpg_load = {}\n".format(ddpg_load))
			file_log.flush()
		
		if ddpg_load:
			agent.load()
			
		global_step_number = 0
		
		for episode in range(n_episodes):
			#print("** rank {}, episode {}".format(rank,episode))
			
			state = env.reset_bullet()
   
			noise.reset()
			episode_reward = 0
			for step in range(n_steps):
				action = agent.get_action(state)
				action = noise.get_action(action, step)
				new_state, reward, done, info = env.step_bullet(action,rank,episode,step)
				agent.memory.push(state, action, reward, new_state, done)
				global_step_number += 1

				if len(agent.memory) > ddpg_batch_size:
					agent.update(ddpg_batch_size)

				state = new_state
				episode_reward += reward
				
				if rank==0:
				   print('=> [{}] rank is: {}, episode is: {}, step is: {}, action is: {}'.format(datetime.now(), rank, episode, step, action))
				

			#print('[{}] rank is: {}, episode is: {}, episode reward is: {:.3f}'.format(datetime.now(), rank, episode, episode_reward))
			
			
			global_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.SUM)/MPI.COMM_WORLD.Get_size()
			list_global_rewards.append(global_reward)
			
			min_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.MIN)
			max_reward = MPI.COMM_WORLD.allreduce(episode_reward, op=MPI.MAX)
			
			if rank == 0:
				f_mean_rewards.write("{} {:.3f}\n".format(episode,global_reward))
				f_min_rewards.write("{} {:.3f}\n".format(episode, min_reward ))
				f_max_rewards.write("{} {:.3f}\n".format(episode, max_reward))
				print('=> [{}] episode is: {}, eval success rate is: {:.3f}'.format(datetime.now(), episode, list_global_rewards[episode]))
				file_log.write('=> [{}] episode is: {}, eval success rate is: {:.3f}\n'.format(datetime.now(), episode, list_global_rewards[episode])) 
				file_log.flush()
				f_mean_rewards.flush()
				f_min_rewards.flush()
				f_max_rewards.flush()
				
				if episode % ddpg_log_interval == 0:
					agent.save()
				  
		if rank == 0:          
		   agent.save()
		   print("end mode train !")
		   print("time elapsed = {}".format(datetime.now()-start))
		   file_log.write("end mode train !\n")
		   file_log.write("time elapsed = {}\n".format(datetime.now()-start))
		   file_log.flush()
		   file_log.close()
		   f_mean_rewards.close()
		   f_min_rewards.close()
		   f_max_rewards.close()
		   	
	elif args.mode == 'simple_test':
		print("SIMPLE TEST MODE !")
		input("hit return to exit !")
		
	else:
		raise NameError("mode wrong!!!")
		     
if __name__ == '__main__':
	main()
