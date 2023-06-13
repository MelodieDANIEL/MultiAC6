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

import pybullet as p
from mpi4py import MPI

import gym_panda_frite
from database_frite import Database_Frite

from gym_panda_frite.envs.environment import Environment

from json_decoder import JsonDecoder


parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='./configs/default/default.json', type=str)

args = parser.parse_args()


def test_limits_pos(env, euler_angle_to_add):
		print("Start Test of Pos corners limits, euler angle to add = {} !".format(euler_angle_to_add))

		nb_out_limits = 0

		w_lp = env.array_low_pos_space
		w_hp = env.array_high_pos_space
		
		#print("test_limits_pos, w_lp = {}, w_hp = {}".format(w_lp,w_hp))
		#print("test_limits_pos, env.array_low_pos_space={}, env.array_high_pos_space={}".format(env.array_low_pos_space,env.array_high_pos_space))

		x_d = w_lp[0]
		x_u = w_hp[0]
		y_d = w_lp[1]
		y_u = w_hp[1]
		z_d = w_lp[2]
		z_u = w_hp[2]
		
		#print("*** Box pos size = [{},{},{},{},{}]".format(x_u,x_d,y_u,y_d,z_d))
		
		p1 = [x_d,y_d,z_u]
		env.debug_gui.draw_text("corner_p1", a_text = "p1", a_pos = p1, a_size = 1.5, a_color = [1, 0, 0])
	
		p1_truncated = env.truncate_array(p1,3)
		q_final_p1, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p1_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p1", a_pos = p1, a_color = [0, 1, 0] )
			print("p1 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p1", a_pos = p1, a_color = [1, 0, 0] )
			print("p1 NOT into limits !")
			nb_out_limits+=1
			
		p2 = [x_d,y_u,z_u]
		env.debug_gui.draw_text("corner_p2", a_text = "p2", a_pos = p2, a_size = 1.5, a_color = [1, 0, 0])

		p2_truncated = env.truncate_array(p2,3)
		q_final_p2, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p2_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p2", a_pos = p2, a_color = [0, 1, 0] )
			print("p2 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p2", a_pos = p2, a_color = [1, 0, 0] )
			print("p2 NOT into limits !")
			nb_out_limits+=1


		p3 = [x_u,y_d,z_u]
		env.debug_gui.draw_text("corner_p3", a_text = "p3", a_pos = p3, a_size = 1.5, a_color = [1, 0, 0])

		p3_truncated = env.truncate_array(p3,3)
		q_final_p3, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p3_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p3", a_pos = p3, a_color = [0, 1, 0] )
			print("p3 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p3", a_pos = p3, a_color = [1, 0, 0] )
			print("p3 NOT into limits !")
			nb_out_limits+=1


		p4 = [x_u,y_u,z_u]
		env.debug_gui.draw_text("corner_p4", a_text = "p4", a_pos = p4, a_size = 1.5, a_color = [1, 0, 0])

		p4_truncated = env.truncate_array(p4,3)
		q_final_p4, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p4_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p4", a_pos = p4, a_color = [0, 1, 0] )
			print("p4 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p4", a_pos = p4, a_color = [1, 0, 0] )
			print("p4 NOT into limits !")
			nb_out_limits+=1

		p5 = [x_d,y_d,z_d]
		env.debug_gui.draw_text("corner_p5", a_text = "p5", a_pos = p5, a_size = 1.5, a_color = [1, 0, 0])

		p5_truncated = env.truncate_array(p5,3)
		q_final_p5, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p5_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p5", a_pos = p5, a_color = [0, 1, 0] )
			print("p5 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p5", a_pos = p5, a_color = [1, 0, 0] )
			print("p5 NOT into limits !")
			nb_out_limits+=1

		p6 = [x_d,y_u,z_d]
		env.debug_gui.draw_text("corner_p6", a_text = "p6", a_pos = p6, a_size = 1.5, a_color = [1, 0, 0])

		p6_truncated = env.truncate_array(p6,3)
		q_final_p6, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p6_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p6", a_pos = p6, a_color = [0, 1, 0] )
			print("p6 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p6", a_pos = p6, a_color = [1, 0, 0] )
			print("p6 NOT into limits !")
			nb_out_limits+=1

		p7 = [x_u,y_d,z_d]
		env.debug_gui.draw_text("corner_p7", a_text = "p7", a_pos = p7, a_size = 1.5, a_color = [1, 0, 0])

		p7_truncated = env.truncate_array(p7,3)
		q_final_p7, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p7_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p7", a_pos = p7, a_color = [0, 1, 0] )
			print("p7 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p7", a_pos = p7, a_color = [1, 0, 0] )
			print("p7 NOT into limits !")
			nb_out_limits+=1


		p8 = [x_u,y_u,z_d]
		env.debug_gui.draw_text("corner_p8", a_text = "p8", a_pos = p8, a_size = 1.5, a_color = [1, 0, 0])

		p8_truncated = env.truncate_array(p8,3)
		q_final_p8, _, _, is_inside_limits =  env.panda_arm.calculateInverseKinematics(p8_truncated, env.initial_orien_gripper, euler_angle_to_add)
		
		if (is_inside_limits):
			env.debug_gui.draw_cross("corner_p8", a_pos = p8, a_color = [0, 1, 0] )
			print("p8 into limits !")
		else:
			env.debug_gui.draw_cross("corner_p8", a_pos = p8, a_color = [1, 0, 0] )
			print("p8 NOT into limits !")
			nb_out_limits+=1
		
		
		print("nb out limits = {}".format(nb_out_limits))
		print("End Test of corners limits, euler angle to add = {} !".format(euler_angle_to_add))
		
		return nb_out_limits, q_final_p1, q_final_p2, q_final_p3, q_final_p4, q_final_p5, q_final_p6, q_final_p7, q_final_p8
		

if __name__ == '__main__':
	
	if not os.path.isfile(args.config_file):
		raise RuntimeError("=> Config file JSON to load does not exit : " + args.config_file)

	json_decoder = JsonDecoder(args.config_file)

	env_name = json_decoder.config_data["env"]["name"]
	database_name = json_decoder.config_data["database"]["name"]

	if not os.path.isfile(json_decoder.config_dir_name  + database_name):
		raise RuntimeError("=> Database file to load does not exit : " + json_decoder.config_dir_name  + database_name)

	rank = MPI.COMM_WORLD.Get_rank()

	env_pybullet = Environment(json_decoder=json_decoder, gui=True)
	env_pybullet.startThread()

	db = Database_Frite(json_decoder=json_decoder)
	env = gym.make(env_name, database=db, json_decoder = json_decoder, env_pybullet=env_pybullet, gui=True, env_rank=rank)
	
	box_action = np.empty(5, dtype=np.float64)
	
	euler_angle_to_add = [0.0, 0.0, 0.0]
	
	# set env type : from_db
	# Think to set "orientation_gripper/index" to 1 (from_db) into json config file !!!
	env.type_orientation_gripper = 1 
	
	current_index_parameter_array = 0
	size_parameter_array = len(json_decoder.config_data["database"]["frite_parameters"]["parameters_array"])
	
	reset_user_debug_parameters = True
	
	pose_values_from_json = np.empty(5, dtype=np.float64)
	orien_gripper_values_from_json = np.empty(3, dtype=np.float64)
	goal_values_from_json = np.empty(5, dtype=np.float64)
	
	while True:
		
		if reset_user_debug_parameters:
			
			pose_values_from_json[0] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["x_up_pos"]
			pose_values_from_json[1] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["x_down_pos"]
			pose_values_from_json[2] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["y_up_pos"]
			pose_values_from_json[3] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["y_down_pos"]
			pose_values_from_json[4] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["z_down_pos"]
			
			orien_gripper_values_from_json[0] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["x_rot"]
			orien_gripper_values_from_json[1] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["y_rot"]
			orien_gripper_values_from_json[2] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["z_rot"]
			
			goal_values_from_json[0] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["x_up_goal"]
			goal_values_from_json[1] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["x_down_goal"]
			goal_values_from_json[2] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["y_up_goal"]
			goal_values_from_json[3] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["y_down_goal"]
			goal_values_from_json[4] = json_decoder.config_data["database"]["frite_parameters"]["parameters_array"][current_index_parameter_array]["z_down_goal"]
			
			list_slider_box = []
			list_slider_box.append(p.addUserDebugParameter("X_UP", -0.1, 1, pose_values_from_json[0]))
			list_slider_box.append(p.addUserDebugParameter("X_DOWN", 0, 1, pose_values_from_json[1]))
			list_slider_box.append(p.addUserDebugParameter("Y_UP", 0, 1, pose_values_from_json[2]))
			list_slider_box.append(p.addUserDebugParameter("Y_DOWN", 0, 1, pose_values_from_json[3]))
			list_slider_box.append(p.addUserDebugParameter("Z_DOWN", 0, 1, pose_values_from_json[4]))
			
			slider_x_roll = p.addUserDebugParameter("X-Roll", -0.5, 0.5, orien_gripper_values_from_json[0])
			slider_y_pitch = p.addUserDebugParameter("Y-Pitch", -0.5, 0.5, orien_gripper_values_from_json[1])
			slider_z_yaw = p.addUserDebugParameter("Z-Yaw", -0.5, 0.5, orien_gripper_values_from_json[2])
			
			button_show_goal_box = p.addUserDebugParameter("SHOW GOAL BOX", 1, 0, 1)
			previous_value_button_show_goal_box = p.readUserDebugParameter(button_show_goal_box)
			
			button_remove_goal_box = p.addUserDebugParameter("REMOVE GOAL BOX", 1, 0, 1)
			previous_value_button_remove_goal_box = p.readUserDebugParameter(button_remove_goal_box)
			
			button_show_angle = p.addUserDebugParameter("SHOW ANGLE", 1, 0, 1)
			previous_value_button_show_angle = p.readUserDebugParameter(button_show_angle)
			
			button_reset_panda = p.addUserDebugParameter("RESET PANDA", 1, 0, 1)
			previous_value_button_reset_panda = p.readUserDebugParameter(button_reset_panda)
			
			button_test_corners = p.addUserDebugParameter("TEST CORNERS", 1, 0, 1)
			previous_value_button_test_corners = p.readUserDebugParameter(button_test_corners)
			
			button_move_to_corners = p.addUserDebugParameter("MOVE TO CORNERS", 1, 0, 1)
			previous_value_button_move_to_corners = p.readUserDebugParameter(button_move_to_corners)
			
			button_reload_json = p.addUserDebugParameter("RELOAD JSON", 1, 0, 1)
			previous_value_button_reload_json = p.readUserDebugParameter(button_reload_json)
			
			button_first_db_box = p.addUserDebugParameter("FIRST - JSON BOX DB", 1, 0, 1)
			previous_value_button_first_db_box = p.readUserDebugParameter(button_first_db_box)
			
			button_next_db_box = p.addUserDebugParameter("NEXT - JSON BOX DB", 1, 0, 1)
			previous_value_button_next_db_box = p.readUserDebugParameter(button_next_db_box)
			
			button_last_db_box = p.addUserDebugParameter("LAST - JSON BOX DB", 1, 0, 1)
			previous_value_button_last_db_box = p.readUserDebugParameter(button_last_db_box)
			
			env.set_gripper_orientation_to_add(0.0, 0.0, 0.0)
			env.set_panda_initial_joints_positions()
			env.update_gripper_orientation_bullet()
			
			reset_user_debug_parameters = False
			
		current_value_button_show_angle = p.readUserDebugParameter(button_show_angle)
		current_value_button_show_goal_box = p.readUserDebugParameter(button_show_goal_box)
		current_value_button_remove_goal_box = p.readUserDebugParameter(button_remove_goal_box)
		
		current_value_button_reset_panda = p.readUserDebugParameter(button_reset_panda)
		current_value_button_test_corners = p.readUserDebugParameter(button_test_corners)
		current_value_button_move_to_corners = p.readUserDebugParameter(button_move_to_corners)
		current_value_button_reload_json = p.readUserDebugParameter(button_reload_json)
		
		current_value_button_first_db_box = p.readUserDebugParameter(button_first_db_box)
		current_value_button_next_db_box = p.readUserDebugParameter(button_next_db_box)
		current_value_button_last_db_box = p.readUserDebugParameter(button_last_db_box)
		
		for i in range(5):
			box_action[i] = p.readUserDebugParameter(list_slider_box[i])
			
		roll_angle = p.readUserDebugParameter(slider_x_roll)
		
		"""
		if roll_angle != -0.5 and roll_angle != 0.5:
			roll_angle = 0.0
		"""
		
		pitch_angle = p.readUserDebugParameter(slider_y_pitch)
		
		"""
		if pitch_angle != -0.5 and pitch_angle != 0.5:
			pitch_angle = 0.0
		"""
			
		yaw_angle = p.readUserDebugParameter(slider_z_yaw)
			
		euler_angle_to_add[0] = roll_angle
		euler_angle_to_add[1] = pitch_angle
		euler_angle_to_add[2] = yaw_angle
				
		env.debug_gui.draw_text(a_name = "box_index", a_text = str(current_index_parameter_array), a_pos = [1,1,1], a_size = 5.0, a_color = [1, 0, 0])
		env.update_pose_space(box_action)
		env.draw_env_box_pose()
			
		if current_value_button_show_goal_box != previous_value_button_show_goal_box:
			previous_value_button_show_goal_box = current_value_button_show_goal_box
			
			env.update_goal_space(goal_values_from_json)
			env.draw_env_box_goal()
			
		if current_value_button_remove_goal_box != previous_value_button_remove_goal_box:
			previous_value_button_remove_goal_box = current_value_button_remove_goal_box
			
			env.remove_env_box_goal()
		
		if current_value_button_first_db_box != previous_value_button_first_db_box:
			previous_value_button_first_db_box = current_value_button_first_db_box
			
			current_index_parameter_array = 0
			print("current_index_parameter_array = {}".format(current_index_parameter_array))
			
			p.removeAllUserParameters()
			reset_user_debug_parameters = True
		
		if current_value_button_next_db_box != previous_value_button_next_db_box:
			previous_value_button_next_db_box = current_value_button_next_db_box
			
			current_index_parameter_array += 1
			
			if (current_index_parameter_array >= size_parameter_array):
				current_index_parameter_array = 0
			
			print("current_index_parameter_array = {}".format(current_index_parameter_array))	
			
			p.removeAllUserParameters()
			reset_user_debug_parameters = True	
			
		if current_value_button_last_db_box != previous_value_button_last_db_box:
			previous_value_button_last_db_box = current_value_button_last_db_box
			
			current_index_parameter_array = size_parameter_array - 1
			print("current_index_parameter_array = {}".format(current_index_parameter_array))	
			
			p.removeAllUserParameters()
			reset_user_debug_parameters = True
		
		if current_value_button_reload_json != previous_value_button_reload_json:
			previous_value_button_reload_json = current_value_button_reload_json
			json_decoder.load_config()
			
			current_index_parameter_array = 0
			print("current_index_parameter_array = {}".format(current_index_parameter_array))
			size_parameter_array = len(json_decoder.config_data["database"]["frite_parameters"]["parameters_array"])
			
			p.removeAllUserParameters()
			reset_user_debug_parameters = True
			
		if current_value_button_show_angle != previous_value_button_show_angle:
			previous_value_button_show_angle = current_value_button_show_angle
			
			print("Show angle, euler to add = {}".format(euler_angle_to_add))
			
			env.set_gripper_orientation_to_add(euler_angle_to_add[0], euler_angle_to_add[1], euler_angle_to_add[2])
			env.set_panda_initial_joints_positions()
			env.update_gripper_orientation_bullet()
		
		if current_value_button_reset_panda != previous_value_button_reset_panda:
			previous_value_button_reset_panda = current_value_button_reset_panda
			env.set_gripper_orientation_to_add(0.0, 0.0, 0.0)
			env.set_panda_initial_joints_positions()
			env.update_gripper_orientation_bullet()

		if current_value_button_test_corners != previous_value_button_test_corners:
			previous_value_button_test_corners = current_value_button_test_corners
			nb_out_limits, q_final_p1, q_final_p2, q_final_p3, q_final_p4, q_final_p5, q_final_p6, q_final_p7, q_final_p8 = test_limits_pos(env, euler_angle_to_add)

		if current_value_button_move_to_corners != previous_value_button_move_to_corners:
			previous_value_button_move_to_corners = current_value_button_move_to_corners
			
			print("Start move to corners !")
			
			env.panda_arm.execute_qvalues(q_final_p1)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p1 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p2)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p2 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p4)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p4 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p3)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p3 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p5)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p5 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p6)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p6 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p8)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p8 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))
			
			env.panda_arm.execute_qvalues(q_final_p7)
			max_distance, time_elapsed = env.wait_until_frite_deform_ended()
			print("Move to p7 with max_distance={}, time elapsed={}".format(max_distance,time_elapsed))

			print("End Move to corners !")
