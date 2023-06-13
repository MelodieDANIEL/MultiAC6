import os
import time

import numpy as np
import pybullet as p

import math

from gym import error, spaces, utils

from gym_panda_frite.envs.ik_dh import IK_DH
from gym_panda_frite.envs.debug_gui import Debug_Gui

from datetime import datetime

class PandaArm():
	
	def __init__(self, name = "Panda", startPosition = None, startOrientation = None, a_debug_gui = None, a_ik_dh = None):
		self._ik_dh = a_ik_dh
		
		self._debug_gui = a_debug_gui
		
		self._description_path = os.path.dirname(os.path.abspath(__file__)) + "/urdf/franka_panda/panda.urdf"
		#print("_description_path = {}".format(self._description_path))
		
		if startPosition is None:
			self._startPosition = [0.0, 0.0, 0.0]
		else:
			self._startPosition = startPosition
			
		if startOrientation is None:
			self._startOrientation = p.getQuaternionFromEuler([0,0,0])
		else:
			self._startOrientation = p.getQuaternionFromEuler(startOrientation)
			

		self._id = p.loadURDF(self._description_path,
								   basePosition=self._startPosition, baseOrientation=self._startOrientation, useFixedBase=True)

		self._num_joints = p.getNumJoints(self._id) # 12 joints
		print("{} arm has {} joints".format(name,self._num_joints))
		
		self._movable_joint_list_ids = [0, 1, 2, 3, 4, 5, 6]
		self._movable_initial_joint_position = [0.0,0.0,0.0,-math.pi/2.,0.0,math.pi/2,math.pi/4.]
		
		
		self._dic_initial_joint_positions = {
			'panda_joint1': 0.0, 'panda_joint2': 0.0, 'panda_joint3': 0.0,
			'panda_joint4': -math.pi/2., 'panda_joint5': 0.0, 'panda_joint6': math.pi/2,
			'panda_joint7': math.pi/4., 'panda_finger_joint1': 0.04, 'panda_finger_joint2': 0.04
		}
		
		self.set_initial_joints_positions()
		
		for i in range(self._num_joints):
			p.changeDynamics(self._id, i, linearDamping=0, angularDamping=0)
			
		self.set_limits()
		
		self._end_eff_idx = 9  # 7 arm only, 9 with gripper
		
		# generate random final pose within joint limits
		np.random.seed(0)
		
		# set parameters of joint motor control
		# frite , force = 100*240
		self._l_positionGains=np.float32(np.array([0.03]*len(self._movable_initial_joint_position)))
		self._l_velocityGains=np.float32(np.array([1.0]*len(self._movable_initial_joint_position)))
		self._l_forces=np.float32(np.array([500]*len(self._movable_initial_joint_position)))
		self._l_targetVelocities=np.float32(np.array([0.0]*len(self._movable_initial_joint_position)))
		
		# execute initial qvalues to use motor properties
		self.execute_qvalues([0.0,0.0,0.0,-math.pi/2.,0.0,math.pi/2,math.pi/4.])
		
	def draw_cross_ee(self):
		pos_ee, quat_ee = self.ee_pose(to_euler=False)
		self._debug_gui.draw_cross("ee_pose", a_pos = pos_ee, a_color = [0, 0, 1] )
		
	def go_to_cartesian(self, pos, orien):
		q_values, _, _, _ =  self.calculateInverseKinematics(pos, orien)
		self.execute_qvalues(q_values)
				
	def calculateInverseKinematics(self, position, quat_orientation, euler_angles_to_add = None):
		#start = datetime.now()
		A_final, euler_angles, quaternion_angles = self.calculate_RT(position,quat_orientation,euler_angles_to_add,debug=False)
		#print("calculate RT time elapsed = {}".format(datetime.now()-start))
		
		#start = datetime.now()
		numSteps = 0
		maxDelta = 0
		q_values, numSteps, maxDelta = self._ik_dh.incremental_ik(self._ik_dh._q_init, self._ik_dh._A_init, A_final, atol=1e-3, maxNumSteps=100)
		q_values[6] += (math.pi/4.0)
		#print("calculate incremental_ik = {}".format(datetime.now()-start))
		
		is_inside_limits = False

		if (self.inside_limits(q_values)):
			is_inside_limits = True
		
		return q_values, numSteps, maxDelta, is_inside_limits
	
	def printPandaAllInfo(self):
		print("=================================")
		print("All Panda Robot joints info")
		num_joints = p.getNumJoints(self._id)
		print("=> num of joints = {0}".format(num_joints))
		for i in range(num_joints):
			joint_info = p.getJointInfo(self._id, i)
			#print(joint_info)
			joint_name = joint_info[1].decode("UTF-8")
			joint_type = joint_info[2]
			child_link_name = joint_info[12].decode("UTF-8")
			link_pos_in_parent_frame = p.getLinkState(self._id, i)[0]
			link_orien_in_parent_frame = p.getLinkState(self._id, i)[1]
			joint_type_name = self._switcher_type_name.get(joint_type,"Invalid type")
			joint_lower_limit, joint_upper_limit = joint_info[8:10]
			joint_limit_effort = joint_info[10]
			joint_limit_velocity = joint_info[11]
			print("i={0}, name={1}, type={2}, lower={3}, upper={4}, effort={5}, velocity={6}".format(i,joint_name,joint_type_name,joint_lower_limit,joint_upper_limit,joint_limit_effort,joint_limit_velocity))
			print("child link name={0}, pos={1}, orien={2}".format(child_link_name,link_pos_in_parent_frame,link_orien_in_parent_frame))
		print("=================================")		
		
	def set_initial_joints_positions(self, init_gripper = True):
		# switch tool to convert a numeric value to string value
		self._switcher_type_name = {
			p.JOINT_REVOLUTE: "JOINT_REVOLUTE",
			p.JOINT_PRISMATIC: "JOINT_PRISMATIC",
			p.JOINT_SPHERICAL: "JOINT SPHERICAL",
			p.JOINT_PLANAR: "JOINT PLANAR",
			p.JOINT_FIXED: "JOINT FIXED"
		}
		
		self._joint_name_to_ids = {}
			
		for i in range(self._num_joints):
			joint_info = p.getJointInfo(self._id, i)
			joint_name = joint_info[1].decode("UTF-8")
			joint_type = joint_info[2]
			
			if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
				assert joint_name in self._dic_initial_joint_positions.keys()

				joint_type_name = self._switcher_type_name.get(joint_type,"Invalid type")
                    
				self._joint_name_to_ids[joint_name] = i
				
				if joint_name == 'panda_finger_joint1' or joint_name == 'panda_finger_joint2':
					if init_gripper:
						p.resetJointState(self._id, i, self._dic_initial_joint_positions[joint_name])
				else:
					p.resetJointState(self._id, i, self._dic_initial_joint_positions[joint_name])
					
	def set_limits(self):
		self._lower_limits = np.float32(np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973]))
		self._upper_limits = np.float32(np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]))
		self._range_limits = np.float32(np.subtract(self._upper_limits,self._lower_limits))
		
		all_joint_states = p.getJointStates(bodyUniqueId=self._id, jointIndices=[0,1,2,3,4,5,6,7])
		#print("all_joint_states ={}".format(all_joint_states))

		self._rest_poses = np.float32(np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]))
		for i in range(len(self._movable_initial_joint_position)+1):
			self._rest_poses[i] = all_joint_states[i][0]
	
	def ee_pose_with_velocity(self):
		eff_link_state = p.getLinkState(self._id, self._end_eff_idx, computeForwardKinematics=1,computeLinkVelocity=1)
			
		gripper_link_pos = np.array(eff_link_state[0]) # gripper cartesian world position = 3 float (x,y,z) = achieved goal
		gripper_link_orien = np.array(eff_link_state[1]) # gripper cartesian world orientation = 4 float (quaternion)
		gripper_link_orien_euler = p.getEulerFromQuaternion(gripper_link_orien) # gripper cartesian world orientation = 3 float (theta_x, theta_y, theta_z)
		gripper_link_vel = np.array(eff_link_state[6]) # gripper cartesian world velocity = 3 float (vx, vy, vz)
		gripper_link_vel_orien = np.array(eff_link_state[7]) # gripper cartesian world angular velocity = 3 float (theta_dot_x, theta_dot_y, theta_dot_z)
					
		return gripper_link_pos, gripper_link_orien, gripper_link_orien_euler, gripper_link_vel, gripper_link_vel_orien
	
	def ee_pose(self, to_euler=True, to_positive_euler=True):
		link_state = p.getLinkState(self._id, self._end_eff_idx, computeForwardKinematics=1)
							
		pos = np.asarray(link_state[0])
		
		if to_euler:
			ori = np.asarray(p.getEulerFromQuaternion(link_state[1]))
			if to_positive_euler:
				for i in range(3):
					if ori[i] < 0.0:
						ori[i] = round(ori[i] + 2*math.pi,2)%round(2*math.pi,2)
		else:
			ori = np.asarray(link_state[1])

		return pos, ori
	
	def calculate_RT(self, T, quat_ee, euler_angles=None, debug=False):
		#print("-> calculate_RT qua_ee = {}".format(quat_ee))
		
		matrix_ee = np.array(p.getMatrixFromQuaternion(quat_ee)).reshape((3, 3))
		if debug:
			print("matrix ee = {}".format(matrix_ee))
		
		if euler_angles is not None:
			quat_random = p.getQuaternionFromEuler([euler_angles[0],euler_angles[1],euler_angles[2]])
			if debug:
				print("quat_random = {}".format(quat_random))
			matrix_random = np.array(p.getMatrixFromQuaternion(quat_random)).reshape((3, 3))
			if debug:
				print("matrix_random = {}".format(matrix_random))
			
			R = np.dot(matrix_ee, matrix_random)
			if debug:
				print("np.dot(matrix_ee, matrix_random) -> R = {}".format(R))
		else:
			R = matrix_ee
		
		# melodie method
		
		beta = -np.arcsin(R[2,0])
		alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
		gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
		
		euler_angle = np.asarray([alpha, beta, gamma])
		
		quat_angle = p.getQuaternionFromEuler(euler_angle)
		
		# 4x4 RT matrix
		"""
		rt_matrix = Matrix(
				[
					[r00, r01, r02, T[0]],
					[r10, r11, r12, T[1]],
					[r20, r21, r22, T[2]],
					[0, 0, 0, 1]
				]
		)
		
		"""
		rt_matrix = [[R[0,0]],[R[1,0]],[R[2,0]],[R[0,1]],[R[1,1]],[R[2,1]],[R[0,2]],[R[1,2]],[R[2,2]],[T[0]],[T[1]],[T[2]]]
		
		return rt_matrix, euler_angle, quat_angle
		
	def inside_limits(self, qvalues):
		for i in range(len(qvalues)):
			if qvalues[i] < self._lower_limits[i] or qvalues[i] > self._upper_limits[i]:
				return False
				
		return True
	
	def close_gripper(self):
		id_finger_joint1  = self._joint_name_to_ids['panda_finger_joint1']
		id_finger_joint2  = self._joint_name_to_ids['panda_finger_joint2']

		p.setJointMotorControl2(self._id, id_finger_joint1, 
								p.POSITION_CONTROL,targetPosition=0.025,
								positionGain=1.0)
								
		p.setJointMotorControl2(self._id, id_finger_joint2 , 
								p.POSITION_CONTROL,targetPosition=0.025,
								positionGain=1.0)

	def execute_qvalues(self, qvalues):
		p.setJointMotorControlArray(	
						bodyUniqueId=self._id,
						jointIndices=self._movable_joint_list_ids,
						controlMode=p.POSITION_CONTROL,
						targetPositions=qvalues,
						targetVelocities=self._l_targetVelocities,
						positionGains=self._l_positionGains,
						velocityGains=self._l_velocityGains,
						forces=self._l_forces
									)


	def joint_angles(self):
		joint_angles = []

		for idx in self._movable_joint_list_ids:
			joint_state = p.getJointState(self._id, idx)

			joint_angles.append(joint_state[0])

		return np.array(joint_angles)
