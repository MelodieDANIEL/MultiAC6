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

# IK to develop : https://gist.github.com/mlaves/a60cbc5541bd6c9974358fbaad9e4c51
# https://www.andre-gaschler.com/rotationconverter/
# https://frankaemika.github.io/docs/control_parameters.html
# pip install sympy
# pip install numba
# print(numba.__version__)
# 0.56.2


from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi
import numpy as np
from sympy import lambdify
from numba import jit

import math
from datetime import datetime
import time

class IK_DH:
	
	def __init__(self):
			init_printing(use_unicode=True)
			# create joint angles as symbols

			self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7 = symbols('theta_1 theta_2 theta_3 theta_4 theta_5 theta_6 theta_7')
			self.joint_angles = [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7]
			
			# construct symbolic direct kinematics from Craig's DH parameters
			# see https://frankaemika.github.io/docs/control_parameters.html
			# {'a':  0.088,  'd': 0.2104, 'alpha':  pi/2}
			self.dh_craig = [
				{'a':  0,      'd': 0.333, 'alpha':  0,  },
				{'a':  0,      'd': 0,     'alpha': -pi/2},
				{'a':  0,      'd': 0.316, 'alpha':  pi/2},
				{'a':  0.0825, 'd': 0,     'alpha':  pi/2},
				{'a': -0.0825, 'd': 0.384, 'alpha': -pi/2},
				{'a':  0,      'd': 0,     'alpha':  pi/2},
				{'a':  0.088,  'd': 0.2104, 'alpha':  pi/2}
			]
			#'d': 0.2104  -> 0.107 (flange) + 0.1034 (eef)

			self.DK = eye(4)
			
			# define joint limits for the Panda robot
			self.limits = [
				(-2.8973, 2.8973),
				(-1.7628, 1.7628),
				(-2.8973, 2.8973),
				(-3.0718, -0.0698),
				(-2.8973, 2.8973),
				(-0.0175, 3.7525),
				(-2.8973, 2.8973)
			]
	
	def truncate(self, f, n):
		return math.floor(f * 10 ** n) / 10 ** n
		
	def forward_K(self, qvalues):
		
		q_values = qvalues.reshape(7, 1)
		
		self.A_lamb(*(q_values.flatten()))
		
		matrix_fk =  self.A_lamb(*(q_values.flatten()))
		
		#print("matrix_fk = {}".format(matrix_fk))
		
		x = matrix_fk[9,0]
		y = matrix_fk[10,0]
		z = matrix_fk[11,0]
		
		x_truncated = self.truncate(x,3)
		y_truncated = self.truncate(y,3)
		z_truncated = self.truncate(z,3)
		
		position = np.array([x_truncated,y_truncated,z_truncated])
		
		return position
		
	def calculate_DK(self):
	
		self.DK = eye(4)
		
		for i, (p, q) in enumerate(zip(reversed(self.dh_craig), reversed(self.joint_angles))):
			d = p['d']
			a = p['a']
			alpha = p['alpha']

			ca = cos(alpha)
			sa = sin(alpha)
			cq = cos(q)
			sq = sin(q)

			transform = Matrix(
				[
					[cq, -sq, 0, a],
					[ca * sq, ca * cq, -sa, -d * sa],
					[sa * sq, cq * sa, ca, d * ca],
					[0, 0, 0, 1],
				]
			)

			self.DK = transform @ self.DK
			
	def calculate_Jacobian(self):
		self.A = self.DK[0:3, 0:4]  # crop last row
		self.A = self.A.transpose().reshape(12,1)  # reshape to column vector A = [a11, a21, a31, ..., a34]

		self.Q = Matrix(self.joint_angles)
		self.J = self.A.jacobian(self.Q)  # compute Jacobian symbolically
		
		self.A_lamb = jit(lambdify((self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7), self.A, 'numpy'), nopython=True)
		self.J_lamb = jit(lambdify((self.q1, self.q2, self.q3, self.q4, self.q5, self.q6, self.q7), self.J, 'numpy'), nopython=True)
		
	#@jit(forceobj=True,cache=True)
	@jit(forceobj=True)
	def incremental_ik(self, q, A, A_final, step=0.1, atol=1e-4, maxNumSteps=100 ):
		numSteps=0
		while True:
			delta_A = (A_final - A)
			if np.max(np.abs(delta_A)) <= atol or numSteps>maxNumSteps:
				break
			J_q = self.J_lamb(q[0,0], q[1,0], q[2,0], q[3,0], q[4,0], q[5,0], q[6,0])
			J_q = J_q / np.linalg.norm(J_q)  # normalize Jacobian
			
			# multiply by step to interpolate between current and target pose
			delta_q = np.linalg.pinv(J_q) @ (delta_A*step)
			
			q = q + delta_q
			A = self.A_lamb(q[0,0], q[1,0],q[2,0],q[3,0],q[4,0],q[5,0],q[6,0])
			numSteps+=1
		return q, numSteps, np.max(np.abs(delta_A))

	def project_point(self, Euler, aPoint):
		
		alpha = Euler[0]
		beta = Euler[1]
		gamma = Euler[2]

		r00 = math.cos(beta)*math.cos(gamma)
		r01 = (math.sin(alpha)*math.sin(beta)*math.cos(gamma)) - (math.cos(alpha)*math.sin(gamma))
		r02 = (math.cos(alpha)*math.sin(beta)*math.cos(gamma)) + (math.sin(alpha)*math.sin(gamma))

		r10 = math.cos(beta)*math.sin(gamma)
		r11 = (math.sin(alpha)*math.sin(beta)*math.sin(gamma)) + (math.cos(alpha)*math.cos(gamma))
		r12 = (math.cos(alpha)*math.sin(beta)*math.sin(gamma)) - (math.sin(alpha)*math.cos(gamma))

		r20 = -math.sin(beta)
		r21 = math.sin(alpha)*math.cos(beta)
		r22 = math.cos(alpha)*math.cos(beta)

		rt_matrix = np.array([
				[r00, r01, r02],
				[r10, r11, r12],
				[r20, r21, r22]
			])

		return np.dot(aPoint,rt_matrix)

	def to_rt_matrix_numpy_from_euler(self, T, Euler):
		# 4x4 RT matrix
		"""
			[
				[r00, r01, r02, T[0]],
				[r10, r11, r12, T[1]],
				[r20, r21, r22, T[2]],
				[0, 0, 0, 1]
			]
		"""
			
		alpha = Euler[0]
		beta = Euler[1]
		gamma = Euler[2]
		
		r00 = math.cos(beta)*math.cos(gamma)
		r01 = (math.sin(alpha)*math.sin(beta)*math.cos(gamma)) - (math.cos(alpha)*math.sin(gamma))
		r02 = (math.cos(alpha)*math.sin(beta)*math.cos(gamma)) + (math.sin(alpha)*math.sin(gamma))
		
		r10 = math.cos(beta)*math.sin(gamma)
		r11 = (math.sin(alpha)*math.sin(beta)*math.sin(gamma)) + (math.cos(alpha)*math.cos(gamma))
		r12 = (math.cos(alpha)*math.sin(beta)*math.sin(gamma)) - (math.sin(alpha)*math.cos(gamma))
		
		r20 = -math.sin(beta)
		r21 = math.sin(alpha)*math.cos(beta)
		r22 = math.cos(alpha)*math.cos(beta)
		
		rt_matrix = [[r00],[r10],[r20],[r01],[r11],[r21],[r02],[r12],[r22],[T[0]],[T[1]],[T[2]]]
		
		return rt_matrix

	def to_rt_matrix_numpy_from_quaternion(self, T, Q):
	
		# Extract the values from Q

		qx = Q[0]
		qy = Q[1]
		qz = Q[2]
		qw = Q[3]

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
		"""
		rt_matrix = Matrix(
				[
					[r00, r01, r02, T[0]],
					[r10, r11, r12, T[1]],
					[r20, r21, r22, T[2]],
					[0, 0, 0, 1]
				]
		)
		
		rt_matrix = rt_matrix[0:3, 0:4]  # crop last row
		rt_matrix = rt_matrix.transpose().reshape(12,1)
		"""
		rt_matrix = [[r00],[r10],[r20],[r01],[r11],[r21],[r02],[r12],[r22],[T[0]],[T[1]],[T[2]]]
		
		return rt_matrix

	def init_IK_DH(self):
		
		print("calculate_DK")
		start = datetime.now()
		self.calculate_DK()
		print("calculate_DK time elapsed = {}\n".format(datetime.now()-start))
		
		print("calculate_Jacobian")
		start = datetime.now()
		self.calculate_Jacobian()
		print("calculate_Jacobian time elapsed = {}\n".format(datetime.now()-start))
		
		print("start init IK DH !")
		print("calculate _A_init")
		start = datetime.now()
		self._q_init = np.array([0.0,0.0,0.0,-math.pi/2.,0.0,math.pi/2,math.pi/4.], dtype=np.float64).reshape(7, 1)
		self._A_init = self.A_lamb(*(self._q_init.flatten()))
		print("calculate _A_init time elapsed = {}\n".format(datetime.now()-start))
		
		print("calculate A_final")
		start = datetime.now()
		q_rand = np.array([np.random.uniform(l, u) for l, u in self.limits], dtype=np.float64).reshape(7, 1)
		A_final = self.A_lamb(*(q_rand).flatten())
		print("calculate _A_init time elapsed = {}\n".format(datetime.now()-start))
		
		print("calculate incremental_ik")
		start = datetime.now()
		self.incremental_ik(self._q_init, self._A_init, A_final, atol=1e-3)
		print("calculate incremental_ik time elapsed = {}\n".format(datetime.now()-start))
		
		print("end init IK DH !")
