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


import pybullet as p
import pybullet_data
import threading
import time
import os, inspect

class Environment():
	
	def __init__(self, json_decoder=None, gui = True):
		
		if json_decoder==None:
			raise RuntimeError("=> Environment class need a JSON Decoder, to get some parameters !!!")
			return
			
		self.gui = gui
		
		# read pybullet parameters from json config file
		self.time_step = json_decoder.config_data["env"]["pybullet_parameters"]["time_step"]
		self.n_substeps = json_decoder.config_data["env"]["pybullet_parameters"]["n_substeps"]
		self.time_step_sleep = json_decoder.config_data["env"]["pybullet_parameters"]["time_step_sleep"]
		
		# init running by default to False
		self.running = False
		
		print("** ENVIRONNEMENT PARAMETERS **")
		print("time_step_sleep = {}, n_substeps = {}, gui = {}".format(self.time_step_sleep, self.n_substeps, self.gui))
		
		p.connect(p.GUI if self.gui else p.DIRECT)
		
		self.reset()
	
	def startThread(self):
		
		self.start()
		self.step_thread = threading.Thread(target=self.do_step_simulation)
		self.step_thread.daemon = True
		self.step_thread.start()
		
	def do_step_simulation(self):
		while True:
			if self.running:
				p.stepSimulation()
			time.sleep(self.time_step_sleep)
			
	
	def pause(self):
		self.running = False
		
	def start(self):
		self.running = True
			
	def reset(self):
		
		self.pause()
			
		# reset pybullet to deformable object
		p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)

		# bullet setup
		# add pybullet path
		currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
		#print("currentdir = {}".format(currentdir))
		p.setAdditionalSearchPath(currentdir)
		
		#p.setPhysicsEngineParameter(numSolverIterations=150, numSubSteps = self.n_substeps)
		p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)

		# Set Gravity to the environment
		p.setGravity(0, 0, -9.81)
		#p.setGravity(0, 0, 0)
		
		#p.setTimeStep(self.time_step)
		
		#p.setRealTimeSimulation(1)
		
		self.start()
