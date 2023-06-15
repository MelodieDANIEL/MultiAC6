On a 2 versions
==============

1 version "mesocentre"
/home/laurent/Bureau/version_complete


1 version "tests en réel"
/home/laurent/rl_melodie/git_project/rl_melodie_mesocentre/version_complete


le 13/05 -> On est revenu à la version avec un thread d'environnement

to do list
==========

-> revoir les fichier main
1 main pour l'entrainement et test (main_json.py)
1 main pour les tests en reel (main_test_database_json.py)

Dans le main_json ajouter un mode test_reel qui remprend le mode test rl du fichier main_test_database_json.py

-> Pour faire la video, un main_json graphique
- qui affiche une seule et unique boite goal et une seule et unique boite pose.
- 1 thread pour afficher des croix (peut-être faire une version graphique)

if self.gui == True:
			print('START THREAD UPDATE CROSS !')
			self.draw_thread = threading.Thread(target=self.loop_update_cross)
			self.draw_thread.start()
    
	
	
	def loop_update_cross(self):
		
		time.sleep(5)
		
		while True:
			if self.update_cross_is_running :
				#print('TREAD UPDATE CROSS : before compute_mesh_pos_to_follow')
				self.compute_mesh_pos_to_follow(draw_normal=False)
				#print('TREAD UPDATE CROSS : after compute_mesh_pos_to_follow')
				self.draw_id_to_follow()
				time.sleep(1)


ajouter des cles dans le fichier config.json

"env":
{...


  "is_graphic_mode": true 
...
}

"graphic_spaces": {

       "goal_space":
		{
			"goal_array":[{"name": "extra_small","x_up": 0.025,"x_down": 0.05,"y_up": 0.15,"y_down": 0.15,"z_down": 0.1},{"name": "small","x_up": 0.05,"x_down": 0.1,"y_up": 0.25,"y_down": 0.25,"z_down": 0.25},{"name": "medium","x_up": 0.05,"x_down": 0.15,"y_up": 0.3,"y_down": 0.3,"z_down": 0.25},{"name": "large","x_up": 0.05,"x_down": 0.15,"y_up": 0.4,"y_down": 0.4,"z_down": 0.3}],
			"index": 1
		},
		"pose_space":
		{
			"pose_array":[{"name": "extra_small","x_up": 0.025,"x_down": 0.05,"y_up": 0.15,"y_down": 0.15,"z_down": 0.1},{"name": "small","x_up": 0.05,"x_down": 0.1,"y_up": 0.25,"y_down": 0.25,"z_down": 0.25},{"name": "medium","x_up": 0.05,"x_down": 0.15,"y_up": 0.3,"y_down": 0.3,"z_down": 0.25},{"name": "large","x_up": 0.05,"x_down": 0.15,"y_up": 0.4,"y_down": 0.4,"z_down": 0.3}],
			"index": 3
		}
}

si en mode graphique et  gui true, remonter la bonne goal space et pose space graphique et dessiner toujours ces boites.


small :
					"x_up_goal" : 0.05,
					"x_down_goal" : 0.1,
					"y_up_goal" : 0.2,
					"y_down_goal" : 0.2,
					"z_down_goal" : 0.25,
					
					"x_up_pos" : 0.05,
					"x_down_pos" : 0.15,
					"y_up_pos" : 0.325,
					"y_down_pos" : 0.325,
					"z_down_pos" : 0.3,

medium :
					"x_up_goal" : 0.05,
					"x_down_goal" : 0.15,
					"y_up_goal" : 0.25,
					"y_down_goal" : 0.25,
					"z_down_goal" : 0.25,
					
					"x_up_pos" : 0.05,
					"x_down_pos" : 0.15,
					"y_up_pos" : 0.325,
					"y_down_pos" : 0.325,
					"z_down_pos" : 0.3,

large :
					"x_up_goal" : 0.05,
					"x_down_goal" : 0.15,
					"y_up_goal" : 0.325,
					"y_down_goal" : 0.325,
					"z_down_goal" : 0.3,
					
					"x_up_pos" : 0.05,
					"x_down_pos" : 0.15,
					"y_up_pos" : 0.325,
					"y_down_pos" : 0.325,
					"z_down_pos" : 0.3,

-> revoir le dossier train_tests

-> pour le mode test
ajouter le calcul de l'erreur intitiale (c'est à dire la max_d pour une reward de type max).
Il faur modifier dans les wrapper la fonction 'reset_bullet' qui recupere une obs initiale. A partir de cette obs on peut calculer une erreur (comme dans la fonction 'setp_bullet').
Il faut ajouter cette erreur initiale dans le fichier de log.
Il faut calculer un 'max' et 'mean' de cette erreur initiale pou tous les episodes.

-> clef "do_reset_env" dans le config
obligatoirement à true si from_db ou bien from_agent.
si from_initial, on peut la mettre à true ou false sans aucun soucis.

option1 : Soit exliquer cela dans le readme.md
option2 : faire les verifs en code avant de faire le mode test ou train (lever une exception en cas d'erreur-> false si from_db ou from_agent)

en mode test:
si from_db ou from_agent et do_reset_env = false -> ERREUR

en mode train:
quelque soit le from_bidule ... le do_reset_env = false -> ERREUR

option3 : 
lorsqu'on change d'épisode et qu'une nouvelle boîte est générée, la dernière position atteinte par le gripper peut-être à l'extérieur de la nouvelle boîte. 
Il faudrait alors, cliper cette position pour etre sur que le gripper soit dans la nouvelle boite.
a en rediscuter

si pas reset env et que from_db ou from_agent alors
  cliper la pose courante du gripper dans la nouvelle boite qui est tiree au hasard lors du changement d'épisode.
  bouger le bras à cette nouvelle position clipée.
   
infos complémentaires :
---------------------
voir dans le def set_action
il y a un clip pour la newpos puis un go_to_cartesian qui execute la newpos.
à faire dans le cas de from_agent et from_db.

dans le cas de from_initial, il y a qu'une seule taille de goal et pose pour toute la db (que ce soit en 3D ou 6D)
par contre en from_agent ou from_db, dans le fichier db.txt généré il y a plusieurs tailles de boites.




Updates :
--------

python main_json.py --mode test --config_file './trains_tests/test_db_small_graphic/config.json' --gui true

panda_frite_env_complete.py :

def read_all_spaces(self):
 ...  (read graphic arrays from config.json and create dim arrays)
def set_gym_spaces(self):
 ... (create spaces.Box graphic from dim arrays )
def __init__(self, ...)
 ...
 # read JSON env properties
 self.is_graphic_mode = self.json_decoder.config_data["env"]["is_graphic_mode"]
 ...
 ...
 # For Graphic Cross Update
 self.mutex_get_mesh_data = threading.Lock()
 self.update_cross_is_running = True

 if self.gui == True and self.is_graphic_mode == True:
	print('START GRAPHIC THREAD TO UPDATE CROSS !')
	self.draw_cross_thread = threading.Thread(target=self.loop_update_cross)
	self.draw_cross_thread.start()
 ... 

def loop_update_cross(self):
 ....
 
def draw_env_box(self):
		if self.is_graphic_mode == True:
			self.debug_gui.draw_box("pos_graphic", self.pos_space_graphic.low, self.pos_space_graphic.high, [0, 0, 1])
			self.debug_gui.draw_box("goal_graphic", self.goal_space_graphic.low, self.goal_space_graphic.high, [1, 0, 0])
		else: 
			...

def compute_mesh_pos_to_follow(self, draw_normal=False):
		self.mutex_get_mesh_data.acquire()
		try:
			data = p.getMeshData(self.frite_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
		finally:
			self.mutex_get_mesh_data.release()
		....
	
main_json.py :

if args.mode == 'test':

	if (args.gui) and env.is_graphic_mode == False:
			env.draw_id_to_follow()
					
add code for : 
		# initial error
		initial_error = 0
		sum_initial_error = 0
		list_initial_error = []					


all reward wrapper :
panda_frite_env_wrapper_dtw.py
panda_frite_env_wrapper_max.py
panda_frite_env_wrapper_mean.py

def reset_bullet(self):
		obs = self.env.reset_bullet()
		
		nb_mesh_to_follow = len(self.env.position_mesh_to_follow)
		
		sum_d = 0
		
		for i in range(nb_mesh_to_follow):
			current_pos_mesh = obs[(self.env.pos_of_mesh_in_obs+(i*3)):(self.env.pos_of_mesh_in_obs+(i*3)+3)]
			goal_pos_id_frite = self.env.goal[i]
			d =  np.linalg.norm(current_pos_mesh - goal_pos_id_frite, axis=-1)
			sum_d+=d
			
		sum_d = np.float32(sum_d)
		
		reward = -sum_d
		
		return obs, reward
		
		
