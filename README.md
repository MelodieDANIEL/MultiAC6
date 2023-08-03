<br>The paper "Multi Actor-Critic DDPG for Robot Action Space Decomposition: A Framework to Control Large 3D Deformation of Soft Linear Objects", written by Mélodie Daniel<a href="#note1" id="note1ref"><sup>1</sup></a>, Aly Magassouba<a href="#note1" id="note1ref"><sup>1</sup></a>, Miguel Aranda<a href="#note2" id="note2ref"><sup>2</sup></a>, Laurent Lequièvre<a href="#note1" id="note1ref"><sup>1</sup></a>, Juan Antonio Corrales Ramón<a href="#note3" id="note3ref"><sup>3</sup></a>, Roberto Iglesias Rodriguez<a href="#note3" id="note3ref"><sup>3</sup></a> and Youcef Mezouar<a href="#note1" id="note1ref"><sup>1</sup></a>, has been submitted for publication in RA-L.

## Abstract

Robotic manipulation of deformable linear objects (DLO) has great potential for applications in diverse fields such as agriculture or industry. However, a major challenge lies in acquiring accurate deformation models that describe the relationship between robot motion and DLO deformations. Such models are difficult to calculate analytically and vary among DLOs. Consequently, manipulating DLOs poses significant challenges, particularly in achieving large deformations that require highly accurate global models. To address these challenges, this paper presents MultiAC6: a new multi Actor-Critic framework for robot action space decomposition to control large 3D deformations of DLOs. In our approach, two deep reinforcement learning (DRL) agents orient and position a robot gripper to deform a DLO into the desired shape. Unlike previous DRL-based studies, MultiAC6 is able to solve the sim-to-real gap, achieving large 3D deformations up to 40 cm in real-world settings. Experimental results also show that MultiAC6 has a 66\% higher success rate than a single-agent approach. Further experimental studies demonstrate that MultiAC6 generalizes well, without retraining, to DLOs with different lengths or materials. 

## Demonstration video

<a href="https://drive.google.com/drive/folders/1brasS0PqnyRLl7BEYrRNotywiQU_-9iO?usp=sharing">
<p align="center">
 <img src="Images/ROS_cat.png" width=600>
    <br> 
    <em>Title : A demonstration video.</em>
</p>
</a>


## Extra real world deformations videos

<a href="https://drive.google.com/drive/folders/1QA5LIckCGPSYDqCitc6a2zXo_HS1pCXU?usp=sharing">
<p align="center">
 <img src="Images/ROS_cat.png" width=600>
    <br> 
    <em>Title : A ROS VIDEO ...</em>
</p>
</a>

## How to install virtualenv on ubuntu 20.04

virtualenv is a tool to create lightweight “virtual environments” with their own site directories isolated from system site directories.
Each "virtual environment" has its own Python binary (which matches the version of the binary that was used to create this environment) 
and can have its own independent set of installed Python packages in its site directories.

sudo apt install python3-pip
<br>pip3 install virtualenv

## How to create a virtualenv named 've_rl' and activate it

virtualenv ve_rl --python=python3
<br>source ve_rl/bin/activate

## Then necessary python3 packages can be installed into that virtualenv

pip install --upgrade pip

<br>pip install gym
<br>pip install torch
<br>pip install matplotlib
<br>pip install mpi4py
<br>pip install pybullet
<br>pip install numba
<br>pip install pyyaml
<br>pip install rospkg

## Details of 'main_json.py' parameters

<br>gui : boolean to show or not the gui (False by default).
<br>mode : string that enable the main file to be launched into a particular mode (train by default). Modes values are train, test, test_real, generate_db.
<br>config_file : string that contains a path (directory) concatenate to the name of the config file (config.json).
<br>example of config_file : './trains_tests/simulation/test/MultiAC6Star/reward_max/small_0_05/config.json'

## Details of config folder

<br>The configuration folder must constain :
<br>3 files at its root :
<br>A file named 'db.txt' containing a database of deformations to be reached by the deformable object.
<br>A file named 'frite.vtk' containing the 3D graphical structure of the deformable object.
<br>A file named 'config.json' containing the parameters necessary for the 'Gym' environment.
<br>2 subfolders :
<br>'env_panda' containing actor/critic neural weigths necessary to achieve a deformation.
<br>'env_rotation_gripper' containing actor/critic neural weights necessary to achieve initial gripper orientation.
<br>Note that, the actor and critic files are named 'actor.pth' and 'critic.pth'.

## Some useful parameters to adapt in the config file 'config.json'

<br>By changing some parameters, you can define different behavior during training and testing mode. 
<br>The initial rotation gripper :
<br>'env'/'panda_parameters'/'orientation_gripper'/'index' : 0 = 'from_initial' (set as initial position), 1 = 'from db' (set as value contain in db), 2 = 'from_agent' (set as the result of the rotation gripper agent).
<br>The reward calculation method :
<br>'env'/'reward_parameters'/'reward_index' : 0 = 'mean' (mean distance error), 1 = 'max' (max distance error), 2 = 'dtw' (sum distance error).
<br>The action space :
<br>'env'/'all_spaces'/'action_space'/'index' : 0 = '3d' (3D movement of the gripper), 1 = '6d' (6D movement of the gripper).
<br>Set number of episodes and steps per episode
<br>'env'/'env_test'/'n_episodes' or 'env'/'env_train'/'n_episodes'
<br>'env'/'env_test'/'n_steps' or 'env'/'env_train'/'n_steps'
<br>The name of the log file :
<br>'env'/'log'/'name' : by default the name is 'log_main_json.txt'.

## How to generate a database of deformations

cd DDPG_GPU_MPI
python main_json.py --mode generate_db --config_file './trains_tests/simulation/generate_db/with_orientation/small/config.json'

The generation of the db takes into account the parameters included in 'config.json' (array specified in the key 'env'/'database'/'frite_parameters'/'parameters_array').
Each element of this array contains the pose and goal space dimension as well as the number of deformations to generate.
The database will be created in a file named 'generate_db.txt'.
If you want to use this generated new db, you will have to copy this file into a config folder and rename it as 'db.txt'. 

## How to train Agent_o

cd DDPG_GPU_MPI
<br>mpirun -n 32 python main_rotation_gripper.py --mode train --config_file './trains_tests/simulation/train/Agent_o/config.json'

<br>The database used is a file named 'db.txt' in the directory './trains_tests/simulation/train/Agent_o'
<br>The neural network weights of the rotation gripper agent will be saved in the directory './trains_tests/simulation/train/Agent_o/env_rotation_gripper'

## How to train Agent_p

cd DDPG_GPU_MPI
<br>mpirun -n 32 python main_json.py --mode train --config_file './trains_tests/simulation/train/Agent_p/reward_max/config.json'

<br>The database used is a file named 'db.txt' in the directory './trains_tests/simulation/train/Agent_p/reward_max'
<br>The neural network weights will be saved in the directory './trains_tests/simulation/train/Agent_p/reward_max/env_panda'
<br>The neural network weights of the rotation gripper agent should be copied from the train folder (e.g., './trains_tests/simulation/train/Agent_o/env_rotation_gripper') and saved in the directory './trains_tests/simulation/train/Agent_p/reward_max/env_rotation_gripper'

## How to test in simulation MultiAC6

cd DDPG_GPU_MPI
<br>python main_json.py --mode test --config_file './trains_tests/simulation/test/MultiAC6/small_0_05/config.json' --gui true

<br>The database used is a file named 'db.txt' in the directory './trains_tests/simulation/test/MultiAC6/small_0_05'
<br>The neural network weights should be copied from the train folder (e.g., './trains_tests/simulation/train/Agent_p/reward_max/env_panda') and saved in the directory './trains_tests/simulation/test/MultiAC6/small_0_05/env_panda'
<br>The neural network weights of the rotation gripper agent should be copied from the train folder (e.g., './trains_tests/simulation/train/Agent_o/env_rotation_gripper') and saved in the directory './trains_tests/simulation/test/MultiAC6/small_0_05/env_rotation_gripper'

## How to test MultiAC6 with a real robot

cd DDPG_GPU_MPI
<br>python main_json.py --mode test_real --config_file './trains_tests/real/MultiAC6/config.json' --gui true

<br>The deformations to reach are contained in a file named "db_selected_save.txt".


<br>
<br><a id="note1" href="#note1ref"><sup>1</sup></a>CNRS, Clermont Auvergne INP, Institut Pascal,  Université Clermont Auvergne, Clermont-Ferrand, France.
<br><a id="note2" href="#note2ref"><sup>2</sup></a>Instituto de Investigación en Ingeniería de Aragón, Universidad de Zaragoza, Zaragoza, Spain.
<br><a id="note3" href="#note3ref"><sup>3</sup></a>Centro Singular de Investigación en Tecnoloxías Intelixentes (CiTIUS),  Universidade de Santiago de Compostela, Santiago de Compostela, Spain.

