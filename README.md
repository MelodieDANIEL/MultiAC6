<br>The paper "bla bla bla", written by Mélodie Hani Daniel Zakaria<a href="#note1" id="note1ref"><sup>1</sup></a>, Miguel Aranda<a href="#note2" id="note2ref"><sup>2</sup></a>, Laurent Lequièvre<a href="#note1" id="note1ref"><sup>1</sup></a>, Sébastien Lengagne<a href="#note1" id="note1ref"><sup>1</sup></a>, Juan Antonio Corrales Ramón<a href="#note3" id="note3ref"><sup>3</sup></a> and Youcef Mezouar<a href="#note1" id="note1ref"><sup>1</sup></a>, has been accepted for publication in the proceedings of bla bla bla.
<br>
<br>We address the problem of bla bla bla.

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
<br>example of config_file : './trains_tests/simulation/test/3d/with_orientation/from_db/reward_max/small_0_05/config.json'

## How to train

cd DDPG_GPU_MPI
<br>mpirun -n 32 python main_json.py --mode train --config_file './trains_tests/simulation/train/3d/with_orientation/reward_max/config.json'

<br>The database used is a file named 'db.txt' in the directory './trains_tests/simulation/train/3d/with_orientation/reward_max'
<br>The neural network weights will be saved in the directory './trains_tests/simulation/train/3d/with_orientation/reward_max/env_panda'

## How to test

cd DDPG_GPU_MPI
<br>python main_json.py --mode test --config_file './trains_tests/simulation/test/3d/with_orientation/from_db/reward_max/small_0_05/config.json' --gui true

<br>The database used is a file named 'db.txt' in the directory './trains_tests/simulation/test/3d/with_orientation/from_db/reward_max/small_0_05'
<br>The neural network weights will be saved in the directory './trains_tests/simulation/test/3d/with_orientation/from_db/reward_max/small_0_05/env_panda'

# RAL

## Abstract

## Videos

<br>
<br><a id="note1" href="#note1ref"><sup>1</sup></a>CNRS, Clermont Auvergne INP, Institut Pascal,  Université Clermont Auvergne, Clermont-Ferrand, France.
<br><a id="note2" href="#note2ref"><sup>2</sup></a>Instituto de Investigación en Ingeniería de Aragón, Universidad de Zaragoza, Zaragoza, Spain.
<br><a id="note3" href="#note3ref"><sup>3</sup></a>Centro Singular de Investigación en Tecnoloxías Intelixentes (CiTIUS),  Universidade de Santiago de Compostela, Santiago de Compostela, Spain.

