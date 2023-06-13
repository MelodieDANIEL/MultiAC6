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

-> revoir le dossier train_tests

-> ajouter des options
1 thread pour afficher des croix (peut-être faire une version graphique)

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




 
