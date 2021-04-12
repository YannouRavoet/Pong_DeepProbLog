# PONG DeepProbLog
Made in context of the Capita Selecta Course: AI - Neurosymbolic AI (2020). 
This program runs an ai opponent for the game of Pong. It perceives the screen through 
a CNN and reasons on which action to take with first-order logic.

## INSTALL
Simply install all dependencies from the requirements.txt file with pip.

## USAGE
In the Game.py file you can comment/uncomment the following functionalities:

* generating data
* training a DeepProblog agent
* training a Pytorch agent
* running a match between a DeepProblog agent and a random agent

After having generated the data, (**before training the agents**), you need to sort the data into appropriate directories 
(since we load training/testing data as ImageFolders).
For this run the data_handling.py file.

An example dataset of 500 images is provided. However, it is advised to generate around 5000 images for 
optimal performance of the deepproblog agent. Generating data takes around 50s/1000images.

<p align="center">
	<img src="https://github.com/YannouRavoet/Pong_DeepProbLog/agentv2.gif"/>
</p>
