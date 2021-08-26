# This project is for the University of Ulm for the project Deep Learning for Autonumous cars #
 By André Henkel Winter term 2020

## Install 
Python 3.7
openAI gym environment - CarRacing-v0: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
pytorch

Create a conda environment for that and install pytorch.
Probably some minor packages will be needed additionally as well.

## Content
You can manually drive and gather data
You can use this data to train the imitation learning agent
You can run the ddpg algorithm to train an actor-critic agent to run the simulation

## Problems
The hidden layer initialization was a massive problem at first
using uniform is very bad, gaussian helped to not directly being biased towards one action



