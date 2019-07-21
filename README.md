# Navigation with DQN
This project applied Deep Reinforcement Learning to solve a banana navigation environment. To visually see the environment, you can refer to the [trained_agent.mp4](trained_agent.mp4). This smart agent is trained with the model built in the repository.

## Project Summary - Banana Navigation
Here is a high-level summary of the environment:
* Task: The task is to collect yellow bananas in a finite space which has randomly distributed yellow and blue bananas.
* Reward: For each yellow banana collected, a +1 reward is granted; For each blue banana a -1 reward is granted. The goal is to maximize the total reward in a single episode, which has finite number of movements
* State space: The state space has 37 dimensions, such as the agent's velocity, along with ray-based perception of objects around the agent's forward direction, etc. 
* Action space: The action space has 4 dimensions, that is to say, at each step there are 4 discrete actions that the agent can choose from:  move forward, move backward, turn left, and turn right.
We consider that the environment is solved when the agent can achieve an average total reward (score) of +13 in 100 consecutive episodes.

## Getting Started
1. Clone this repository.
2. Follow the instructions in [the DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies)
3. Download and unzip the Unity Environment:
    * Linux: [download]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    * Mac OSX: [download]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    * Windows (32-bit): [download]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    * Windows (64-bit): [download]( https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
   This repository has a Banana folder that's the same as the one unzipped from Windows 64-bit above.

## Instructions to Train and Agent
To train an agent or watch a smart agent, you only need to modify the [Navigation.ipynb](Navigation.ipynb) file from this repository. 
1. In part II. Environment, change the file path to ```env = UnityEnvironment(file_name=PATH_TO_THE_ENV)```. The ```PATH_TO_THE_ENV``` is the path to the app from you unzipped folder from Step 3 in Getting Started.
2. If you want to train an agent from scratch, set ```train_model = True``` in part III. Train an Agent.
3. If you just want to watch a smart agent, set ```train_model = False``` in part III. Train an Agent. The script will load the smart agent’s model weight from a pre-trained model.
4. For details about the model and the agent, please refer to [dqn_agent.py](dqn_agent.py) and [model.py](model.py).

