![bomber test](https://github.com/eidelen/bomber/actions/workflows/testBomberEnv.yml/badge.svg)

# bomber
Reinforcement learning project with own gym. This is the self chosen final assignment of an advanced machine learning course. 

## Simple Bomber Environment
Given a simple 10 x 10 gridword, there is an agent surrounded by rocks. 
The main goal of the agent is to blast all the rocks. 
The agent's actions are moving up, down, left, right and detonate a bomb.
The detonation destroys all rocks within a 3x3 square around the bomb, respectively the agent.
<img src="https://github.com/eidelen/bomber/blob/main/rsc/explanation.png" width="700">
<img src="https://github.com/eidelen/bomber/blob/main/rsc/simple-episode.gif" width="600">


## Installation Notes
### Windows (Python 3.9 and 3.10)
```
python -m venv venv
.\venv\Scripts\activate.ps1    # if problem: Open Powershell as Admin: > Set-ExecutionPolicy RemoteSigned
python -m pip install -U pip
python -m pip install -U setuptools
pip install torch torchvision
pip install -U "ray[rllib]"
pip install tqdm tensorboard matplotlib
```

### OSX
```
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -U setuptools
pip install torch torchvision
pip install -U "ray[rllib]"
pip install tqdm tensorboard matplotlib
```
