![bomber test](https://github.com/eidelen/bomber/actions/workflows/testBomberEnv.yml/badge.svg)

# bomber
Reinforcement learning project with own gym. This is the self chosen final assignment of an advanced machine learning course at IDSIA, SUPSI Lugano. 

## Bomber Environment
Given a simple 10 x 10 gridword, there is an agent surrounded by rocks. 
The main goal of the agent is to blast all the rocks. 
The agent's actions are moving up, down, left, right and detonate a bomb.
The detonation destroys all rocks within a 3x3 square around the bomb, respectively the agent.
The agent gets a reward for each blasted rock and a special reward when all rocks are destroyed.
On the other side, the agent gets a penalty when colliding with the walls or rocks, when planting a bomb and for each move.
All these environment properties can be customized.

<img src="https://github.com/eidelen/bomber/blob/main/rsc/explanation.png" width="460">


## The Simple Bomber
In the simple bomber case, the bomb blasts immediatly after the agent plants the bomb.
Therefore the agent does not get a penalty when being close to the blasts.
I tried many different varations of learning, like agent dies or gets penalty when colliding with rocks and wall.

<img src="https://github.com/eidelen/bomber/blob/main/rsc/simple-episode.gif" width="365">

## The Reduced Simple Bomber
The reduced simple bomber has the same properties as the above simple bomber. 
But the reduced bomber agent only observes its surrounding fields - therefore the observation space is reduced to a patch of 3x3. 
This leads to a lack of knowledge for the agent, but also increases the learning speed and most importantly, one agent can be applied on different grid sizes.

<img src="https://github.com/eidelen/bomber/blob/main/rsc/trained6x6-put-6x6.gif" width="300"><img src="https://github.com/eidelen/bomber/blob/main/rsc/trained6x6-put-8x8.gif" width="300"><img src="https://github.com/eidelen/bomber/blob/main/rsc/trained6x6-put-10x10.gif" width="300">

Above agent was trained on a 6x6 grid, but then also tested on an 8x8 and 10x10 play area.
Unlike the simple bomber, the model here uses an additional LSTM layer.
One can see that the agent isn't performing great on unknown grid sizes. 


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
