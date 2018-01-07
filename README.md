# sparse-model-learning

Code for Yonadav Shavit's Masters Thesis, "[Learning Environment Representations from Sparse Signals](http://yonadavshavit.com/assets/files/masters-engineering-thesis.pdf)"

The code was written for Tensorflow 1.0 or later, and the system heavily utilizes Tensorboard for visualizing learned embeddings.

![Learning architecture](/figures/meng-network-diagram.png)

For an in-depth explanation of the algorithm, see [this blog post](http://yonadavshavit.com/Masters-Thesis/).

## Table of Contents
`train_model.py` is the primary file for running experiments, including training and visualizing environment models.
 Call `python train_model.py -h` for the possible pre-configured experiments you can run.

`model.EnvModel` defines the environment-simulating network architecture.

`modellearner.ModelLearner` wraps `EnvModel` and implements functions to gather environment training data and train the model.

`configs.py` contains a large set of possible experiment configurations, and a simple interface for defining new experiments.

`agents.py` defines different RL agents utilizing the learned models, including a BFS-based planner and a random-rollout planner.

`test_planning.py` lets you compare an agent's learned environment representation to the real environment.
 It does this by letting you take actions and displaying side-by-side the true state and the agent's best guess of the current state.
 
`run_analytics.py`, `feature_analytics.py`, and `bfs_analytics.py` generate the respective figures for Chapter 3, Chapter 4, and the final model-based agents' performance.

![Example learned latent space](/figures/Linear_3step_full.png)

(A learned environment embedding)
