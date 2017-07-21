import numpy as np
import random

class RandomRolloutAgent:
    """
    An agent that does a bunch of simulated random rollouts, and then executes the
    actions that led to the best one.
    """
    def __init__(self, envmodel, pct_random=0.5, num_rollouts=10,
                 rollout_length=10):
        self.envmodel = envmodel
        self.pct_random = pct_random
        self.num_rollouts = num_rollouts
        self.rollout_length = rollout_length
    def policy(self, s, gs):
        if random.random() < self.pct_random:
            return random.randint(self.envmodel.ac_space)
        states = np.tile(np.expand_dims(s, 0), [self.num_rollouts])
        actions = np.random.randint(self.envmodel.ac_space,
                                    size=[self.num_rollouts,
                                          self.rollout_length])
        latents = self.envmodel.encode(states)
        latent_goal = self.envmodel.encode(gs)
        latent_goals_flattened = np.stack([latent_goal for _ in
                                 range(self.rollout_length*self.num_rollouts)],
                                          axis=0)
        future_latents = self.envmodel.stepforward(latents, actions)
        future_latents_flattened = np.reshape(future_latents, [-1, self.envmodel.latent_size])
        goal_values = self.envmodel.checkgoal(future_latents_flattened,
                                                       latent_goals_flattened).reshape([self.num_rollouts,
                                                                                        self.rollout_length])
        # we don't explicitly calculate game-overs, so
        # if anywhere in the trajectory the goal-value is high, we pick that
        # trajectory
        max_goal_per_rollout = np.max(goal_values, axis=1)
        best_rollout = np.argmax(max_goal_per_rollout)
        return actions[best_rollout]



