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
        assert self.num_rollouts < self.envmodel.test_batchsize,\
                "Number of random rollouts must be within envmodel's default test batchsize (was {}, must be <= {})".format(self.num_rollouts, self.envmodel.test_batchsize)
        self.rollout_length = rollout_length
    def policy(self, s, gs):
        if random.random() < self.pct_random:
            return random.randrange(self.envmodel.ac_space)
        states = np.tile(np.expand_dims(s, 0), [self.num_rollouts] +
                         [1 for _ in
                          range(len(self.envmodel.ob_space))])
        actions = np.random.randint(self.envmodel.ac_space,
                                    size=[self.num_rollouts,
                                          self.rollout_length])
        latents = self.envmodel.encode(states)
        latent_goal = self.envmodel.encode(gs)
        latent_goals_flattened = np.stack([latent_goal for _ in
                                 range(self.rollout_length*self.num_rollouts)],
                                          axis=0)
        _, future_latents = self.envmodel.stepforward(latents, actions)
        future_latents_flattened = np.reshape(future_latents, [-1, self.envmodel.latent_size])
        goal_values = []
        b = self.envmodel.test_batchsize
        n_values = future_latents_flattened.shape[0]
        for i in range(n_values//b + 1):
            lower = i*b
            upper = min((i+1)*b, n_values)
            latents_chunk = future_latents_flattened[lower:upper]
            goalstates_chunk = latent_goals_flattened[lower:upper]
            goalvalues_chunk = self.envmodel.checkgoal(latents_chunk, goalstates_chunk)
            goal_values.append(goalvalues_chunk)
        goal_values = np.concatenate(goal_values, axis=0).reshape([self.num_rollouts,
                                                                   self.rollout_length])
        # we don't explicitly calculate game-overs, so
        # if anywhere in the trajectory the goal-value is high, we pick that
        # trajectory
        max_goal_per_rollout = np.max(goal_values, axis=1)
        best_rollout = np.argmax(max_goal_per_rollout)
        return actions[best_rollout]

import collections
class BFSAgent:
    def __init__(self, envmodel, horizon=5):
        self.envmodel = envmodel
        self.horizon = horizon

    def policy(self, s, sg):
        x0 = self.envmodel.encode(s)
        actions = np.arange(self.envmodel.ac_space)
        xg = self.envmodel.encode(sg)
        xg_tiled = np.tile(np.expand_dims(xg, axis=0), [len(actions), 1])
        q = collections.deque()
        depth = 0
        node0 = [x0, 0, None, None, depth] # latent, goalvalue, preceding action, parent, depth
        q.append(node0)
        best_node = None
        best_node_value = -np.Inf
        while len(q) != 0:
            node = q.popleft()
            depth = node[-1]
            if depth == self.horizon:
                print(depth)
                break
            x = node[0]
            x = np.tile(x, [4, 1])
            _, xn = self.envmodel.stepforward(x, actions)
            xn = np.squeeze(xn, axis=1)
            gn = self.envmodel.checkgoal(xn, xg_tiled)
            for a in range(len(actions)):
                newnode = [xn[a, :], gn[a], a, node, depth+1]
                q.append(newnode)
                if gn[a] > best_node_value:
                    best_node = newnode
                    best_node_value = gn[a]
        best_node_actions = []
        parent = best_node
        while parent is not None:
            best_node_actions.append(parent[2])
            parent = parent[3]
        best_node_actions.pop() # remove the None action at the end
        return list(reversed(best_node_actions)), best_node_value












