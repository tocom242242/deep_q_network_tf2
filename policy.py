﻿import numpy as np


class EpsGreedyQPolicy:
    def __init__(self, eps=.1, eps_decay_rate=.99, min_eps=.1):
        self.eps = eps
        self.eps_decay_rate = eps_decay_rate
        self.min_eps = min_eps

    def select_action(self, q_values, is_training=True):
        nb_actions = q_values.shape[0]

        if is_training:
            if np.random.uniform() < self.eps:
                action = np.random.randint(0, nb_actions)
            else:
                action = np.argmax(q_values)
        else:
            action = np.argmax(q_values)

        return action

    def decay_eps_rate(self):
        self.eps = self.eps*self.eps_decay_rate
        if self.eps < self.min_eps:
            self.eps = self.min_eps
