import os
import pickle

import numpy as np

from .features import state_to_features, ACTIONS


def features_to_key(features):
    """Convert a feature vector into a hashable dict key."""
    if features is None:
        return None
    return tuple(int(f) for f in features)


def get_q(q_table, state_key):
    """Get Q-values for a state, initializing to zeros if unseen."""
    if state_key not in q_table:
        q_table[state_key] = np.zeros(len(ACTIONS))
    return q_table[state_key]


def setup(self):
    """
    Setup your code. Called once when loading each agent.
    """
    self.epsilon = 0.2  # exploration rate during training

    if os.path.isfile("my-saved-model.pt"):
        self.logger.info("Loading existing Q-table.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)
    else:
        self.logger.info("Setting up Q-table from scratch.")
        self.q_table = {}


def act(self, game_state: dict) -> str:
    features = state_to_features(game_state)
    state_key = features_to_key(features)

    if self.train and np.random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random (exploration).")
        return np.random.choice(ACTIONS)

    q_values = get_q(self.q_table, state_key)
    max_q = np.max(q_values)
    best_actions = [a for a, q in zip(ACTIONS, q_values) if q == max_q]
    self.logger.debug(f"Q-values: {dict(zip(ACTIONS, q_values))}")
    return np.random.choice(best_actions)
