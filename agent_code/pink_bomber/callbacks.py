import os
import pickle
from collections import deque

import numpy as np

from .features import state_to_features, ACTIONS, get_danger_map, bfs_escape_exists

DIR_MAP = {'UP': (0, -1), 'RIGHT': (1, 0), 'DOWN': (0, 1), 'LEFT': (-1, 0)}


def features_to_key(features):
    if features is None:
        return None
    return tuple(int(f) for f in features)


def get_q(q_table, state_key):
    if state_key not in q_table:
        q_table[state_key] = np.zeros(len(ACTIONS))
    return q_table[state_key]


def urgency_horizon(bombs):
    """How many steps until the soonest bomb explodes? (countdown 0 = about to explode)."""
    if not bombs:
        return 1
    return max(1, min(timer for (_, timer) in bombs) + 1)


def get_survival_actions(game_state, base_allowed):
    """
    Multi-step lookahead: only allow actions that leave a genuine escape route
    (using the real remaining bomb countdown as the time budget), not just
    actions whose immediate next tile happens to be danger-free.
    """
    field = game_state['field']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    _, _, _, (x, y) = game_state['self']

    danger = get_danger_map(field, bombs, explosion_map)
    occupied = {b[0] for b in bombs}
    width, height = field.shape
    horizon = urgency_horizon(bombs)

    movement_survivable = []
    for action in base_allowed:
        if action == 'BOMB':
            continue
        if action == 'WAIT':
            new_pos = (x, y)
        else:
            dx, dy = DIR_MAP[action]
            new_pos = (x + dx, y + dy)
            nx, ny = new_pos
            if not (0 <= nx < width and 0 <= ny < height) or field[nx, ny] != 0 or new_pos in occupied:
                continue
        if new_pos not in danger:
            movement_survivable.append(action)
        elif bfs_escape_exists(field, danger, occupied, new_pos, horizon):
            movement_survivable.append(action)

    if movement_survivable:
        if 'BOMB' in base_allowed:
            movement_survivable.append('BOMB')
        return movement_survivable
    return base_allowed  # truly cornered -- masking can't help


def setup(self):
    self.epsilon = 0.2

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

    allowed = list(ACTIONS)
    if game_state is not None and features is not None and len(features) > 19:
        if features[19] == 0.0:
            allowed = [a for a in allowed if a != 'BOMB']
        allowed = get_survival_actions(game_state, allowed)

    if self.train and np.random.random() < self.epsilon:
        self.logger.debug("Choosing action purely at random (exploration).")
        return np.random.choice(allowed)

    q_values = get_q(self.q_table, state_key)
    q_values_allowed = {a: q for a, q in zip(ACTIONS, q_values) if a in allowed}
    max_q = max(q_values_allowed.values())
    best_actions = [a for a, q in q_values_allowed.items() if q == max_q]
    self.logger.debug(f"Q-values: {dict(zip(ACTIONS, q_values))}")
    return np.random.choice(best_actions)
