import os
import json
import pickle
from typing import List

import numpy as np

import events as e
from .features import state_to_features, ACTIONS
from .callbacks import features_to_key, get_q

REWARDS = {
    e.COIN_COLLECTED: 10,
    e.KILLED_OPPONENT: 50,
    e.KILLED_SELF: -50,
    e.GOT_KILLED: -50,
    e.CRATE_DESTROYED: 2,
    e.COIN_FOUND: 3,
    e.INVALID_ACTION: -2,
    e.WAITED: -0.5,
    e.SURVIVED_ROUND: 5,
}

ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor


def reward_from_events(events_list: List[str]) -> float:
    return sum(REWARDS.get(ev, 0) for ev in events_list)


def q_update(self, old_key, action, reward, new_key, done):
    if old_key is None or action is None:
        return
    action_idx = ACTIONS.index(action)
    old_q = get_q(self.q_table, old_key)
    if done or new_key is None:
        future = 0.0
    else:
        future = np.max(get_q(self.q_table, new_key))
    old_q[action_idx] += ALPHA * (reward + GAMMA * future - old_q[action_idx])


def setup_training(self):
    """
    Initialise self for training purpose. Called once, after setup() in callbacks.py.
    """
    pass  # self.q_table already initialized in setup()


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step with the events that occurred as a result of the last action.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(events)
    old_key = features_to_key(state_to_features(old_game_state))
    new_key = features_to_key(state_to_features(new_game_state))
    q_update(self, old_key, self_action, reward, new_key, done=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once at the end of each round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(events)
    last_key = features_to_key(state_to_features(last_game_state))
    q_update(self, last_key, last_action, reward, None, done=True)

    # DIAGNOSTIC: per-round self-kill trend over training
    if not hasattr(self, 'death_log'):
        self.death_log = []
    self.death_log.append(1 if 'KILLED_SELF' in events else 0)
    death_log_path = os.path.join(os.path.dirname(__file__), 'death_log.json')
    with open(death_log_path, 'w') as f:
        json.dump(self.death_log, f)

    # Log score for this round (for plotting training progress later)
    score = last_game_state['self'][1]
    if not hasattr(self, 'score_log'):
        self.score_log = []
    self.score_log.append(score)

    log_path = os.path.join(os.path.dirname(__file__), 'training_scores.json')
    with open(log_path, 'w') as f:
        json.dump(self.score_log, f)

    # Store the Q-table
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)
