from collections import namedtuple, deque
import os
import json
import pickle
from typing import List

import events as e
from .features import state_to_features

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


def reward_from_events(events_list: List[str]) -> float:
    return sum(REWARDS.get(ev, 0) for ev in events_list)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

TRANSITION_HISTORY_SIZE = 3
RECORD_ENEMY_TRANSITIONS = 1.0


def setup_training(self):
    """
    Initialise self for training purpose. Called once, after setup() in callbacks.py.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step with the events that occurred as a result of the last action.
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    reward = reward_from_events(events)
    self.transitions.append(Transition(
        state_to_features(old_game_state),
        self_action,
        state_to_features(new_game_state),
        reward,
    ))
    # TODO: use self.transitions to update your model (Q-table update / NN training step)


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called once at the end of each round.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    reward = reward_from_events(events)
    self.transitions.append(Transition(
        state_to_features(last_game_state),
        last_action,
        None,
        reward,
    ))

    # Log score for this round (for plotting training progress later)
    score = last_game_state['self'][1]
    if not hasattr(self, 'score_log'):
        self.score_log = []
    self.score_log.append(score)

    log_path = os.path.join(os.path.dirname(__file__), 'training_scores.json')
    with open(log_path, 'w') as f:
        json.dump(self.score_log, f)

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)
