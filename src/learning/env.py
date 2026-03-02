import numpy as np

from engine.state import GameState
from engine.simulate import apply_action, generate_legal_actions
from engine.action import BuildRoad, BuildSettlement, BuildCity, TradeWithBank, EndTurn

from learning.action_mapping import decode_action


class CatanEnv:
    def __init__(self):
        self.action_space_size = 201
        self.state = None

    def reset(self):
        self.state = self._initialize_game()
        return self.state.state_to_tensor()

    def step(self, action_id):
        decoded = decode_action(action_id)

        self.state = apply_action(
            self.state,
            self.state.current_player_idx(),
            decoded
        )

        reward = self._compute_reward()
        done = self._check_done()

        return self.state.state_to_tensor(), reward, done, {}

    def legal_action_mask(self, state):
        legal_actions = generate_legal_actions(state)
        mask = np.zeros(self.action_space_size, dtype=np.bool_)

        for action in legal_actions:
            idx = self.encode_action(action)
            mask[idx] = True

        return mask
