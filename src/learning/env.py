import numpy as np
import random

from engine.state import GameState
from engine.simulate import execute_action, generate_legal_actions, roll, resource_production, seed_starting_positions, starting_resources, cap_player_resources
from engine.action import GameAction, BuildRoad, BuildSettlement, BuildCity, TradeWithBank, EndTurn

from logic.strategies import HeuristicStrategy

from map.generation import initialise_game

from tools.visualiser import draw
from copy import deepcopy


class CatanEnv:
    def __init__(self):
        self.action_space_size = 201
        self.state = None
        self.seed = None

    def reset(self):
        self.state = initialise_game(self.seed)
        
        # Randomly place the two settlement and roads for each player using the heuristic agent 
        strategies = [HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy()]
    
        for i in range(len(self.state.players)):
            seed_starting_positions(self.state, i, strategies[i], random.Random(self.seed))

        for i in range(len(self.state.players)):
            idx = len(self.state.players) - 1 - i
            seed_starting_positions(self.state, idx, strategies[idx], random.Random(self.seed))
    
        starting_resources(self.state)
                
        # These two lines are for debuging. 
        # draw(self.state)
        # print(self.state)
    
        return self.state.state_to_tensor()

    def step(self, action_id):
        player_idx = self.state.get_current_player_idx()
        # prev_vp = self.state.players[player_idx].victory_points
        
        # Decode the action ID into the proper Action object
        decoded = self.decode_action(action_id)

        # --- Handle Bank Trades ---
        if isinstance(decoded, TradeWithBank):
            # Find the legal TradeWithBank object that matches give/receive
            matched = False
            for action in generate_legal_actions(self.state):
                if (isinstance(action, TradeWithBank) and
                    action.give_resource == decoded.give_resource and
                    action.receive_resource == decoded.receive_resource):
                    decoded = action  # now decoded has correct give_amount
                    matched = True
                    break
            if not matched:
                raise ValueError("Selected trade action is not legal!")

        # if isinstance(decoded, BuildSettlement) or isinstance(decoded, BuildCity):
        #     print(decoded)

        prev_state = deepcopy(self.state)
        
        self.state = execute_action(
            self.state,
            player_idx,
            decoded
        )

        # --- Compute reward and done flag ---
        reward = self._compute_reward(prev_state)
        done = self._check_done()
        
        # --- Turn transition and dice/resource mechanics ---
        if isinstance(decoded, EndTurn):
            # Advance the turn counter
            self.state.turn += 1

            # Roll the dice for the next player's turn
            dice_roll = roll(self.state.rng)

            # Distribute resources according to dice roll
            resource_production(self.state, dice_roll)


        # Return the flattened tensor, reward, done, and info dict
        return self.state.state_to_tensor(), reward, done, {}

    def legal_action_mask(self, state):
        mask = np.zeros(self.action_space_size, dtype=np.bool_)

        legal_actions = generate_legal_actions(state)

        for action in legal_actions:

            # Collapse trade ratios to direction-only encoding
            if isinstance(action, TradeWithBank):

                # Ignore give_amount from generator
                collapsed_action = TradeWithBank(
                    give_resource=action.give_resource,
                    give_amount=0,   # dummy; ignored by encoder
                    receive_resource=action.receive_resource,
                    receive_amount=1
                )

                idx = self.encode_action(collapsed_action)

            else:
                idx = self.encode_action(action)

            if idx < 0 or idx >= self.action_space_size:
                raise RuntimeError(f"Invalid action index {idx}")

            mask[idx] = True
            
        if not mask.any():
            raise RuntimeError("No legal actions produced for state")

        return mask
    
    def decode_action(self, action_id: int) -> GameAction:
        """
        Decode an action ID into the corresponding Action object.
        Assumes:
        0-71   : BuildRoad
        72-125 : BuildSettlement
        126-179: BuildCity
        180    : EndTurn
        181+   : Bank trades (TradeWithBank)
        """
        num_resources = 5  # wood, brick, wheat, sheep, ore

        # --- Build Actions ---
        if action_id < 72:
            return BuildRoad(action_id)
        elif action_id < 126:
            return BuildSettlement(action_id - 72)
        elif action_id < 180:
            return BuildCity(action_id - 126)
        elif action_id == 180:
            return EndTurn()

        # --- Bank Trades ---
        trade_index = action_id - 181
        give_resource = trade_index // (num_resources - 1)
        receive_adj = trade_index % (num_resources - 1)
        receive_resource = receive_adj if receive_adj < give_resource else receive_adj + 1

        # Here we cannot infer give_amount; it will be set when generating legal actions
        return TradeWithBank(
            give_resource=give_resource,
            give_amount=None,       # placeholder; real amount comes from the Action object
            receive_resource=receive_resource,
            receive_amount=1
        )
        
    def encode_action(self, action: GameAction) -> int:
        """
        Map an Action object back to an action_id.
        Assumes:
        BuildRoad, BuildSettlement, BuildCity, EndTurn, TradeWithBank
        """
        num_resources = 5

        if isinstance(action, BuildRoad):
            return action.edge
        
        elif isinstance(action, BuildSettlement):
            return 72 + action.vertex
        
        elif isinstance(action, BuildCity):
            return 126 + action.vertex
        
        elif isinstance(action, EndTurn):
            return 180
        
        elif isinstance(action, TradeWithBank):
            give = action.give_resource
            receive = action.receive_resource
            receive_adj = receive if receive < give else receive - 1
            trade_index = give * (num_resources - 1) + receive_adj
            return 181 + trade_index
        
        else:
            raise ValueError(f"Unknown action type: {type(action)}")
        
    # def _compute_reward(self, prev_vp) -> float:
    #     player = self.state.players[self.state.get_current_player_idx()]

    #     reward = 0.0

    #     # Reward victory points
    #     new_vp = player.victory_points

    #     # Reward a point per victory point gained in a turn
    #     reward = new_vp - prev_vp
        
    #     # Terminal bonus
    #     if self.state.is_terminal():
    #         winner = self.state.get_winner()
    #         reward += 5.0 if winner == self.state.get_current_player_idx() else -5.0

    #     return reward
    
    def _compute_reward(self, prev_state) -> float:
        """
        Reward based on structured state deltas.
        Uses only true game variables — not tensor reconstruction.
        """

        player_idx = self.state.get_current_player_idx()

        player = self.state.players[player_idx]
        prev_player = prev_state.players[player_idx]

        reward = 0.0

        # --------------------------------------------------
        # 1. Victory Point Progress (primary objective)
        # --------------------------------------------------
        vp_delta = player.victory_points - prev_player.victory_points
        reward += 8.0 * vp_delta

        # --------------------------------------------------
        # 2. Settlement placement
        # --------------------------------------------------
        settlement_delta = len(player.settlements) - len(prev_player.settlements)
        reward += 3.0 * settlement_delta

        # --------------------------------------------------
        # 3. City upgrades
        # --------------------------------------------------
        city_delta = len(player.cities) - len(prev_player.cities)
        reward += 5.0 * city_delta

        # --------------------------------------------------
        # 4. Road building (light shaping only)
        # --------------------------------------------------
        road_delta = len(player.roads) - len(prev_player.roads)
        reward += 0.3 * road_delta

        # --------------------------------------------------
        # 5. Resource acquisition (very small shaping)
        # --------------------------------------------------
        resource_delta = int(np.sum(player.resources)) - int(np.sum(prev_player.resources))
        reward += 0.2 * resource_delta

        # --------------------------------------------------
        # 6. Resource spending penalty (prevents hoarding reward exploit)
        # --------------------------------------------------
        if resource_delta < 0:
            reward += 0.1 * resource_delta  # small penalty for losing resources

        # --------------------------------------------------
        # 7. Terminal condition (dominant signal)
        # --------------------------------------------------
        if self.state.is_terminal():
            winner = self.state.get_winner()
            reward += 100.0 if winner == player_idx else -100.0

        return reward
    
    def _check_done(self) -> bool:   
        return self.state.is_terminal()