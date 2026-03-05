import random
import torch

from main import simulate_game

from map.generation import initialise_game 
from map.geometry import TOTAL_ACTIONS

from learning.env import CatanEnv
from learning.model import ActorCritic


from logic.strategies import HeuristicStrategy

def evaluate(model, env):
    RESOURCE_NAMES = ["Wood", "Brick", "Sheep", "Wheat", "Ore"]  # index matches resource ID
    
    # simulate a game
    game_state = initialise_game(random.randint(0, 1000))
    strategies = [HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy(), HeuristicStrategy()]
    game_history = simulate_game(strategies=strategies, env=env, game_state=game_state, n_turns=100, visualise=False, rng=game_state.rng, log=False)        

    check_amount = len(game_history)
    print(f"""
    For evaluate the model simply I have implemented this which looks at {check_amount} state from a previously simulated game \n using 4 heuristic agents. Then I can see what my ppo_agent would do given a state printing out the resulting entropy and the probs tensor.    
        """)

    for i in range(0, len(game_history), len(game_history) // check_amount):
        print(game_history[i])
        print(f"Player {game_history[i].turn % 4 + 1}'s Turn")
        
        with torch.no_grad():
            state_tensor = torch.tensor(game_history[i].state_to_tensor(), dtype=torch.float32)
            logits, value = model(state_tensor)
            
            # Legal action mask
            mask = torch.tensor(env.legal_action_mask(game_history[i]), dtype=torch.bool)

            # Mask logits for illegal actions
            masked_logits = logits.clone()
            masked_logits[~mask] = -1e9

            # Softmax only over legal actions
            probs = torch.softmax(masked_logits, dim=-1)

            # Compute entropy over legal actions
            entropy = -(probs[mask] * torch.log(probs[mask] + 1e-12)).sum().item()
            print(f"Entropy (legal only): {entropy:.4f}\n")

            print("Legal Action Probabilities:")
            legal_indices = mask.nonzero(as_tuple=False).squeeze(-1)
            
            for idx in legal_indices:
                idx = idx.item()
                p = probs[idx].item()

                # Decode index to action type
                if idx < 72:
                    action_type = "BuildRoad"
                    info = f"edge={idx}"
                elif idx < 126:
                    action_type = "BuildSettlement"
                    info = f"vertex={idx - 72}"
                elif idx < 180:
                    action_type = "BuildCity"
                    info = f"vertex={idx - 126}"
                elif idx == 180:
                    action_type = "EndTurn"
                    info = ""
                else:
                    action_type = "TradeWithBank"
                    trade_idx = idx - 181
                    give = trade_idx // 4
                    receive = trade_idx % 4
                    receive = receive if receive < give else receive + 1
                    info = f"give={give} ({RESOURCE_NAMES[give]}), receive={receive} ({RESOURCE_NAMES[receive]})"

                print(f"  {action_type:22}  |  {info:22}  |    prob = {p:.6f}")

print("""
----------------------------------------------------------
--------------------- Evaluating -------------------------
----------------------------------------------------------
""")

filename = input("Filename: ")
path     = ".\\learning\\models\\"

# Pramas
state_dim   = 1027
action_dim  = TOTAL_ACTIONS

# # Define the environment
env = CatanEnv()

# Recreate the model architecture
model = ActorCritic(state_dim=state_dim, action_dim=action_dim)

# Load parameters
model.load_state_dict(torch.load(f"{path}{filename}.pth"))

# Set to evaluation mode
model.eval()
evaluate(model, env)