import torch
import time
import numpy as np

from learning.buffer import RolloutBuffer
from learning.utils import compute_gae
from learning.env import CatanEnv
from learning.agent import PPOAgent
from learning.model import ActorCritic
from map.geometry import TOTAL_ACTIONS

def train(env, agent, num_episodes=40000):
    total_start = time.time()  # Start total training timer
    # Firstly I define the buffer. This is what stores the actions that have been taken. And looks like this: 
    """
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    """
    buffer = RolloutBuffer()

    # Next we start training loop. An episode is a game.
    for episode in range(num_episodes):
        start_time = time.time()  # Track episode start
        # Firstly, reset the state, buffer and done flag.
        # The state here is actually our input vector representation of the state. 
        state = env.reset()
        done = False
        buffer.clear()
        steps = 0
        max_steps = 400
        truncated = False

        # Simulate the episode (game).
        while not done:
            steps += 1
            
            # Defines the state tensor. Converts the input vector representation of the state into something that Pytorch understands
            state_tensor = torch.tensor(state, dtype=torch.float32)

            # Defines a mask which we use to remove all the non legal actions.
            mask = env.legal_action_mask(env.state)
            
            # Selects an action out of all our legal actions
            action_id, log_prob, value = agent.select_action(
                state_tensor,
                mask
            )
            
            # Defines the next state, reward from the turn and if the episode is done.
            next_state, reward, done, _ = env.step(action_id)

            # Store transition
            buffer.states.append(state_tensor)
            buffer.actions.append(action_id)
            buffer.log_probs.append(log_prob)
            buffer.rewards.append(reward)
            buffer.dones.append(done)
            buffer.values.append(value.squeeze())
            buffer.action_mask.append(torch.from_numpy(mask).bool())

            state = next_state
            np.set_printoptions(threshold=np.inf)  # disable truncation
            # print(next_state)
            
            # Safety check: max steps reached
            if steps >= max_steps:
                truncated = True
                done = True
                print(f"Max steps {max_steps} reached, terminating episode.")

        # --- After episode ends ---

        if truncated:
            # Episode ended due to time limit — bootstrap critic
            with torch.no_grad():
                next_state_tensor = torch.tensor(state, dtype=torch.float32)
                _, last_value_tensor = model(next_state_tensor)
                last_value = last_value_tensor.item()
        else:
            # True terminal state
            last_value = 0.0

        if np.isnan(buffer.values).any() or np.isinf(buffer.values).any():
            print("Warning: value contains NaN/Inf")

        advantages, returns = compute_gae(
            buffer.rewards,
            [v.item() for v in buffer.values],
            buffer.dones,
            last_value
        )
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        agent.update(buffer, advantages, returns)
        
        # Compute entropy of the action distribution for diagnostic purposes
        if episode % 1000 == 0:
            with torch.no_grad():
                logits, _ = agent.model(state_tensor)  # shape: [201]
                probs = torch.softmax(logits, dim=-1)  # full distribution
                entropy = -(probs * torch.log(probs + 1e-12)).sum().item()  # full entropy
                print(f"Episode {episode}, Entropy: {entropy:.4f}")

        end_time = time.time()  # Track episode end
        episode_duration = end_time - start_time
        print(f"Episode {episode} complete in {episode_duration:.2f} seconds, length: {len(buffer.rewards)}")
            
    total_end = time.time()  # End total training timer
    total_duration = total_end - total_start
    avg_episode = total_duration / num_episodes
    print(f"\nTotal training time: {total_duration:.2f} seconds")
    print(f"Average time per episode: {avg_episode:.2f} seconds")
    
    
print("""
----------------------------------------------------------
---------------------- Training --------------------------
----------------------------------------------------------
""")

filename = input("Filename: ")
path     = ".\\learning\\models\\"

# Define the params
epsilon     = 3e-4      # Learning Rate
state_dim   = 1027
action_dim  = TOTAL_ACTIONS

# Create model instance first 
model = ActorCritic(state_dim=state_dim, action_dim=action_dim)

# Define Agent 
agent = PPOAgent(
    model=model,
    optimiser=torch.optim.Adam(model.parameters(), lr=epsilon),
    debug=False
)

# # Define the environment
env = CatanEnv()

# Run 
train(env, agent)

# Save the model
torch.save(model.state_dict(), f"{path}{filename}.pth")