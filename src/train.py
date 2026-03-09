import os

import torch
import time
import numpy as np

from learning.buffer import RolloutBuffer
from learning.utils import compute_gae
from learning.env import CatanEnv
from learning.agent import PPOAgent
from learning.model import ActorCritic
from map.geometry import TOTAL_ACTIONS
from tools.training_metrics import LossMetrics

def train(env, agent, model_name, num_episodes=40000, metrics=None, metric_freq=100):
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
            state_tensor = torch.from_numpy(state).float()

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
        mean_episode_reward = np.sum(buffer.rewards)
        values = torch.stack(buffer.values).detach()
        
        if truncated:
            # Episode ended due to time limit — bootstrap critic
            with torch.no_grad():
                next_state_tensor = torch.tensor(state, dtype=torch.float32)
                _, last_value_tensor = agent.model(next_state_tensor)
                last_value = last_value_tensor.item()
        else:
            # True terminal state
            last_value = 0.0

        if torch.isnan(torch.stack(buffer.values)).any():
            print("Warning: value contains NaN")

        advantages, returns = compute_gae(
            buffer.rewards,
            [v.item() for v in buffer.values],
            buffer.dones,
            last_value
        )
        
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        policy_loss, value_loss, entropy, kl_div = agent.update(
            buffer,
            advantages,
            returns
        )

        if episode % metric_freq == 0:
            metrics.policy_losses.append(policy_loss)
            metrics.value_losses.append(value_loss)
            metrics.entropy_losses.append(entropy)
            metrics.kl_divergences.append(kl_div)
            metrics.episodes.append(episode)
            
            metrics.mean_rewards.append(mean_episode_reward)
            
            metrics.compute_explained_variance(
                values.cpu().numpy(),
                returns.cpu().numpy()
            )
        
        # Compute entropy of the action distribution for diagnostic purposes
        if episode % (metric_freq * 100) == 0 and episode != 0:
            print(f"Saving model and metrics at episode {episode}...")
            torch.save(model.state_dict(), f"{model_name}\\model.pth")
            metrics.save(f"{model_name}\\metrics") 
            metrics.save_plot(model_name)  # Plot losses every 100 metric updates (i.e., every 10,000 episodes if metric_freq=100)

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

model_name = input("Model Name: ").strip()

num_episodes = int(input("Number of Episodes: "))
# Define Training Metrics Object 
metrics = LossMetrics()

metric_freq = int(input("Metric Frequency (episodes): "))  # How often to record metrics (in episodes)
existing_model = input("Use Existing Model (y/n): ").strip()

# Define the path
path = ".\\learning\\settlers_ppo_training\\"


# Model Params
epsilon     = 3e-5      # Learning Rate
state_dim   = 1027
action_dim  = TOTAL_ACTIONS

# Create model instance first 
model = ActorCritic(state_dim=state_dim, action_dim=action_dim)

# Load existing learnt model params 
if existing_model == 'y': # If user provided an existing model filename, load it
    existing_models = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    input_model = input(f"Existing Models: {existing_models}\nEnter model name to load: ").strip()
    if input_model not in existing_models:
        print("Invalid input for existing model. Starting with a new model.")    
        print("Exiting Program")
        exit()

    else:
        model.load_state_dict(torch.load(f"{path}{input_model}\\model.pth"))
        metrics.load(f"{path}{input_model}\\metrics")  # Load corresponding training metrics
        print(f"Loaded existing model ({input_model}) and its training metrics.")
    
        # Create the folder inside path (in case it doesn't exist, or to ensure plots folder exists)
        full_path = os.path.join(path, model_name)
        os.makedirs(full_path, exist_ok=True)
        os.makedirs(f"{full_path}\\plots\\", exist_ok=True)
        print(f"Folder created: {full_path}")
    

elif existing_model == 'n':
    print("Starting with a new model.")
    
    # Create the folder inside path
    full_path = os.path.join(path, model_name)
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(f"{full_path}\\plots\\", exist_ok=True)

    print(f"Folder created: {full_path}")
else:
    print("Invalid input for using existing model")    
    print("Exiting Program")
    exit()

# Define Agent 
agent = PPOAgent(
    model=model,
    optimiser=torch.optim.Adam(model.parameters(), lr=epsilon),
    debug=False
)

# # Define the environment
env = CatanEnv()

# Run 
train(env, agent, f"{path}{model_name}", num_episodes=num_episodes, metrics=metrics, metric_freq=metric_freq)

# Save the model
torch.save(model.state_dict(), f"{path}{model_name}\\model.pth")
metrics.save(f"{path}{model_name}\\metrics") 

# Plot the losses
metrics.plot()
metrics.save_plot(f"{path}{model_name}")
