import matplotlib.pyplot as plt 
import pickle
import numpy as np


class LossMetrics:  
    path = ".\\learning\\settlers_ppo_training\\"
    
    def __init__(self):
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.kl_divergences = []
        self.explained_variances = []
        self.mean_rewards = []
        self.episodes = []
        
    def plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(12,8))

        # Policy + Value Loss
        axs[0,0].plot(self.policy_losses, label="Policy Loss")
        axs[0,0].plot(self.value_losses, label="Value Loss")
        axs[0,0].set_title("Policy / Value Loss")
        axs[0,0].set_xlabel("Training Updates")
        axs[0,0].set_ylabel("Loss")
        axs[0,0].legend()
        axs[0,0].grid(True)

        # Entropy
        axs[0,1].plot(self.entropy_losses, label="Entropy Loss")
        axs[0,1].set_title("Entropy")
        axs[0,1].set_xlabel("Training Updates")
        axs[0,1].set_ylabel("Entropy")
        axs[0,1].legend()
        axs[0,1].grid(True)

        # KL Divergence
        axs[1,0].plot(self.kl_divergences, label="KL Divergence")
        axs[1,0].set_title("KL Divergence")
        axs[1,0].set_xlabel("Training Updates")
        axs[1,0].set_ylabel("KL")
        axs[1,0].legend()
        axs[1,0].grid(True)

        # Mean Reward
        axs[1,1].plot(self.mean_rewards, label="Mean Episode Reward")
        axs[1,1].set_title("Mean Reward")
        axs[1,1].set_xlabel("Training Updates")
        axs[1,1].set_ylabel("Reward")
        axs[1,1].legend()
        axs[1,1].grid(True)

        plt.tight_layout()
        plt.show()
        
    def compute_explained_variance(self, values, returns):
        """
        values: predicted value estimates
        returns: empirical returns
        """
        var_returns = np.var(returns)
        if var_returns == 0:
            return np.nan
        return 1 - np.var(returns - values) / var_returns
    
    def save(self, model_name):
        with open(f"{model_name}.pkl", 'wb') as f:
            pickle.dump(self, f)
            
        print(f"Training data saved to {model_name}.pkl")
        
    def load(self, model_name):
        with open(f"{model_name}.pkl", 'rb') as f:
            loaded_metrics = pickle.load(f)
            self.policy_losses = loaded_metrics.policy_losses
            self.value_losses = loaded_metrics.value_losses
            self.entropy_losses = loaded_metrics.entropy_losses
            self.episodes = loaded_metrics.episodes
            
    def save_plot(self, model_name):
        fig, axs = plt.subplots(2,2, figsize=(12,8))

        axs[0,0].plot(self.policy_losses, label="Policy Loss")
        axs[0,0].plot(self.value_losses, label="Value Loss")
        axs[0,0].legend()
        axs[0,0].grid(True)
        axs[0,0].set_title("Policy / Value Loss")

        axs[0,1].plot(self.entropy_losses, label="Entropy")
        axs[0,1].legend()
        axs[0,1].grid(True)
        axs[0,1].set_title("Entropy")

        axs[1,0].plot(self.kl_divergences, label="KL Divergence")
        axs[1,0].legend()
        axs[1,0].grid(True)
        axs[1,0].set_title("KL Divergence")

        axs[1,1].plot(self.mean_rewards, label="Mean Reward")
        axs[1,1].legend()
        axs[1,1].grid(True)
        axs[1,1].set_title("Mean Episode Reward")

        plt.tight_layout()
        plt.savefig(f"{model_name}\\plots\\training_metrics_{self.episodes[-1]}.png")
        plt.close()
