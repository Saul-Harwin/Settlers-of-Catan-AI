import matplotlib.pyplot as plt 
import pickle


class LossMetrics:  
    path = ".\\learning\\settlers_ppo_training\\"
    
    def __init__(self):
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.episodes = []
        
    def plot(self):
        plt.figure(figsize=(10,6))

        plt.plot(self.policy_losses, label="Policy Loss")
        # plt.plot(self.value_losses, label="Value Loss")
        plt.plot(self.entropy_losses, label="Entropy Loss")

        plt.xlabel("Training Updates")
        plt.ylabel("Loss")
        plt.title("PPO Loss Curves (10-Point Catan)")
        plt.legend()
        plt.grid(True)

        plt.show()
        
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
        plt.figure(figsize=(10,6))

        plt.plot(self.policy_losses, label="Policy Loss")
        # plt.plot(self.value_losses, label="Value Loss")
        plt.plot(self.entropy_losses, label="Entropy Loss")

        plt.xlabel("Training Updates")
        plt.ylabel("Loss")
        plt.title("PPO Loss Curves (10-Point Catan)")
        plt.legend()
        plt.grid(True)

        plt.savefig(f"{model_name}\\plots\\policy_loss_plot_{self.episodes[-1]}.png")  # Save plot with episode number in filename
        plt.close()  # Close the plot to free memory
