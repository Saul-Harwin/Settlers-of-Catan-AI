import numpy as np

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):    # Generalised Advantage Estimator
    advantages = []
    gae = 0

    values = values + [last_value]  # bootstrap

    # print(values)

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        # gae = np.clip(gae, -1e3, 1e3) # At the moment this is the only thing that makes my code work. 
        # Suspected issue is when a game has run for a very long time then we get overflow errors. 
        # print(f"t={t}, delta={delta}, gae={gae}")
        advantages.insert(0, gae)

    returns = [a + v for a, v in zip(advantages, values[:-1])]

    # print("=== GAE Debug ===")
    # print(f"Rewards: {rewards}")
    # print(f"Values: {values[:-1]}")
    # print(f"Dones: {dones}")
    # print(f"Advantages (before clipping): {advantages}")
    # print(f"Returns: {returns}")
    # print(f"Any NaNs? {np.isnan(advantages).any() or np.isnan(returns).any()}")
    # print("=================")

    return advantages, returns