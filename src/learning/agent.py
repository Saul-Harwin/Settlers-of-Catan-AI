import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, model, optimiser, clip_eps=0.2, debug=False):
        self.model = model
        self.optimiser = optimiser
        self.clip_eps = clip_eps
        self.debug = debug

    def select_action(self, state_tensor, action_mask):
        logits, value = self.model(state_tensor)

        mask = torch.tensor(action_mask, dtype=torch.bool)

        if not mask.any():
            raise RuntimeError("Attempting to sample from empty action mask")
        
        masked_logits = logits.clone()
        masked_logits[~mask] = -1e9

        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.detach(), value.detach()

    def update(self, buffer, advantages, returns,
           clip_eps=0.2,
           value_coef=0.5,
           entropy_coef=0.01,
           epochs=4,
           batch_size=256):

        import torch
        from torch.distributions import Categorical
        import numpy as np

        # Debug - If this triggers at the beginning of an update call, then: The previous optimizer step created NaN weights.
        for name, p in self.model.named_parameters():
            if torch.isnan(p).any():
                raise RuntimeError(f"NaN in parameter before update: {name}")

        # ---- LOSS TRACKING (for plots) ----
        policy_losses = []
        value_losses = []
        entropy_losses = []

        # ---- Prepare tensors ----

        states = torch.stack(buffer.states)
        actions = torch.tensor(buffer.actions)
        old_log_probs = torch.stack(buffer.log_probs).detach()
        old_values = torch.stack(buffer.values).detach()
        action_mask = torch.stack(buffer.action_mask)

        returns = torch.tensor(returns, dtype=torch.float32)

        # Defensive normalization (safe even if already normalized)
        advantages = advantages.detach()
        
        # Debug:
        if self.debug:
            print("Adv abs max (pre-norm):", advantages.abs().max().item())
            print("Adv std (pre-norm):", advantages.std().item())
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)

        for _ in range(epochs):

            indices = torch.randperm(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_values = old_values[batch_idx]
                batch_action_masks = action_mask[batch_idx]

                if self.debug:
                    print("Returns abs max:", batch_returns.abs().max().item())

                # ---- Forward pass ----

                logits, values = self.model(batch_states)
                
                
                # ---- DEBUG: value magnitude ----
                if self.debug:
                    print("Value abs max:", values.abs().max().item())

                if torch.isnan(logits).any():
                    raise RuntimeError("NaN produced by model forward pass")

                if torch.isnan(values).any():
                    raise RuntimeError("NaN in value head")

                # Prevent overflow in softmax
                logits = torch.clamp(logits, -20, 20)

                # Apply mask
                assert batch_action_masks.shape == logits.shape
                logits = logits.masked_fill(~batch_action_masks, -1e9)

                # ---- Critical check ----
                valid_counts = batch_action_masks.sum(dim=1)
                if (valid_counts == 0).any():
                    bad_rows = (valid_counts == 0).nonzero(as_tuple=True)[0]
                    raise RuntimeError(f"No valid actions in rows {bad_rows}")

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("NaN or Inf detected in logits")
                
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # ---- Policy Loss ----

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                if torch.isnan(ratio).any():
                    raise RuntimeError("NaN in PPO ratio")

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # ---- Value Loss (Clipped) ----

                values = values.squeeze()

                value_pred_clipped = batch_old_values + \
                    (values - batch_old_values).clamp(-clip_eps, clip_eps)

                value_loss_1 = (values - batch_returns).pow(2)
                value_loss_2 = (value_pred_clipped - batch_returns).pow(2)

                value_loss = torch.max(value_loss_1, value_loss_2).mean()

                # ---- Total Loss ----

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                # ---- Record losses for plotting ----
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())

                # ---- Backprop ----

                self.optimiser.zero_grad()
                loss.backward()

                # Debug - If you see numbers like:
                # 10 → normal
                # 100 → unstable
                # 1000+ → catastrophic
                # You have confirmed the explosion source.

                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                if self.debug:
                    print("Grad norm BEFORE clip:", total_norm)

                # Critical for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimiser.step()

        # ---- Return mean losses for plotting ----

        return (
            float(np.mean(policy_losses)),
            float(np.mean(value_losses)),
            float(np.mean(entropy_losses))
        )