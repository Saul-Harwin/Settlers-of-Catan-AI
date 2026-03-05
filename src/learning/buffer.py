import torch


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_mask = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()
        self.action_mask.clear()
        
    def __str__(self):
        def slice_preview(lst):
            if not lst:
                return []
            if len(lst) <= 7:
                return lst
            return lst[:2] + ["..."] + lst[-5:]

        lengths = [
            len(self.states),
            len(self.actions),
            len(self.log_probs),
            len(self.rewards),
            len(self.dones),
            len(self.values),
            len(self.action_mask)
        ]
        consistency = "All buffer components have consistent lengths." \
            if len(set(lengths)) == 1 else "Warning: Inconsistent buffer lengths detected."

        result = [
            "RolloutBuffer Preview",
            "---------------------",
            f"States   : {slice_preview(self.states)}",
            f"Actions  : {slice_preview(self.actions)}",
            f"Log probs: {slice_preview(self.log_probs)}",
            f"Rewards  : {slice_preview(self.rewards)}",
            f"Dones    : {slice_preview(self.dones)}",
            f"Values   : {slice_preview(self.values)}",
            f"Actions Mark: {slice_preview(self.values)}",
            f"Number of transitions: {len(self.states)}",
            consistency
        ]
        
        return "\n".join(result)