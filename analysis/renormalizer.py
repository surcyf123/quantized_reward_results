import os
import torch

class RewardNormalizer:
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var
        
    def normalize_rewards(self, reward: float) -> float:
        reward = torch.tensor([reward])
        reward = reward - self.mean
        if self.var > 0:
            reward /= torch.sqrt(torch.tensor(self.var))
        reward = 0.5 * (1 + torch.erf(reward / torch.sqrt(torch.tensor([2.0]))))
        return reward.item()

def process_file(filepath):
    # Define your mean and variance sets
    mean_var_sets = {
        "rlhf_mean": (0.75, 1.69),
        "reciprocate_reward_mean": (2.91, 13.35),
        "dpo_mean": (-11.78, 4.36)
    }

def process_file(filepath):
    # Read the file

    mean_var_sets = {
        "rlhf_mean": (0.75, 1.69),
        "reciprocate_reward_mean": (2.91, 13.35),
        "dpo_mean": (-11.78, 4.36)
    }
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Check if the file contains only the two specified entries and skip processing if true
    if len(lines) <= 2 and any("gpu_name" in line for line in lines) and any("mean_duration" in line for line in lines):
        return

    # Ensure that the rest of the code in the function is indented correctly
    for i, line in enumerate(lines):

        norm_values = {}
        relevance_pass_rate = 1.0  # Default to 1 if not found
        for i, line in enumerate(lines):
            if "relevance_pass_rate" in line:
                relevance_pass_rate = float(line.split()[-1])
            for raw_str, (mean, var) in mean_var_sets.items():
                if raw_str in line:
                    raw_value = float(line.split()[-1])
                    normalizer = RewardNormalizer(mean, var)
                    norm_value = normalizer.normalize_rewards(raw_value)
                    norm_values[raw_str] = norm_value
                    # Check if there is a next line and "_norm" is in it
                    if i < len(lines) - 1 and "_norm" in lines[i+1]:
                        lines[i+1] = lines[i+1].split()[0] + " " + str(norm_value) + "\n"


    # Compute total_reward
    total_reward = (norm_values["rlhf_mean"] * 0.4 + norm_values["reciprocate_reward_mean"] * 0.3 + norm_values["dpo_mean"] * 0.3) * relevance_pass_rate
    total_reward_str = f"total_reward {total_reward}\n"

    # Insert total_reward into the file
    lines.append(total_reward_str)

    # Write back to the file
    with open(filepath, 'w') as f:
        f.writelines(lines)

# Iterate over all .txt files in the parent directory
parent_directory = "."
for filename in os.listdir(parent_directory):
    if filename.endswith(".txt") and filename != "normalizer.py":  # Exclude the script itself
        filepath = os.path.join(parent_directory, filename)
        process_file(filepath)
