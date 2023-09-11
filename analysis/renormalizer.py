import os
import torch
import pandas as pd

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
    # Define your mean and variance sets with their corresponding names in the txt file
    mean_var_sets = {
        "rlhf": {"mean_var": (0.75, 1.69), "txt_name": "rlhf_mean_norm"},
        "reciprocate": {"mean_var": (2.91, 13.35), "txt_name": "reciprocate_reward_mean_norm"},
        "dpo": {"mean_var": (-11.78, 4.36), "txt_name": "dpo_norm_mean"}
    }
    
    # Read the corresponding CSV file
    csv_filepath = filepath.replace('.txt', '.csv')
    if not os.path.exists(csv_filepath):
        return
    
    df = pd.read_csv(csv_filepath)

    # Normalize each individual data point for the three metrics
    norm_averages = {}
    for metric, data in mean_var_sets.items():
        mean, var = data["mean_var"]
        if metric in df.columns:
            normalizer = RewardNormalizer(mean, var)
            df[f'{metric}_norm'] = df[metric].apply(normalizer.normalize_rewards)
            # Compute true normalized averages
            norm_averages[metric] = df[f'{metric}_norm'].mean()

    # Read the TXT file
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Check if the file contains only the two specified entries and skip processing if true
    if len(lines) <= 2 and any("gpu_name" in line for line in lines) and any("mean_duration" in line for line in lines):
        return

    relevance_pass_rate = 1.0  # Default to 1 if not found
    for i, line in enumerate(lines):
        if "relevance_pass_rate" in line:
            relevance_pass_rate = float(line.split()[-1])
            continue
        for metric, data in mean_var_sets.items():
            if data["txt_name"] in line:
                lines[i] = f"{data['txt_name']} {norm_averages[metric]}\n"

    # Compute total_reward and total_reward_excluding_relevance
    total_reward = (norm_averages.get("rlhf", 0) * 0.4 + norm_averages.get("reciprocate", 0) * 0.3 + norm_averages.get("dpo", 0) * 0.3) * relevance_pass_rate
    total_reward_excluding_relevance = (norm_averages.get("rlhf", 0) * 0.4 + norm_averages.get("reciprocate", 0) * 0.3 + norm_averages.get("dpo", 0) * 0.3)
    
    total_reward_str = f"total_reward {total_reward}\n"
    total_reward_excluding_relevance_str = f"total_reward_excluding_relevance {total_reward_excluding_relevance}\n"

    # Update the total_reward or total_reward_excluding_relevance lines if they are already present
    for i, line in enumerate(lines):
        if "total_reward " in line:
            lines[i] = total_reward_str
        elif "total_reward_excluding_relevance " in line:
            lines[i] = total_reward_excluding_relevance_str

    # Write back to the file
    with open(filepath, 'w') as f:
        f.writelines(lines)

# Iterate over all .txt files in the parent directory
parent_directory = "../"
for filename in os.listdir(parent_directory):
    if filename.endswith(".txt") and filename != "normalizer.py":  # Exclude the script itself
        filepath = os.path.join(parent_directory, filename)
        process_file(filepath)
