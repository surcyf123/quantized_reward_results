import torch
import pandas as pd

class RewardNormalizer:
    def __init__(self, mean: float, var: float):
        self.mean = mean
        self.var = var
        
    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        rewards = rewards - self.mean
        if self.var > 0:
            rewards /= torch.sqrt(torch.tensor(self.var))
        rewards = 0.5 * (1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device)))
        return rewards

def normalize_csv_columns(csv_path, mean_var_sets, raw_cols, norm_cols):
    data = pd.read_csv(csv_path)
    
    for (mean, var), raw_col, norm_col in zip(mean_var_sets, raw_cols, norm_cols):
        normalizer = RewardNormalizer(mean, var)
        data[norm_col] = normalizer.normalize_rewards(torch.tensor(data[raw_col].values)).numpy()
    
    data.to_csv(csv_path, index=False)

# Define your three unique mean and variance sets
mean_var_sets = [
    (mean_1, var_1),
    (mean_2, var_2),
    (mean_3, var_3)
]

# Columns to normalize
raw_cols = ['dpo', 'rlhf', 'reciprocate']
norm_cols = ['dpo_norm', 'rlhf_norm', 'reciprocate_norm']

# Apply the normalization to your CSV
normalize_csv_columns('your_csv_path.csv', mean_var_sets, raw_cols, norm_cols)
