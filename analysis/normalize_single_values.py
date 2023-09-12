import torch

def normalize_reward(raw_value, mean, var):
    raw_value = torch.tensor([raw_value])
    normalized = raw_value - mean
    if var > 0:
        normalized /= torch.sqrt(torch.tensor(var))
    normalized = 0.5 * (1 + torch.erf(normalized / torch.sqrt(torch.tensor([2.0]))))
    return normalized.item()

# Mean and variance values
mean_var_sets = {
    "rlhf_mean": (0.75, 1.69),
    "reciprocate_reward_mean": (2.91, 13.35),
    "dpo_mean": (-11.78, 4.36)
}

# Raw values from the dictionary
raw_values = {
    'dpo_mean': -12.129820823669434,
    'rlhf_mean': 1.420127272605896,
    'reciprocate_reward_mean': 5.359351634979248
}

# Normalize the values
normalized_values = {}
for key, (mean, var) in mean_var_sets.items():
    normalized_key = key.replace("_mean", "_norm_mean")
    normalized_values[normalized_key] = normalize_reward(raw_values[key], mean, var)

relevance_pass_rate = 1.0
# Compute total reward
total_reward = (normalized_values['rlhf_norm_mean'] * 0.4 + 
                normalized_values['reciprocate_reward_norm_mean'] * 0.3 + 
                normalized_values['dpo_norm_mean'] * 0.3) * relevance_pass_rate

normalized_values['total_reward'] = total_reward
print(normalized_values)