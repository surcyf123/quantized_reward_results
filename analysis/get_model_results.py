import os
import csv

# Path to the directory
directory_path = 'quantized_reward_results'

# Benchmark row = benchmark,,nan,100,,100,,100,-12.129820823669434,,1.420127272605896,,5.359351634979248,
# Metrics to extract
metrics = [
    'gpu_name', 'mean_duration', 'relevance_pass_rate', 'bert_raw_avg',
    'bert_pass_rate', 'mpnet_raw_avg', 'mpnet_pass_rate', 'dpo_mean',
    'dpo_norm_mean', 'rlhf_mean', 'rlhf_mean_norm', 'reciprocate_reward_mean',
    'reciprocate_reward_mean_norm'
]

benchmark_data = {
    'model_name': 'benchmark',
    'gpu_name': '',
    'mean_duration': 'nan',
    'relevance_pass_rate': '1',
    'bert_raw_avg': '',
    'bert_pass_rate': '100',
    'mpnet_raw_avg': '',
    'mpnet_pass_rate': '100',
    'dpo_mean': '-12.129820823669434',
    'dpo_norm_mean': '',
    'rlhf_mean': '1.420127272605896',
    'rlhf_mean_norm': '',
    'reciprocate_reward_mean': '5.359351634979248',
    'reciprocate_reward_mean_norm': ''
}
# List to hold the data
data = []

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory_path, filename)
        
        # Extract model name from filename
        model_name = filename.split('-', 1)[1].rsplit('.', 1)[0]
        
        # Dictionary to hold the metrics for the current file
        file_data = {'model_name': model_name}
        
        # Open and read the content of the file
        with open(filepath, 'r') as file:
            for line in file:
                # Split the line into key and value
                key, value = line.strip().split(' ')
                
                # Only add the metric if it's in our list of metrics to extract
                if key in metrics:
                    file_data[key] = value
        
        # Add the file data to our main data list
        data.append(file_data)

# Update metrics list to include the new 'model_name' column at the beginning
metrics = ['model_name'] + metrics

data = [benchmark_data] + data

# Write the data to a CSV file
output_file = 'compiled_results_with_model_name.csv'
with open(output_file, 'w') as file:
    writer = csv.DictWriter(file, fieldnames=metrics)
    writer.writeheader()
    for row in data:
        writer.writerow(row)


