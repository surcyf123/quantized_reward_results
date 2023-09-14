import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('prod_results.csv')

# Ensure column names are stripped of any whitespaces
df.columns = df.columns.str.strip()

# Splice the model_name to the first 15 characters
df['model_name'] = df['model_name'].str[:30]

# Metrics to extract and compute averages for
metrics_to_extract = ['bert', 'MPNet', 'RLHF', 'reciprocate', 'DPO']

# Extract raw and normalized values for the metrics
for metric in metrics_to_extract:
    df[f"{metric}_raw"], df[f"{metric}_norm"] = zip(*df[metric].str.strip('[]').str.split(', ').tolist())
    df[f"{metric}_raw"] = df[f"{metric}_raw"].astype(float)
    df[f"{metric}_norm"] = df[f"{metric}_norm"].astype(float)

# Compute the averages for each model
model_averages = df.groupby('model_name').mean()

# Save the model averages to a new CSV file
output_file = 'model_averages.csv'
model_averages.to_csv(output_file)

# Load the new CSV file with averages
avg_df = pd.read_csv(output_file)

# Visualize the averages using bar charts
def plot_averages(column_name):
    sorted_df = avg_df.sort_values(by=column_name, ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_df['model_name'], sorted_df[column_name])
    plt.xticks(rotation=90)
    plt.ylabel(column_name)
    plt.xlabel('Model Name')
    plt.title(f'Average {column_name} for each Model')
    plt.tight_layout()
    plt.show()

# List of metrics you'd like to visualize
# Edit this list to determine which metrics are visualized
metrics_to_visualize = [
    # 'bert_raw', 
    # 'MPNet_norm',
    'total_reward'
]

# Generate bar charts for each specified metric
for col in metrics_to_visualize:
    plot_averages(col)
