import matplotlib.pyplot as plt
import pandas as pd

df_with_model_updated = pd.read_csv('compiled_results_with_model_name.csv')
df_with_model_updated['model_name'] = df_with_model_updated['model_name'].str.replace('-GPTQ', '', regex=False)

# Columns specified by the user to analyze
specified_cols = [
    # 'mean_duration', 
    'relevance_pass_rate', 
    'dpo_norm_mean', 'rlhf_mean_norm', 'reciprocate_reward_mean_norm', 'total_reward'
]

def plot_specific_data_updated(col):
    """
    Function to plot data for a given column with consideration for "relevance_pass_rate".
    """
    plt.figure(figsize=(12, 6))
    
    # Extract benchmark value for the metric
    benchmark_value = df_with_model_updated[df_with_model_updated['model_name'] == 'benchmark'][col].values[0]
    
    # Plotting the data
    plt.bar(df_with_model_updated['model_name'][1:], df_with_model_updated[col][1:])
    
    # Plot benchmark line only if it's not NaN
    if not pd.isna(benchmark_value):
        plt.axhline(y=benchmark_value, color='r', linestyle='--', label="Benchmark")
        
    # Adjust y-axis limits for "relevance_pass_rate"
    if col == "relevance_pass_rate":
        plt.ylim(0, 1.1)
    
    plt.xticks(rotation=90)
    plt.title(f'{col} by Model')
    plt.ylabel(col)
    plt.xlabel('Model Name')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Visualize the specified metrics using the updated plotting function
for col in specified_cols:
    plot_specific_data_updated(col)


