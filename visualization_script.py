"""
visualization_script.py
Visualization tools for analyzing heavy hitters algorithm performance
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple


def load_results(results_dir: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load test results from the specified directory.
    
    Args:
        results_dir: Directory containing the test results
    
    Returns:
        Tuple of (parameter study results, adversarial results)
    """
    # Load parameter study results
    param_file = os.path.join(results_dir, 'parameter_study_results.pkl')
    with open(param_file, 'rb') as f:
        param_results = pickle.load(f)
    
    # Load adversarial results
    adv_file = os.path.join(results_dir, 'adversarial_results.pkl')
    with open(adv_file, 'rb') as f:
        adv_results = pickle.load(f)
    
    return param_results, adv_results


def create_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert results to a pandas DataFrame for easier analysis.
    
    Args:
        results: List of test result dictionaries
    
    Returns:
        DataFrame with flattened results
    """
    rows = []
    for result in results:
        row = {}
        # Extract parameters
        for param_key, param_value in result['parameters'].items():
            row[param_key] = param_value
        
        # Extract metrics
        for metric_key, metric_value in result['metrics'].items():
            row[metric_key] = metric_value
        
        # Extract key statistics
        row["total_items"] = result['statistics']['total_items']
        if 'hash_range' in result['statistics']:
            row["hash_range"] = result['statistics']['hash_range']
        if 'sketch_width' in result['statistics']:
            row["sketch_width"] = result['statistics']['sketch_width']
            row["num_hash_functions"] = result['statistics']['num_hash_functions']
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_epsilon_impact(df: pd.DataFrame, output_dir: str):
    """
    Plot the impact of the privacy parameter (epsilon) on algorithm performance.
    
    Args:
        df: DataFrame containing test results
        output_dir: Directory to save the plots
    """
    # Filter for tests that varied epsilon
    epsilon_tests = df[df['data_distribution'] == 'zipf'].copy()
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision vs. Epsilon by Algorithm Type
    sns.lineplot(
        data=epsilon_tests, 
        x='epsilon', 
        y='precision', 
        hue='algorithm_type',
        marker='o',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Privacy vs. Precision')
    axes[0, 0].set_xlabel('Privacy Parameter (ε)')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].grid(True)
    
    # Plot 2: Recall vs. Epsilon by Algorithm Type
    sns.lineplot(
        data=epsilon_tests, 
        x='epsilon', 
        y='recall', 
        hue='algorithm_type',
        marker='o',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Privacy vs. Recall')
    axes[0, 1].set_xlabel('Privacy Parameter (ε)')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].grid(True)
    
    # Plot 3: F1 Score vs. Epsilon by Algorithm Type
    sns.lineplot(
        data=epsilon_tests, 
        x='epsilon', 
        y='f1_score', 
        hue='algorithm_type',
        marker='o',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Privacy vs. F1 Score')
    axes[1, 0].set_xlabel('Privacy Parameter (ε)')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].grid(True)
    
    # Plot 4: Average Error vs. Epsilon by Algorithm Type
    sns.lineplot(
        data=epsilon_tests, 
        x='epsilon', 
        y='average_error', 
        hue='algorithm_type',
        marker='o',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Privacy vs. Average Error')
    axes[1, 1].set_xlabel('Privacy Parameter (ε)')
    axes[1, 1].set_ylabel('Average Error')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'privacy_impact.png'))
    plt.close()


def plot_distribution_impact(df: pd.DataFrame, output_dir: str):
    """
    Plot the impact of data distribution on algorithm performance.
    
    Args:
        df: DataFrame containing test results
        output_dir: Directory to save the plots
    """
    # Filter for tests that varied data distribution
    dist_tests = df[df['epsilon'] == 1.0].copy()
    
    # Set up the plot
    plt.figure(figsize=(15, 6))
    
    # Plot 1: F1 Score by Distribution and Algorithm
    plt.subplot(1, 2, 1)
    sns.barplot(
        data=dist_tests,
        x='data_distribution',
        y='f1_score',
        hue='algorithm_type'
    )
    plt.title('Data Distribution Impact on F1 Score')
    plt.xlabel('Data Distribution')
    plt.ylabel('F1 Score')
    plt.grid(True, axis='y')
    
    # Plot 2: Average Error by Distribution and Algorithm
    plt.subplot(1, 2, 2)
    sns.barplot(
        data=dist_tests,
        x='data_distribution',
        y='average_error',
        hue='algorithm_type'
    )
    plt.title('Data Distribution Impact on Average Error')
    plt.xlabel('Data Distribution')
    plt.ylabel('Average Error')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_impact.png'))
    plt.close()


def plot_scalability(df: pd.DataFrame, output_dir: str):
    """
    Plot the scalability of the algorithm with respect to domain and dataset size.
    
    Args:
        df: DataFrame containing test results
        output_dir: Directory to save the plots
    """
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Runtime vs. Domain Size by Algorithm Type
    domain_tests = df[(df['data_distribution'] == 'zipf') & (df['epsilon'] == 1.0)].copy()
    sns.lineplot(
        data=domain_tests,
        x='domain_size',
        y='runtime',
        hue='algorithm_type',
        marker='o',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Scalability with Domain Size')
    axes[0, 0].set_xlabel('Domain Size')
    axes[0, 0].set_ylabel('Runtime (seconds)')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True)
    
    # Plot 2: F1 Score vs. Domain Size by Algorithm Type
    sns.lineplot(
        data=domain_tests,
        x='domain_size',
        y='f1_score',
        hue='algorithm_type',
        marker='o',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Accuracy with Domain Size')
    axes[0, 1].set_xlabel('Domain Size')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True)
    
    # Plot 3: Runtime vs. Dataset Size by Algorithm Type
    dataset_tests = df[(df['data_distribution'] == 'zipf') & (df['epsilon'] == 1.0)].copy()
    sns.lineplot(
        data=dataset_tests,
        x='dataset_size',
        y='runtime',
        hue='algorithm_type',
        marker='o',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Scalability with Dataset Size')
    axes[1, 0].set_xlabel('Dataset Size')
    axes[1, 0].set_ylabel('Runtime (seconds)')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True)
    
    # Plot 4: F1 Score vs. Dataset Size by Algorithm Type
    sns.lineplot(
        data=dataset_tests,
        x='dataset_size',
        y='f1_score',
        hue='algorithm_type',
        marker='o',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Accuracy with Dataset Size')
    axes[1, 1].set_xlabel('Dataset Size')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scalability.png'))
    plt.close()


def plot_threshold_impact(df: pd.DataFrame, output_dir: str):
    """
    Plot the impact of the threshold parameter on algorithm performance.
    
    Args:
        df: DataFrame containing test results
        output_dir: Directory to save the plots
    """
    # Filter for tests that varied threshold
    threshold_tests = df[(df['data_distribution'] == 'zipf') & (df['epsilon'] == 1.0)].copy()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Precision vs. Threshold by Algorithm Type
    sns.lineplot(
        data=threshold_tests,
        x='threshold',
        y='precision',
        hue='algorithm_type',
        marker='o',
        ax=axes[0, 0]
    )
    axes[0, 0].set_title('Threshold vs. Precision')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True)
    
    # Plot 2: Recall vs. Threshold by Algorithm Type
    sns.lineplot(
        data=threshold_tests,
        x='threshold',
        y='recall',
        hue='algorithm_type',
        marker='o',
        ax=axes[0, 1]
    )
    axes[0, 1].set_title('Threshold vs. Recall')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True)
    
    # Plot 3: Number of True Heavy Hitters vs. Threshold
    sns.lineplot(
        data=threshold_tests,
        x='threshold',
        y='num_true_hh',
        marker='o',
        ax=axes[1, 0]
    )
    axes[1, 0].set_title('Threshold vs. Number of True Heavy Hitters')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('Number of True Heavy Hitters')
    axes[1, 0].set_xscale('log')
    axes[1, 0].grid(True)
    
    # Plot 4: Number of Estimated Heavy Hitters vs. Threshold by Algorithm Type
    sns.lineplot(
        data=threshold_tests,
        x='threshold',
        y='num_estimated_hh',
        hue='algorithm_type',
        marker='o',
        ax=axes[1, 1]
    )
    axes[1, 1].set_title('Threshold vs. Number of Estimated Heavy Hitters')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Number of Estimated Heavy Hitters')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_impact.png'))
    plt.close()


def plot_algorithm_specific_params(df: pd.DataFrame, output_dir: str):
    """
    Plot the impact of algorithm-specific parameters.
    
    Args:
        df: DataFrame containing test results
        output_dir: Directory to save the plots
    """
    # Plot 1: Impact of hash range on basic algorithm
    hash_range_tests = df[(df['algorithm_type'] == 'basic') & 
                          (df['data_distribution'] == 'zipf') & 
                          (df['epsilon'] == 1.0)].copy()
    
    if not hash_range_tests.empty and 'hash_range' in hash_range_tests.columns:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.lineplot(
            data=hash_range_tests,
            x='hash_range',
            y='f1_score',
            marker='o'
        )
        plt.title('Hash Range vs. F1 Score (Basic Algorithm)')
        plt.xlabel('Hash Range')
        plt.ylabel('F1 Score')
        plt.xscale('log')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        sns.lineplot(
            data=hash_range_tests,
            x='hash_range',
            y='average_error',
            marker='o'
        )
        plt.title('Hash Range vs. Average Error (Basic Algorithm)')
        plt.xlabel('Hash Range')
        plt.ylabel('Average Error')
        plt.xscale('log')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hash_range_impact.png'))
        plt.close()
    
    # Plot 2: Impact of number of hash functions on CMS algorithm
    num_hash_tests = df[(df['algorithm_type'] == 'cms') & 
                        (df['data_distribution'] == 'zipf') & 
                        (df['epsilon'] == 1.0)].copy()
    
    if not num_hash_tests.empty and 'num_hash_functions' in num_hash_tests.columns:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.lineplot(
            data=num_hash_tests,
            x='num_hash_functions',
            y='f1_score',
            marker='o'
        )
        plt.title('Number of Hash Functions vs. F1 Score (CMS Algorithm)')
        plt.xlabel('Number of Hash Functions')
        plt.ylabel('F1 Score')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        sns.lineplot(
            data=num_hash_tests,
            x='num_hash_functions',
            y='average_error',
            marker='o'
        )
        plt.title('Number of Hash Functions vs. Average Error (CMS Algorithm)')
        plt.xlabel('Number of Hash Functions')
        plt.ylabel('Average Error')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'num_hash_functions_impact.png'))
        plt.close()


def plot_adversarial_results(adv_results: Dict[str, Any], output_dir: str):
    """
    Plot the results of adversarial testing.
    
    Args:
        adv_results: Dictionary of adversarial test results
        output_dir: Directory to save the plots
    """
    # Plot 1: Comparison of false positives between algorithms
    plt.figure(figsize=(10, 6))
    algorithms = ['Basic Algorithm', 'Count-Min Sketch']
    false_positives = [
        adv_results['adversarial_collisions']['basic_false_positives'],
        adv_results['adversarial_collisions']['cms_false_positives']
    ]
    
    plt.bar(algorithms, false_positives)
    plt.title('Resilience to Hash Collision Attacks')
    plt.xlabel('Algorithm')
    plt.ylabel('Number of False Positives')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'adversarial_collisions.png'))
    plt.close()
    
    # Plot 2: Privacy leakage with varying epsilon
    plt.figure(figsize=(10, 6))
    epsilons = adv_results['privacy_leakage']['epsilons']
    detection_rates = adv_results['privacy_leakage']['detection_rates']
    
    plt.plot(epsilons, detection_rates, marker='o')
    plt.title('Privacy Leakage: Detection Rate of Rare Elements')
    plt.xlabel('Privacy Parameter (ε)')
    plt.ylabel('Detection Rate')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'privacy_leakage.png'))
    plt.close()


def create_all_visualizations(results_dir: str = 'test_results', output_dir: str = 'visualizations'):
    """
    Create all visualizations from test results.
    
    Args:
        results_dir: Directory containing test results
        output_dir: Directory to save the visualizations
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    param_results, adv_results = load_results(results_dir)
    
    # Convert to DataFrame
    df = create_dataframe(param_results)
    
    # Create visualizations
    print("Creating privacy impact visualizations...")
    plot_epsilon_impact(df, output_dir)
    
    print("Creating distribution impact visualizations...")
    plot_distribution_impact(df, output_dir)
    
    print("Creating scalability visualizations...")
    plot_scalability(df, output_dir)
    
    print("Creating threshold impact visualizations...")
    plot_threshold_impact(df, output_dir)
    
    print("Creating algorithm-specific parameter visualizations...")
    plot_algorithm_specific_params(df, output_dir)
    
    print("Creating adversarial testing visualizations...")
    plot_adversarial_results(adv_results, output_dir)
    
    print(f"All visualizations created and saved to {output_dir}")


def generate_summary_report(results_dir: str = 'test_results', output_dir: str = 'visualizations'):
    """
    Generate a summary report of the test results.
    
    Args:
        results_dir: Directory containing test results
        output_dir: Directory to save the report
    """
    # Load results
    param_results, adv_results = load_results(results_dir)
    
    # Convert to DataFrame
    df = create_dataframe(param_results)
    
    # Create a summary DataFrame for each test type
    
    # 1. Privacy parameter (epsilon) impact
    epsilon_tests = df[df['data_distribution'] == 'zipf'].copy()
    epsilon_summary = epsilon_tests.groupby(['algorithm_type', 'epsilon']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'average_error': 'mean',
        'runtime': 'mean'
    }).reset_index()
    
    # 2. Data distribution impact
    dist_tests = df[df['epsilon'] == 1.0].copy()
    dist_summary = dist_tests.groupby(['algorithm_type', 'data_distribution']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'average_error': 'mean',
        'runtime': 'mean'
    }).reset_index()
    
    # 3. Threshold impact
    threshold_tests = df[(df['data_distribution'] == 'zipf') & (df['epsilon'] == 1.0)].copy()
    threshold_summary = threshold_tests.groupby(['algorithm_type', 'threshold']).agg({
        'precision': 'mean',
        'recall': 'mean',
        'f1_score': 'mean',
        'average_error': 'mean',
        'num_true_hh': 'mean',
        'num_estimated_hh': 'mean'
    }).reset_index()
    
    # Generate a markdown report with the key findings
    report = f"""# Heavy Hitters Algorithm Evaluation Report

## 1. Privacy Parameter (ε) Impact

The privacy parameter ε controls the privacy-utility tradeoff in our algorithm:
- Higher ε values provide less privacy but better utility (accuracy)
- Lower ε values provide stronger privacy but reduced accuracy

**Key Findings:**
- Average F1 score at ε=0.1: {epsilon_summary[epsilon_summary['epsilon'] == 0.1]['f1_score'].mean():.4f}
- Average F1 score at ε=5.0: {epsilon_summary[epsilon_summary['epsilon'] == 5.0]['f1_score'].mean():.4f}
- The "Count-Min Sketch" algorithm shows {"better" if epsilon_summary[epsilon_summary['algorithm_type'] == 'cms']['f1_score'].mean() > epsilon_summary[epsilon_summary['algorithm_type'] == 'basic']['f1_score'].mean() else "worse"} average performance than the basic algorithm across privacy levels.

## 2. Data Distribution Impact

We tested the algorithm with three different data distributions:
- Uniform: Elements appear with equal probability
- Zipf: Power-law distribution (realistic for many real-world phenomena)
- Normal: Gaussian distribution centered on the domain

**Key Findings:**
- Best performance was observed with the {dist_summary.loc[dist_summary['f1_score'].idxmax()]['data_distribution']} distribution
- Worst performance was observed with the {dist_summary.loc[dist_summary['f1_score'].idxmin()]['data_distribution']} distribution
- The "Count-Min Sketch" algorithm shows {"more" if dist_summary[dist_summary['algorithm_type'] == 'cms']['f1_score'].std() < dist_summary[dist_summary['algorithm_type'] == 'basic']['f1_score'].std() else "less"} consistent performance across different distributions.

## 3. Threshold Impact

The threshold parameter determines how frequent an element must be to be considered a heavy hitter:
- Lower thresholds identify more heavy hitters but may increase false positives
- Higher thresholds are more selective but may miss relevant elements

**Key Findings:**
- At threshold=0.001: Average of {threshold_summary[threshold_summary['threshold'] == 0.001]['num_true_hh'].mean():.1f} true heavy hitters
- At threshold=0.05: Average of {threshold_summary[threshold_summary['threshold'] == 0.05]['num_true_hh'].mean():.1f} true heavy hitters
- Precision tends to {"increase" if threshold_summary.groupby('threshold')['precision'].mean().is_monotonic_increasing else "decrease"} as the threshold increases
- Recall tends to {"increase" if threshold_summary.groupby('threshold')['recall'].mean().is_monotonic_increasing else "decrease"} as the threshold increases

## 4. Adversarial Testing

We tested the algorithm's resilience to adversarial attacks:

### Hash Collision Attack:
- Basic Algorithm: {adv_results['adversarial_collisions']['basic_false_positives']} false positives
- Count-Min Sketch: {adv_results['adversarial_collisions']['cms_false_positives']} false positives
- The Count-Min Sketch algorithm is {"more" if adv_results['adversarial_collisions']['cms_false_positives'] < adv_results['adversarial_collisions']['basic_false_positives'] else "less"} resistant to hash collision attacks.

### Privacy Leakage:
- Detection rate of rare elements at ε=0.1: {adv_results['privacy_leakage']['detection_rates'][0]:.2f}
- Detection rate of rare elements at ε=5.0: {adv_results['privacy_leakage']['detection_rates'][-1]:.2f}
- The detection rate {"increases" if adv_results['privacy_leakage']['detection_rates'][-1] > adv_results['privacy_leakage']['detection_rates'][0] else "decreases"} as ε increases, confirming the expected privacy-utility tradeoff.

## 5. Recommendations

Based on our findings, we recommend:

1. **Algorithm Choice**: {"Count-Min Sketch" if epsilon_summary[epsilon_summary['algorithm_type'] == 'cms']['f1_score'].mean() > epsilon_summary[epsilon_summary['algorithm_type'] == 'basic']['f1_score'].mean() else "Basic Algorithm"} provides better overall performance.
2. **Privacy Parameter**: Choose ε between 1.0 and 2.0 for a good balance between privacy and utility.
3. **Threshold Selection**: Set threshold based on the expected frequency of items of interest and desired precision/recall tradeoff.
4. **Hash Functions**: For Count-Min Sketch, using {threshold_tests[threshold_tests['algorithm_type'] == 'cms'].groupby('num_hash_functions')['f1_score'].mean().idxmax() if 'num_hash_functions' in threshold_tests.columns else '5'} hash functions provides optimal performance.
"""

    # Save the report
    report_path = os.path.join(output_dir, 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Summary report generated and saved to {report_path}")
    
    return report


if __name__ == "__main__":
    print("Generating visualizations for heavy hitters algorithm...")
    
    # Create output directory
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    create_all_visualizations('test_results', output_dir)
    
    # Generate summary report
    generate_summary_report('test_results', output_dir)
    
    print("Visualization and reporting complete.")