"""
heavy_hitters_testing.py
Testing framework for comparing heavy hitters algorithms
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import algorithms from your implementation files
from heavy_hitters_implementation import (
    LocalModelHeavyHitters, 
    OLHHeavyHitters,
    RAPPORHeavyHitters
)
from bit_by_bit_heavyhitters import BitByBitHeavyHitters

# Import data generators
from data_generators import (
    generate_zipf_data,
    generate_uniform_data,
    generate_normal_data
)

# Helper functions for testing

def calculate_true_frequencies(data: List[int]) -> Dict[int, float]:
    """
    Calculate true frequencies of elements in the data.
    
    Args:
        data: List of elements
        
    Returns:
        Dictionary mapping elements to their frequencies
    """
    counts = {}
    for element in data:
        counts[element] = counts.get(element, 0) + 1
    
    total = len(data)
    frequencies = {element: count/total for element, count in counts.items()}
    
    return frequencies

def get_true_heavy_hitters(data: List[int], threshold: float) -> Dict[int, float]:
    """
    Get elements that exceed the threshold (true heavy hitters).
    
    Args:
        data: List of elements
        threshold: Minimum frequency to be considered a heavy hitter
        
    Returns:
        Dictionary mapping heavy hitter elements to their frequencies
    """
    freqs = calculate_true_frequencies(data)
    return {elem: freq for elem, freq in freqs.items() if freq > threshold}

def calculate_metrics(true_hh: Dict[int, float], 
                     estimated_hh: Dict[int, float], 
                     all_elements: List[int]) -> Dict[str, float]:
    """
    Calculate performance metrics for the algorithm.
    
    Args:
        true_hh: Dictionary of true heavy hitters
        estimated_hh: Dictionary of estimated heavy hitters
        all_elements: Complete list of elements
        
    Returns:
        Dictionary of metrics (precision, recall, f1_score, avg_error)
    """
    true_set = set(true_hh.keys())
    est_set = set(estimated_hh.keys())
    
    # Handle edge cases
    if not true_set and not est_set:
        precision = 1.0
        recall = 1.0
    elif not est_set:
        precision = 0.0
        recall = 0.0
    elif not true_set:
        precision = 0.0
        recall = 1.0
    else:
        precision = len(true_set.intersection(est_set)) / len(est_set)
        recall = len(true_set.intersection(est_set)) / len(true_set)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average frequency error on all elements
    true_freqs = calculate_true_frequencies(all_elements)
    
    # Candidates for error calculation (true or estimated heavy hitters)
    candidates = true_set.union(est_set)
    error_sum = 0
    
    for element in candidates:
        true_freq = true_freqs.get(element, 0)
        est_freq = estimated_hh.get(element, 0)
        error_sum += abs(true_freq - est_freq)
    
    avg_error = error_sum / len(candidates) if candidates else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_error': avg_error,
        'num_true_hh': len(true_set),
        'num_estimated_hh': len(est_set)
    }

def generate_highly_skewed_data(domain_size: int, sample_size: int) -> List[int]:
    """
    Create a highly skewed distribution with a few dominant elements.
    
    Args:
        domain_size: Number of possible unique elements
        sample_size: Total number of elements to generate
        
    Returns:
        List of elements with a skewed distribution
    """
    # 3 heavy hitters with frequencies 15%, 10%, and 5%
    data = []
    data.extend([0] * int(sample_size * 0.15))
    data.extend([1] * int(sample_size * 0.10))
    data.extend([2] * int(sample_size * 0.05))
    
    # Rest follows a Zipf with alpha=2.0
    remaining_samples = sample_size - len(data)
    probs = np.array([1/((i+3)**2.0) for i in range(3, domain_size)])
    probs /= probs.sum()
    data.extend(np.random.choice(range(3, domain_size), size=remaining_samples, p=probs))
    
    # Shuffle the data
    np.random.shuffle(data)
    return data

def run_algorithm_comparison(test_params: Dict[str, Any], algorithms: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Run algorithm comparison with specified parameters.
    
    Args:
        test_params: Dictionary of test parameters
        algorithms: List of algorithms to test
        
    Returns:
        Dictionary of results indexed by algorithm
    """
    if algorithms is None:
        algorithms = ['RR', 'OLH']
        
    results = {}
    
    # Generate data based on distribution
    if test_params['distribution'] == 'zipf':
        data = generate_zipf_data(
            domain_size=test_params['domain_size'],
            num_elements=test_params['sample_size'],
            alpha=test_params.get('alpha', 1.5)
        )
    elif test_params['distribution'] == 'uniform':
        data = generate_uniform_data(
            domain_size=test_params['domain_size'],
            num_elements=test_params['sample_size']
        )
    elif test_params['distribution'] == 'normal':
        data = generate_normal_data(
            domain_size=test_params['domain_size'],
            num_elements=test_params['sample_size']
        )
    elif test_params['distribution'] == 'custom_skewed':
        data = generate_highly_skewed_data(
            domain_size=test_params['domain_size'],
            sample_size=test_params['sample_size']
        )
    else:
        raise ValueError(f"Unknown distribution: {test_params['distribution']}")
    
    # Get true heavy hitters for comparison
    true_freqs = calculate_true_frequencies(data)
    true_heavy_hitters = {elem: freq for elem, freq in true_freqs.items() 
                          if freq > test_params['threshold']}
    
    print(f"Generated dataset with {len(data)} elements, {len(true_heavy_hitters)} true heavy hitters")
    print(f"Top 5 true heavy hitters: {sorted(true_heavy_hitters.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    for algo_type in algorithms:
        print(f"Running {algo_type}...")
        start_time = time.time()
        
        # Initialize algorithm
        if algo_type == "RR":
            algorithm = LocalModelHeavyHitters(
                domain_size=test_params['domain_size'],
                epsilon=test_params['epsilon'],
                threshold=test_params['threshold']
            )
        elif algo_type == "OLH":
            # Remove hash_range parameter since it's not supported in your implementation
            algorithm = OLHHeavyHitters(
                domain_size=test_params['domain_size'],
                epsilon=test_params['epsilon'],
                threshold=test_params['threshold']
            )
        elif algo_type == "RAPPOR":
            algorithm = RAPPORHeavyHitters(
                domain_size=test_params['domain_size'],
                epsilon=test_params['epsilon'],
                threshold=test_params['threshold'],
                num_hash_functions=5
            )
        elif algo_type == "Bit-by-Bit":
            algorithm = BitByBitHeavyHitters(
                domain_size=test_params['domain_size'],
                epsilon=test_params['epsilon'],
                threshold=test_params['threshold']
            )
        else:
            raise ValueError(f"Unknown algorithm type: {algo_type}")
        
        # Add data
        algorithm.add_elements(data)
        
        # Identify estimated heavy hitters
        candidate_set = list(range(test_params['domain_size']))
        estimated_heavy_hitters = algorithm.identify_heavy_hitters(candidate_set)
        
        # Calculate metrics
        metrics = calculate_metrics(true_heavy_hitters, estimated_heavy_hitters, data)
        
        # Get algorithm statistics
        stats = algorithm.get_statistics()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        results[algo_type] = {
            'metrics': metrics,
            'stats': stats,
            'runtime': runtime,
            'estimated_heavy_hitters': estimated_heavy_hitters,
            'true_heavy_hitters': true_heavy_hitters
        }
        
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Average Error: {metrics['avg_error']:.4f}")
        print(f"  Runtime: {runtime:.2f} seconds")
        
    return results

def generate_summary_table(test_runs: Dict[str, Dict[str, Dict[str, Any]]], metric: str = 'f1_score') -> pd.DataFrame:
    """
    Generate a summary table from test runs.
    
    Args:
        test_runs: Dictionary of test results by test name
        metric: Metric to display (f1_score, precision, recall, avg_error)
        
    Returns:
        Pandas DataFrame with summary table
    """
    rows = []
    for test_name, test_results in test_runs.items():
        row = {'Test': test_name}
        for algo, results in test_results.items():
            row[algo] = results['metrics'][metric]
        rows.append(row)
    
    return pd.DataFrame(rows)

def plot_metric_comparison(test_results: Dict[str, Dict[str, Any]], 
                           metric: str = 'f1_score',
                           title: str = None) -> go.Figure:
    """
    Create a bar chart comparing algorithms on a specific metric.
    
    Args:
        test_results: Dictionary of test results by algorithm
        metric: Metric to compare (f1_score, precision, recall, avg_error)
        title: Custom title for the plot
        
    Returns:
        Plotly figure object
    """
    algorithms = list(test_results.keys())
    values = [test_results[algo]['metrics'][metric] for algo in algorithms]
    
    if title is None:
        title = f"{metric.replace('_', ' ').title()} Comparison"
    
    fig = go.Figure(data=[
        go.Bar(
            x=algorithms,
            y=values,
            text=[f"{v:.4f}" for v in values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Algorithm",
        yaxis_title=metric.replace('_', ' ').title(),
        yaxis=dict(range=[0, 1.0]) if metric != 'avg_error' else None
    )
    
    return fig

def run_test_suite() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Run a comprehensive test suite with a variety of parameters.
    
    Returns:
        Dictionary of test results indexed by test name
    """
    all_results = {}
    
    # Test 1: Basic Algorithm Validation
    test_name = "Basic Algorithm Validation"
    print(f"\nRunning {test_name}...")
    basic_params = {
        "epsilon": 2.0,
        "domain_size": 100,
        "sample_size": 10000,
        "distribution": "zipf",
        "alpha": 1.5,  # For Zipf distribution
        "threshold": 0.02
    }
    all_results[test_name] = run_algorithm_comparison(basic_params)
    
    # Test 2: Privacy-Utility Tradeoff
    privacy_tests = [
        {"epsilon": 0.5, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 1.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 4.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01}
    ]
    
    for i, params in enumerate(privacy_tests):
        epsilon = params["epsilon"]
        test_name = f"Privacy Test (ε={epsilon})"
        print(f"\nRunning {test_name}...")
        all_results[test_name] = run_algorithm_comparison(params)
    
    # Test 3: Distribution Effects
    distribution_tests = [
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "uniform", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "normal", "threshold": 0.01}
    ]
    
    for i, params in enumerate(distribution_tests):
        dist = params["distribution"]
        test_name = f"Distribution Test ({dist})"
        print(f"\nRunning {test_name}...")
        all_results[test_name] = run_algorithm_comparison(params)
    
    # Test 4: Sample Size Sensitivity
    sample_size_tests = [
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 5000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 10000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 200000, "distribution": "zipf", "threshold": 0.01}
    ]
    
    for i, params in enumerate(sample_size_tests):
        size = params["sample_size"]
        test_name = f"Sample Size Test (n={size})"
        print(f"\nRunning {test_name}...")
        all_results[test_name] = run_algorithm_comparison(params)
    
    # Test 5: Domain Size Scaling
    domain_size_tests = [
        {"epsilon": 2.0, "domain_size": 100, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 10000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01}
    ]
    
    for i, params in enumerate(domain_size_tests):
        domain = params["domain_size"]
        test_name = f"Domain Size Test (d={domain})"
        print(f"\nRunning {test_name}...")
        all_results[test_name] = run_algorithm_comparison(params)
    
    # Test 6: Threshold Sensitivity
    threshold_tests = [
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.005},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.01},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.02},
        {"epsilon": 2.0, "domain_size": 1000, "sample_size": 50000, "distribution": "zipf", "threshold": 0.05}
    ]
    
    for i, params in enumerate(threshold_tests):
        threshold = params["threshold"]
        test_name = f"Threshold Test (t={threshold})"
        print(f"\nRunning {test_name}...")
        all_results[test_name] = run_algorithm_comparison(params)
    
    # Remove OLH Hash Range test since your implementation doesn't support it
    
    # Test 8: Special Case - Skewed Distribution
    test_name = "Highly Skewed Distribution"
    print(f"\nRunning {test_name}...")
    skewed_test = {
        "epsilon": 2.0, 
        "domain_size": 1000, 
        "sample_size": 50000, 
        "distribution": "custom_skewed",
        "threshold": 0.03  # Threshold between the top heavy hitters
    }
    all_results[test_name] = run_algorithm_comparison(skewed_test)
    
    return all_results

# Main execution function
def main():
    """
    Main function to execute tests and generate summary results.
    """
    print("Starting Heavy Hitters Algorithm Test Suite")
    print("=" * 80)
    
    # Run tests
    test_results = run_test_suite()
    
    # Generate summary tables
    print("\nGenerating summary tables...")
    
    f1_table = generate_summary_table(test_results, 'f1_score')
    precision_table = generate_summary_table(test_results, 'precision')
    recall_table = generate_summary_table(test_results, 'recall')
    error_table = generate_summary_table(test_results, 'avg_error')
    
    print("\nF1 Score Summary:")
    print(f1_table)
    
    print("\nPrecision Summary:")
    print(precision_table)
    
    print("\nRecall Summary:")
    print(recall_table)
    
    print("\nAverage Error Summary:")
    print(error_table)
    
    # Save results to CSV
    print("\nSaving results to CSV files...")
    f1_table.to_csv("hh_f1_scores.csv", index=False)
    precision_table.to_csv("hh_precision.csv", index=False)
    recall_table.to_csv("hh_recall.csv", index=False)
    error_table.to_csv("hh_avg_error.csv", index=False)
    
    print("\nTest suite completed!")

if __name__ == "__main__":
    main()