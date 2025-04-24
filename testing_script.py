"""
testing_script.py
Comprehensive testing framework for the heavy hitters algorithm
"""

import numpy as np
import pandas as pd
import time
import os
import pickle
from typing import Dict, List, Any, Tuple
from heavy_hitters_implementation import LocalModelHeavyHitters, CountMinSketchHeavyHitters

def generate_uniform_data(domain_size: int, num_elements: int) -> List[int]:
    """
    Generate uniformly distributed data.
    
    Args:
        domain_size: Number of unique elements in the domain
        num_elements: Total number of elements to generate
    
    Returns:
        List of generated elements
    """
    return np.random.randint(0, domain_size, size=num_elements).tolist()

def generate_zipf_data(domain_size: int, num_elements: int, alpha: float = 1.07) -> List[int]:
    """
    Generate data following a Zipf distribution (power law).
    
    Args:
        domain_size: Number of unique elements in the domain
        num_elements: Total number of elements to generate
        alpha: Parameter of the Zipf distribution (larger means more skewed)
    
    Returns:
        List of generated elements
    """
    probs = np.array([1/(i**alpha) for i in range(1, domain_size + 1)])
    probs /= probs.sum()
    return np.random.choice(domain_size, size=num_elements, p=probs).tolist()

def generate_normal_data(domain_size: int, num_elements: int, 
                         mean_factor: float = 0.5, std_factor: float = 0.15) -> List[int]:
    """
    Generate normally distributed data centered around the domain.
    
    Args:
        domain_size: Number of unique elements in the domain
        num_elements: Total number of elements to generate
        mean_factor: Factor to multiply domain_size for the mean (0-1)
        std_factor: Factor to multiply domain_size for the standard deviation
    
    Returns:
        List of generated elements
    """
    mean = domain_size * mean_factor
    std = domain_size * std_factor
    data = np.random.normal(mean, std, num_elements)
    # Clip to domain range and convert to integers
    data = np.clip(data, 0, domain_size - 1).astype(int)
    return data.tolist()

def get_true_heavy_hitters(data: List[int], threshold: float) -> Dict[int, float]:
    """
    Calculate the true heavy hitters and their frequencies.
    
    Args:
        data: Input data
        threshold: Frequency threshold
    
    Returns:
        Dictionary mapping heavy hitter elements to their frequencies
    """
    counts = {}
    for element in data:
        counts[element] = counts.get(element, 0) + 1
    
    num_elements = len(data)
    heavy_hitters = {elem: count/num_elements for elem, count in counts.items() 
                    if count/num_elements > threshold}
    return heavy_hitters

def calculate_metrics(true_hh: Dict[int, float], 
                      estimated_hh: Dict[int, float], 
                      all_data: List[int]) -> Dict[str, float]:
    """
    Calculate performance metrics.
    
    Args:
        true_hh: Dictionary of true heavy hitters and their frequencies
        estimated_hh: Dictionary of estimated heavy hitters and their frequencies
        all_data: Complete dataset
    
    Returns:
        Dictionary of performance metrics
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
    
    # Calculate average frequency error
    error_sum = 0
    count = 0
    
    # True frequencies
    true_freqs = {}
    for element in all_data:
        true_freqs[element] = true_freqs.get(element, 0) + 1
    true_freqs = {k: v/len(all_data) for k, v in true_freqs.items()}
    
    # Error for all elements that are in either true or estimated heavy hitters
    all_elements = true_set.union(est_set)
    for element in all_elements:
        true_freq = true_freqs.get(element, 0)
        est_freq = estimated_hh.get(element, 0)
        error_sum += abs(true_freq - est_freq)
        count += 1
    
    avg_error = error_sum / count if count > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'average_error': avg_error,
        'num_true_hh': len(true_set),
        'num_estimated_hh': len(est_set)
    }

def run_test(algorithm_type: str, 
             data_distribution: str, 
             domain_size: int, 
             dataset_size: int, 
             epsilon: float, 
             threshold: float,
             hash_range: int = None,
             num_hash_functions: int = 5) -> Dict[str, Any]:
    """
    Run a single test with specified parameters.
    
    Args:
        algorithm_type: 'basic' or 'cms' (Count-Min Sketch)
        data_distribution: 'uniform', 'zipf', or 'normal'
        domain_size: Number of unique elements in the domain
        dataset_size: Number of data points
        epsilon: Privacy parameter
        threshold: Frequency threshold
        hash_range: Hash range parameter (for basic algorithm)
        num_hash_functions: Number of hash functions (for CMS)
    
    Returns:
        Dictionary of test results
    """
    # Generate data based on specified distribution
    if data_distribution == 'uniform':
        data = generate_uniform_data(domain_size, dataset_size)
    elif data_distribution == 'zipf':
        data = generate_zipf_data(domain_size, dataset_size)
    elif data_distribution == 'normal':
        data = generate_normal_data(domain_size, dataset_size)
    else:
        raise ValueError(f"Unknown data distribution: {data_distribution}")
    
    # Get true heavy hitters
    true_heavy_hitters = get_true_heavy_hitters(data, threshold)
    
    # Initialize and run algorithm
    start_time = time.time()
    
    if algorithm_type == 'basic':
        algorithm = LocalModelHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold,
            hash_range=hash_range
        )
    elif algorithm_type == 'cms':
        algorithm = CountMinSketchHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold,
            num_hash_functions=num_hash_functions
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    # Add elements
    algorithm.add_elements(data)
    
    # Identify heavy hitters
    candidate_set = list(range(domain_size))
    estimated_heavy_hitters = algorithm.identify_heavy_hitters(candidate_set)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(true_heavy_hitters, estimated_heavy_hitters, data)
    metrics['runtime'] = runtime
    
    # Get algorithm statistics
    stats = algorithm.get_statistics()
    
    return {
        'parameters': {
            'algorithm_type': algorithm_type,
            'data_distribution': data_distribution,
            'domain_size': domain_size,
            'dataset_size': dataset_size,
            'epsilon': epsilon,
            'threshold': threshold,
            'hash_range': hash_range if algorithm_type == 'basic' else None,
            'num_hash_functions': num_hash_functions if algorithm_type == 'cms' else None
        },
        'metrics': metrics,
        'statistics': stats
    }

def run_parameter_study(output_dir: str = 'results'):
    """
    Run a comprehensive parameter study and save results.
    
    Args:
        output_dir: Directory to save results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define parameter ranges to test
    algorithm_types = ['basic', 'cms']
    data_distributions = ['uniform', 'zipf', 'normal']
    domain_sizes = [100, 1000, 10000]
    dataset_sizes = [1000, 10000, 100000]
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    thresholds = [0.001, 0.01, 0.05]
    hash_ranges = [None, 10000, 100000]  # None means default (domain_size * 10)
    num_hash_functions_values = [3, 5, 7]
    
    # Store all results
    all_results = []
    
    # For simplicity, we'll use a subset of the full parameter space
    # In practice, you'd want to define specific combinations to test
    
    # Test 1: Vary privacy parameter (epsilon)
    print("Test 1: Varying privacy parameter (epsilon)...")
    for algorithm_type in algorithm_types:
        for epsilon in epsilons:
            result = run_test(
                algorithm_type=algorithm_type,
                data_distribution='zipf',  # Zipf is more realistic
                domain_size=1000,
                dataset_size=10000,
                epsilon=epsilon,
                threshold=0.01,
                hash_range=None,
                num_hash_functions=5
            )
            all_results.append(result)
    
    # Test 2: Vary data distribution
    print("Test 2: Varying data distribution...")
    for algorithm_type in algorithm_types:
        for data_distribution in data_distributions:
            result = run_test(
                algorithm_type=algorithm_type,
                data_distribution=data_distribution,
                domain_size=1000,
                dataset_size=10000,
                epsilon=1.0,
                threshold=0.01,
                hash_range=None,
                num_hash_functions=5
            )
            all_results.append(result)
    
    # Test 3: Vary domain size
    print("Test 3: Varying domain size...")
    for algorithm_type in algorithm_types:
        for domain_size in domain_sizes:
            result = run_test(
                algorithm_type=algorithm_type,
                data_distribution='zipf',
                domain_size=domain_size,
                dataset_size=10000,
                epsilon=1.0,
                threshold=0.01,
                hash_range=None,
                num_hash_functions=5
            )
            all_results.append(result)
    
    # Test 4: Vary dataset size
    print("Test 4: Varying dataset size...")
    for algorithm_type in algorithm_types:
        for dataset_size in dataset_sizes:
            result = run_test(
                algorithm_type=algorithm_type,
                data_distribution='zipf',
                domain_size=1000,
                dataset_size=dataset_size,
                epsilon=1.0,
                threshold=0.01,
                hash_range=None,
                num_hash_functions=5
            )
            all_results.append(result)
    
    # Test 5: Vary threshold
    print("Test 5: Varying threshold...")
    for algorithm_type in algorithm_types:
        for threshold in thresholds:
            result = run_test(
                algorithm_type=algorithm_type,
                data_distribution='zipf',
                domain_size=1000,
                dataset_size=10000,
                epsilon=1.0,
                threshold=threshold,
                hash_range=None,
                num_hash_functions=5
            )
            all_results.append(result)
    
    # Test 6: Vary algorithm-specific parameters
    print("Test 6: Varying algorithm-specific parameters...")
    # For basic algorithm, vary hash range
    for hash_range in hash_ranges:
        if hash_range is not None:  # Skip None since we've tested it in other configurations
            result = run_test(
                algorithm_type='basic',
                data_distribution='zipf',
                domain_size=1000,
                dataset_size=10000,
                epsilon=1.0,
                threshold=0.01,
                hash_range=hash_range,
                num_hash_functions=5
            )
            all_results.append(result)
    
    # For CMS algorithm, vary number of hash functions
    for num_hash_functions in num_hash_functions_values:
        result = run_test(
            algorithm_type='cms',
            data_distribution='zipf',
            domain_size=1000,
            dataset_size=10000,
            epsilon=1.0,
            threshold=0.01,
            hash_range=None,
            num_hash_functions=num_hash_functions
        )
        all_results.append(result)
    
    # Save all results to a file
    results_file = os.path.join(output_dir, 'parameter_study_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Also save as CSV for easier analysis
    # Convert nested dictionaries to flat DataFrame
    rows = []
    for result in all_results:
        row = {}
        # Add parameters
        for param_key, param_value in result['parameters'].items():
            row[f"param_{param_key}"] = param_value
        
        # Add metrics
        for metric_key, metric_value in result['metrics'].items():
            row[f"metric_{metric_value}"] = metric_value
        
        # Add key statistics
        row["stat_total_items"] = result['statistics']['total_items']
        if 'hash_range' in result['statistics']:
            row["stat_hash_range"] = result['statistics']['hash_range']
        if 'sketch_width' in result['statistics']:
            row["stat_sketch_width"] = result['statistics']['sketch_width']
            row["stat_num_hash_functions"] = result['statistics']['num_hash_functions']
        
        rows.append(row)
    
    # Create and save DataFrame
    results_df = pd.DataFrame(rows)
    csv_file = os.path.join(output_dir, 'parameter_study_results.csv')
    results_df.to_csv(csv_file, index=False)
    
    print(f"Parameter study complete. Results saved to {results_file} and {csv_file}")
    
    return all_results

def test_adversarial_scenarios():
    """
    Test the algorithm against potential adversarial scenarios.
    """
    print("Testing adversarial scenarios...")
    
    # Scenario 1: Adversary trying to force collisions
    print("Scenario 1: Adversary forcing hash collisions")
    
    # Generate normal data
    domain_size = 1000
    dataset_size = 10000
    epsilon = 1.0
    threshold = 0.01
    
    normal_data = generate_zipf_data(domain_size, dataset_size)
    
    # Create algorithm instance
    algorithm = LocalModelHeavyHitters(
        domain_size=domain_size,
        epsilon=epsilon,
        threshold=threshold
    )
    
    # Add normal data
    algorithm.add_elements(normal_data)
    
    # Now simulate an adversary who knows the hash function and tries to create
    # elements that hash to the same bucket to cause false positives
    
    # For simplicity, we'll just use the first element and find others that collide
    # In practice, an adversary would need to reverse-engineer the hash function
    target_elem = normal_data[0]
    target_hash = algorithm._hash_element(target_elem)
    
    # Generate adversarial data - elements that hash to the same bucket
    # In a real scenario, this would be more sophisticated
    adversarial_elems = []
    for i in range(domain_size, domain_size + 1000):  # Try elements outside normal domain
        if algorithm._hash_element(i) == target_hash:
            adversarial_elems.append(i)
            if len(adversarial_elems) >= 100:  # Limit to 100 colliding elements
                break
    
    print(f"Found {len(adversarial_elems)} elements that collide with target element {target_elem}")
    
    # Add these adversarial elements to the algorithm
    algorithm.add_elements(adversarial_elems * 10)  # Add each element 10 times
    
    # Check if this causes false positives
    candidate_set = list(range(domain_size)) + adversarial_elems
    heavy_hitters = algorithm.identify_heavy_hitters(candidate_set)
    
    # Count false positives (elements that aren't truly frequent but are reported)
    true_counts = {}
    for elem in normal_data + (adversarial_elems * 10):
        true_counts[elem] = true_counts.get(elem, 0) + 1
    
    true_heavy_hitters = {elem for elem, count in true_counts.items() 
                         if count / (dataset_size + len(adversarial_elems) * 10) > threshold}
    
    false_positives = [elem for elem in heavy_hitters if elem not in true_heavy_hitters]
    
    print(f"Adversarial attack resulted in {len(false_positives)} false positives")
    
    # Scenario 2: Test Count-Min Sketch resilience to the same attack
    print("\nScenario 2: Testing Count-Min Sketch resilience to hash collisions")
    
    cms_algorithm = CountMinSketchHeavyHitters(
        domain_size=domain_size,
        epsilon=epsilon,
        threshold=threshold,
        num_hash_functions=5
    )
    
    # Add normal data
    cms_algorithm.add_elements(normal_data)
    
    # Add adversarial data
    cms_algorithm.add_elements(adversarial_elems * 10)
    
    # Check for false positives
    cms_heavy_hitters = cms_algorithm.identify_heavy_hitters(candidate_set)
    cms_false_positives = [elem for elem in cms_heavy_hitters if elem not in true_heavy_hitters]
    
    print(f"Count-Min Sketch attack resulted in {len(cms_false_positives)} false positives")
    
    # Scenario 3: Privacy leakage test - can we detect a specific rare element?
    print("\nScenario 3: Testing privacy leakage for rare elements")
    
    # Create a new dataset where one specific element appears only once
    rare_elem = domain_size + 2000
    privacy_data = generate_zipf_data(domain_size, dataset_size - 1)
    privacy_data.append(rare_elem)
    
    # Run multiple trials with different epsilon values
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    detection_rates = []
    
    num_trials = 20
    for eps in epsilons:
        detected_count = 0
        
        for _ in range(num_trials):
            # Create a new algorithm instance with this epsilon
            priv_algorithm = LocalModelHeavyHitters(
                domain_size=domain_size + 3000,  # Larger domain to include rare element
                epsilon=eps,
                threshold=0.0001  # Very low threshold to detect rare elements
            )
            
            # Add the data
            priv_algorithm.add_elements(privacy_data)
            
            # Check if the rare element is detected
            rare_freq = priv_algorithm.get_estimated_frequency(rare_elem)
            if rare_freq > 0:
                detected_count += 1
        
        detection_rate = detected_count / num_trials
        detection_rates.append(detection_rate)
        print(f"Epsilon = {eps}: Rare element detection rate = {detection_rate:.2f}")
    
    return {
        'adversarial_collisions': {
            'basic_false_positives': len(false_positives),
            'cms_false_positives': len(cms_false_positives)
        },
        'privacy_leakage': {
            'epsilons': epsilons,
            'detection_rates': detection_rates
        }
    }

if __name__ == "__main__":
    print("Running comprehensive testing for heavy hitters algorithm...")
    
    # Create results directory
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Run parameter study
    results = run_parameter_study(results_dir)
    
    # Test adversarial scenarios
    adversarial_results = test_adversarial_scenarios()
    
    # Save adversarial results
    with open(os.path.join(results_dir, 'adversarial_results.pkl'), 'wb') as f:
        pickle.dump(adversarial_results, f)
    
    print("Testing complete. All results saved to", results_dir)