"""
demo_script.py
Simple demonstration of the heavy hitters algorithm with interactive components
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from typing import List, Dict, Any
from heavy_hitters_implementation import LocalModelHeavyHitters, CountMinSketchHeavyHitters


def generate_data(distribution: str, domain_size: int, num_elements: int) -> List[int]:
    """
    Generate data according to the specified distribution.
    
    Args:
        distribution: Data distribution type ('uniform', 'zipf', or 'normal')
        domain_size: Number of unique elements in the domain
        num_elements: Total number of elements to generate
    
    Returns:
        List of generated data
    """
    if distribution == 'uniform':
        # Uniform distribution
        return np.random.randint(0, domain_size, size=num_elements).tolist()
    
    elif distribution == 'zipf':
        # Zipf distribution (power law)
        alpha = 1.07  # Parameter controlling the skewness
        probs = np.array([1/(i**alpha) for i in range(1, domain_size + 1)])
        probs /= probs.sum()
        return np.random.choice(domain_size, size=num_elements, p=probs).tolist()
    
    elif distribution == 'normal':
        # Normal distribution centered around the domain
        mean = domain_size / 2
        std = domain_size / 6  # Standard deviation
        data = np.random.normal(mean, std, num_elements)
        # Clip to domain range and convert to integers
        data = np.clip(data, 0, domain_size - 1).astype(int)
        return data.tolist()
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def calculate_true_frequencies(data: List[int]) -> Dict[int, float]:
    """
    Calculate the true frequencies of elements in the data.
    
    Args:
        data: Input data
    
    Returns:
        Dictionary mapping elements to their frequencies
    """
    counts = {}
    for element in data:
        counts[element] = counts.get(element, 0) + 1
    
    total = len(data)
    frequencies = {element: count/total for element, count in counts.items()}
    
    return frequencies


def run_demo(params: Dict[str, Any]):
    """
    Run the algorithm with the specified parameters and show results.
    
    Args:
        params: Dictionary of algorithm parameters
    """
    print("\n===== Heavy Hitters Algorithm Demonstration =====\n")
    
    # Extract parameters
    algorithm_type = params.get('algorithm_type', 'basic')
    distribution = params.get('distribution', 'zipf')
    domain_size = params.get('domain_size', 100)
    num_elements = params.get('num_elements', 10000)
    epsilon = params.get('epsilon', 1.0)
    threshold = params.get('threshold', 0.01)
    
    print(f"Parameters:")
    print(f"  Algorithm: {algorithm_type}")
    print(f"  Data Distribution: {distribution}")
    print(f"  Domain Size: {domain_size}")
    print(f"  Number of Elements: {num_elements}")
    print(f"  Privacy Parameter (ε): {epsilon}")
    print(f"  Threshold: {threshold}")
    
    # Generate data
    print("\nGenerating data...")
    data = generate_data(distribution, domain_size, num_elements)
    print(f"Generated {len(data)} elements")
    
    # Calculate true frequencies and heavy hitters
    true_freqs = calculate_true_frequencies(data)
    true_heavy_hitters = {elem: freq for elem, freq in true_freqs.items() if freq > threshold}
    
    print(f"\nTrue Heavy Hitters: {len(true_heavy_hitters)}")
    print("Top 5 true heavy hitters:")
    for item, freq in sorted(true_heavy_hitters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Element {item}: {freq:.4f}")
    
    # Initialize algorithm
    print(f"\nInitializing {algorithm_type} algorithm...")
    
    if algorithm_type == 'basic':
        algorithm = LocalModelHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold
        )
    elif algorithm_type == 'cms':
        algorithm = CountMinSketchHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold,
            num_hash_functions=5
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    # Process data
    print("Processing data...")
    start_time = time.time()
    
    # Process in batches to show progress
    batch_size = min(1000, num_elements)
    num_batches = (num_elements + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_elements)
        
        batch = data[start_idx:end_idx]
        algorithm.add_elements(batch)
        
        # Show progress
        progress = end_idx / num_elements * 100
        print(f"  Progress: {progress:.1f}% ({end_idx}/{num_elements})", end='\r')
    
    print("\nData processing complete                    ")
    
    # Get algorithm statistics
    stats = algorithm.get_statistics()
    print("\nAlgorithm Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Identify heavy hitters
    print("\nIdentifying heavy hitters...")
    candidate_set = list(range(domain_size))
    estimated_heavy_hitters = algorithm.identify_heavy_hitters(candidate_set)
    
    print(f"Estimated Heavy Hitters: {len(estimated_heavy_hitters)}")
    print("Top 5 estimated heavy hitters:")
    for item, freq in sorted(estimated_heavy_hitters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Element {item}: {freq:.4f}")
    
    # Calculate metrics
    end_time = time.time()
    runtime = end_time - start_time
    
    true_hh_set = set(true_heavy_hitters.keys())
    est_hh_set = set(estimated_heavy_hitters.keys())
    
    if not est_hh_set:
        precision = 0.0
    else:
        precision = len(true_hh_set.intersection(est_hh_set)) / len(est_hh_set)
    
    if not true_hh_set:
        recall = 1.0
    else:
        recall = len(true_hh_set.intersection(est_hh_set)) / len(true_hh_set)
    
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    print("\nPerformance Metrics:")
    print(f"  Runtime: {runtime:.4f} seconds")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    
    # Get top heavy hitters for visualization
    all_elements = sorted(list(true_hh_set.union(est_hh_set)))[:20]  # Limit to top 20
    
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    x = np.arange(len(all_elements))
    true_vals = [true_freqs.get(e, 0) for e in all_elements]
    est_vals = [algorithm.get_estimated_frequency(e) for e in all_elements]
    
    # Create bar chart
    plt.bar(x - 0.2, true_vals, 0.4, label='True Frequency')
    plt.bar(x + 0.2, est_vals, 0.4, label='Estimated Frequency')
    
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Element')
    plt.ylabel('Frequency')
    plt.title(f'Heavy Hitters: True vs. Estimated Frequencies (ε = {epsilon})')
    plt.xticks(x, all_elements, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and show the plot
    plt.savefig('demo_result.png')
    print("Visualization saved as 'demo_result.png'")
    
    try:
        plt.show()
    except:
        print("Could not display the plot (no GUI available)")
    
    print("\n===== Demonstration Complete =====")


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Heavy Hitters Algorithm Demo")
    
    parser.add_argument('--algorithm', type=str, choices=['basic', 'cms'], default='basic',
                        help='Algorithm type (basic or Count-Min Sketch)')
    parser.add_argument('--distribution', type=str, choices=['uniform', 'zipf', 'normal'], default='zipf',
                        help='Data distribution')
    parser.add_argument('--domain-size', type=int, default=100,
                        help='Domain size (number of possible unique elements)')
    parser.add_argument('--num-elements', type=int, default=10000,
                        help='Number of data elements to generate')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Privacy parameter (higher values mean less privacy)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Frequency threshold for heavy hitters')
    
    return parser.parse_args()


def main():
    """
    Main function to run the demo.
    """
    args = parse_args()
    
    # Convert arguments to parameter dictionary
    params = {
        'algorithm_type': args.algorithm,
        'distribution': args.distribution,
        'domain_size': args.domain_size,
        'num_elements': args.num_elements,
        'epsilon': args.epsilon,
        'threshold': args.threshold
    }
    
    # Run the demo
    run_demo(params)


if __name__ == "__main__":
    main()