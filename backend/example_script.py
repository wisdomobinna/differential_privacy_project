"""
example_script.py
Simplified demonstration of the heavy hitters algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
from heavy_hitters_implementation import LocalModelHeavyHitters, CountMinSketchHeavyHitters

def generate_zipf_data(domain_size: int, num_elements: int, alpha: float = 1.07) -> list:
    """
    Generate data following a Zipf distribution (power law).
    
    Args:
        domain_size: Number of unique elements in the domain
        num_elements: Total number of elements to generate
        alpha: Parameter of the Zipf distribution (larger means more skewed)
    
    Returns:
        List of generated elements
    """
    # Generate probabilities following Zipf distribution
    probs = np.array([1/(i**alpha) for i in range(1, domain_size + 1)])
    probs /= probs.sum()  # Normalize to sum to 1
    
    # Sample from this distribution
    elements = np.random.choice(domain_size, size=num_elements, p=probs)
    return elements.tolist()

def main():
    # Parameters
    domain_size = 1000  # Number of possible unique elements
    num_elements = 10000  # Dataset size
    epsilon = 2.0  # Privacy parameter
    threshold = 0.01  # Frequency threshold for heavy hitters
    
    print("Generating Zipf-distributed test data...")
    data = generate_zipf_data(domain_size, num_elements)
    
    # Count true frequencies for comparison
    true_counts = {}
    for element in data:
        true_counts[element] = true_counts.get(element, 0) + 1
    
    # Identify true heavy hitters
    true_heavy_hitters = {elem: count/num_elements for elem, count in true_counts.items() 
                         if count/num_elements > threshold}
    
    print(f"True heavy hitters: {len(true_heavy_hitters)}")
    print("Top 5 true heavy hitters:")
    for item, freq in sorted(true_heavy_hitters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Element {item}: {freq:.4f}")
    
    print("\nRunning LocalModelHeavyHitters algorithm...")
    # Initialize and run the algorithm
    hh_algorithm = LocalModelHeavyHitters(
        domain_size=domain_size,
        epsilon=epsilon,
        threshold=threshold
    )
    
    # Add all elements to the algorithm
    hh_algorithm.add_elements(data)
    
    # Get statistics
    stats = hh_algorithm.get_statistics()
    print("Algorithm statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Find heavy hitters (using all elements as candidates)
    candidate_set = list(range(domain_size))
    estimated_heavy_hitters = hh_algorithm.identify_heavy_hitters(candidate_set)
    
    print(f"\nEstimated heavy hitters: {len(estimated_heavy_hitters)}")
    print("Top 5 estimated heavy hitters:")
    for item, freq in sorted(estimated_heavy_hitters.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  Element {item}: {freq:.4f}")
    
    # Calculate precision and recall
    true_hh_set = set(true_heavy_hitters.keys())
    est_hh_set = set(estimated_heavy_hitters.keys())
    
    precision = len(true_hh_set.intersection(est_hh_set)) / len(est_hh_set) if est_hh_set else 0
    recall = len(true_hh_set.intersection(est_hh_set)) / len(true_hh_set) if true_hh_set else 0
    
    print("\nPerformance metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Demonstrate Count-Min Sketch algorithm
    print("\nRunning CountMinSketchHeavyHitters algorithm...")
    cms_algorithm = CountMinSketchHeavyHitters(
        domain_size=domain_size,
        epsilon=epsilon,
        threshold=threshold
    )
    
    # Add all elements
    cms_algorithm.add_elements(data)
    
    # Find heavy hitters
    cms_heavy_hitters = cms_algorithm.identify_heavy_hitters(candidate_set)
    
    print(f"Count-Min Sketch estimated heavy hitters: {len(cms_heavy_hitters)}")
    
    # Calculate precision and recall for CMS
    cms_hh_set = set(cms_heavy_hitters.keys())
    cms_precision = len(true_hh_set.intersection(cms_hh_set)) / len(cms_hh_set) if cms_hh_set else 0
    cms_recall = len(true_hh_set.intersection(cms_hh_set)) / len(true_hh_set) if true_hh_set else 0
    
    print("Count-Min Sketch performance metrics:")
    print(f"  Precision: {cms_precision:.4f}")
    print(f"  Recall: {cms_recall:.4f}")
    
    # Simple visualization of results
    print("\nPlotting true vs estimated frequencies...")
    plt.figure(figsize=(10, 6))
    
    # Get frequencies for elements that are either true or estimated heavy hitters
    all_heavy_hitters = sorted(list(true_hh_set.union(est_hh_set)))[:20]  # Limit to top 20 for clarity
    
    x = np.arange(len(all_heavy_hitters))
    true_freqs = [true_counts.get(e, 0)/num_elements for e in all_heavy_hitters]
    est_freqs = [hh_algorithm.get_estimated_frequency(e) for e in all_heavy_hitters]
    
    plt.bar(x - 0.2, true_freqs, 0.4, label='True Frequency')
    plt.bar(x + 0.2, est_freqs, 0.4, label='Estimated Frequency')
    
    plt.xlabel('Element ID')
    plt.ylabel('Frequency')
    plt.title(f'True vs Estimated Frequencies (ε = {epsilon})')
    plt.xticks(x, all_heavy_hitters)
    plt.legend()
    plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
    plt.tight_layout()
    plt.savefig('frequency_comparison.png')
    print("Plot saved as 'frequency_comparison.png'")
    
    # Compare both algorithms
    print("\nComparing both algorithms...")
    common_elements = list(true_hh_set.union(est_hh_set).union(cms_hh_set))[:15]  # Top 15
    
    x = np.arange(len(common_elements))
    true_freqs = [true_counts.get(e, 0)/num_elements for e in common_elements]
    basic_est_freqs = [hh_algorithm.get_estimated_frequency(e) for e in common_elements]
    cms_est_freqs = [cms_algorithm.get_estimated_frequency(e) for e in common_elements]
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.3, true_freqs, 0.2, label='True Frequency')
    plt.bar(x, basic_est_freqs, 0.2, label='Basic Algorithm')
    plt.bar(x + 0.3, cms_est_freqs, 0.2, label='Count-Min Sketch')
    
    plt.xlabel('Element ID')
    plt.ylabel('Frequency')
    plt.title(f'Algorithm Comparison (ε = {epsilon})')
    plt.xticks(x, common_elements)
    plt.legend()
    plt.axhline(y=threshold, color='r', linestyle='-', label='Threshold')
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    print("Comparison plot saved as 'algorithm_comparison.png'")

if __name__ == "__main__":
    main()