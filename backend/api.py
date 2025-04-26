"""
api.py
Flask-based API to expose heavy hitters algorithm functionality
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import json
import traceback
from heavy_hitters_implementation import LocalModelHeavyHitters, CountMinSketchHeavyHitters

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Function to generate data based on distribution
def generate_data(distribution, domain_size, num_elements):
    """Generate data according to the specified distribution."""
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


@app.route('/api/run-algorithm', methods=['POST'])
def run_algorithm():
    """Run the heavy hitters algorithm with the specified parameters."""
    try:
        # Get parameters from request
        params = request.json
        
        algorithm_type = params.get('algorithm_type', 'basic')
        distribution = params.get('distribution', 'zipf')
        domain_size = int(params.get('domain_size', 100))
        num_elements = int(params.get('num_elements', 10000))
        epsilon = float(params.get('epsilon', 1.0))
        threshold = float(params.get('threshold', 0.01))
        num_hash_functions = int(params.get('num_hash_functions', 5))
        
        # Generate data
        data = generate_data(distribution, domain_size, num_elements)
        
        # Calculate true frequencies and heavy hitters
        true_freqs = {}
        for element in data:
            true_freqs[element] = true_freqs.get(element, 0) + 1
        
        total = len(data)
        true_freqs_normalized = {element: count/total for element, count in true_freqs.items()}
        true_heavy_hitters = {elem: freq for elem, freq in true_freqs_normalized.items() if freq > threshold}
        
        # Initialize algorithm
        start_time = time.time()
        
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
                num_hash_functions=num_hash_functions
            )
        else:
            return jsonify({'error': f"Unknown algorithm type: {algorithm_type}"}), 400
        
        # Add elements to the algorithm
        algorithm.add_elements(data)
        
        # Get algorithm statistics
        stats = algorithm.get_statistics()
        
        # Identify heavy hitters
        candidate_set = list(range(domain_size))
        estimated_heavy_hitters = algorithm.identify_heavy_hitters(candidate_set)
        
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
        
        # Prepare visualization data
        all_heavy_hitters = list(true_hh_set.union(est_hh_set))
        all_heavy_hitters.sort(key=lambda x: true_freqs_normalized.get(x, 0), reverse=True)
        all_heavy_hitters = all_heavy_hitters[:30]  # Limit to top 30 for visualization
        
        visualization_data = []
        for element in all_heavy_hitters:
            visualization_data.append({
                'element': element,
                'trueFrequency': true_freqs_normalized.get(element, 0),
                'estimatedFrequency': algorithm.get_estimated_frequency(element)
            })
        
        # Prepare response
        response = {
            'runtime': runtime,
            'metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'statistics': stats,
            'numTrueHeavyHitters': len(true_heavy_hitters),
            'numEstimatedHeavyHitters': len(estimated_heavy_hitters),
            'visualization': visualization_data
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/privacy-impact', methods=['GET'])
def privacy_impact():
    """Get data for privacy impact visualization."""
    try:
        # Default parameters
        domain_size = 100
        num_elements = 5000
        distribution = 'zipf'
        threshold = 0.01
        
        # Generate data
        data = generate_data(distribution, domain_size, num_elements)
        
        # Test different epsilon values
        epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        results = []
        
        for epsilon in epsilon_values:
            # Basic algorithm
            basic_algorithm = LocalModelHeavyHitters(
                domain_size=domain_size,
                epsilon=epsilon,
                threshold=threshold
            )
            basic_algorithm.add_elements(data)
            
            # Count-Min Sketch algorithm
            cms_algorithm = CountMinSketchHeavyHitters(
                domain_size=domain_size,
                epsilon=epsilon,
                threshold=threshold
            )
            cms_algorithm.add_elements(data)
            
            # Calculate true heavy hitters
            true_freqs = {}
            for element in data:
                true_freqs[element] = true_freqs.get(element, 0) + 1
            true_freqs = {elem: count/num_elements for elem, count in true_freqs.items()}
            true_heavy_hitters = {elem for elem, freq in true_freqs.items() if freq > threshold}
            
            # Identify estimated heavy hitters
            candidate_set = list(range(domain_size))
            basic_heavy_hitters = set(basic_algorithm.identify_heavy_hitters(candidate_set).keys())
            cms_heavy_hitters = set(cms_algorithm.identify_heavy_hitters(candidate_set).keys())
            
            # Calculate metrics
            if not basic_heavy_hitters:
                basic_precision = 0.0
            else:
                basic_precision = len(true_heavy_hitters.intersection(basic_heavy_hitters)) / len(basic_heavy_hitters)
            
            if not cms_heavy_hitters:
                cms_precision = 0.0
            else:
                cms_precision = len(true_heavy_hitters.intersection(cms_heavy_hitters)) / len(cms_heavy_hitters)
            
            if not true_heavy_hitters:
                basic_recall = 1.0
                cms_recall = 1.0
            else:
                basic_recall = len(true_heavy_hitters.intersection(basic_heavy_hitters)) / len(true_heavy_hitters)
                cms_recall = len(true_heavy_hitters.intersection(cms_heavy_hitters)) / len(true_heavy_hitters)
            
            basic_f1 = 2 * basic_precision * basic_recall / (basic_precision + basic_recall) if basic_precision + basic_recall > 0 else 0
            cms_f1 = 2 * cms_precision * cms_recall / (cms_precision + cms_recall) if cms_precision + cms_recall > 0 else 0
            
            results.append({
                'epsilon': epsilon,
                'basic': {
                    'precision': basic_precision,
                    'recall': basic_recall,
                    'f1_score': basic_f1
                },
                'cms': {
                    'precision': cms_precision,
                    'recall': cms_recall,
                    'f1_score': cms_f1
                }
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/distribution-impact', methods=['GET'])
def distribution_impact():
    """Get data for distribution impact visualization."""
    try:
        # Default parameters
        domain_size = 100
        num_elements = 5000
        epsilon = 1.0
        threshold = 0.01
        
        distributions = ['uniform', 'zipf', 'normal']
        results = []
        
        for distribution in distributions:
            # Generate data
            data = generate_data(distribution, domain_size, num_elements)
            
            # Basic algorithm
            basic_algorithm = LocalModelHeavyHitters(
                domain_size=domain_size,
                epsilon=epsilon,
                threshold=threshold
            )
            basic_algorithm.add_elements(data)
            
            # Count-Min Sketch algorithm
            cms_algorithm = CountMinSketchHeavyHitters(
                domain_size=domain_size,
                epsilon=epsilon,
                threshold=threshold
            )
            cms_algorithm.add_elements(data)
            
            # Calculate true frequencies and heavy hitters
            true_freqs = {}
            for element in data:
                true_freqs[element] = true_freqs.get(element, 0) + 1
            true_freqs = {elem: count/num_elements for elem, count in true_freqs.items()}
            true_heavy_hitters = {elem for elem, freq in true_freqs.items() if freq > threshold}
            
            # Calculate metrics (same as in privacy_impact)
            candidate_set = list(range(domain_size))
            basic_heavy_hitters = set(basic_algorithm.identify_heavy_hitters(candidate_set).keys())
            cms_heavy_hitters = set(cms_algorithm.identify_heavy_hitters(candidate_set).keys())
            
            if not basic_heavy_hitters:
                basic_precision = 0.0
            else:
                basic_precision = len(true_heavy_hitters.intersection(basic_heavy_hitters)) / len(basic_heavy_hitters)
            
            if not cms_heavy_hitters:
                cms_precision = 0.0
            else:
                cms_precision = len(true_heavy_hitters.intersection(cms_heavy_hitters)) / len(cms_heavy_hitters)
            
            if not true_heavy_hitters:
                basic_recall = 1.0
                cms_recall = 1.0
            else:
                basic_recall = len(true_heavy_hitters.intersection(basic_heavy_hitters)) / len(true_heavy_hitters)
                cms_recall = len(true_heavy_hitters.intersection(cms_heavy_hitters)) / len(true_heavy_hitters)
            
            basic_f1 = 2 * basic_precision * basic_recall / (basic_precision + basic_recall) if basic_precision + basic_recall > 0 else 0
            cms_f1 = 2 * cms_precision * cms_recall / (cms_precision + cms_recall) if cms_precision + cms_recall > 0 else 0
            
            results.append({
                'distribution': distribution,
                'numTrueHeavyHitters': len(true_heavy_hitters),
                'basic': {
                    'precision': basic_precision,
                    'recall': basic_recall,
                    'f1_score': basic_f1
                },
                'cms': {
                    'precision': cms_precision,
                    'recall': cms_recall,
                    'f1_score': cms_f1
                }
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)