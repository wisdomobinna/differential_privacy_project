"""
experiment_runner.py
Experiment automation and reporting for the heavy hitters algorithm
"""

import os
import sys
import time
import argparse
import json
from typing import Dict, List, Any
from datetime import datetime

# Import other scripts
from testing_script import run_test, run_parameter_study, test_adversarial_scenarios
from visualization_script import create_all_visualizations, generate_summary_report


class ExperimentRunner:
    """
    Class to manage the execution of experiments for the heavy hitters algorithm.
    """
    
    def __init__(self, config_file: str = None):
        """
        Initialize the experiment runner.
        
        Args:
            config_file: Path to a JSON configuration file (optional)
        """
        self.config = self._load_config(config_file)
        self.results_dir = self.config.get('results_dir', 'results')
        self.visualizations_dir = self.config.get('visualizations_dir', 'visualizations')
        
        # Create directories if they don't exist
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Set up logging
        self.log_file = os.path.join(self.results_dir, 'experiment_log.txt')
    
    def _load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load configuration from a JSON file, or use defaults.
        
        Args:
            config_file: Path to a JSON configuration file
            
        Returns:
            Dictionary containing configuration
        """
        default_config = {
            'results_dir': 'results',
            'visualizations_dir': 'visualizations',
            'tests': {
                'parameter_study': True,
                'adversarial': True
            },
            'parameter_study_config': {
                'algorithm_types': ['basic', 'cms'],
                'data_distributions': ['uniform', 'zipf', 'normal'],
                'domain_sizes': [100, 1000, 10000],
                'dataset_sizes': [1000, 10000, 100000],
                'epsilons': [0.1, 0.5, 1.0, 2.0, 5.0],
                'thresholds': [0.001, 0.01, 0.05],
                'hash_ranges': [None, 10000, 100000],
                'num_hash_functions_values': [3, 5, 7]
            }
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Update default config with user config
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def log(self, message: str):
        """
        Log a message to both console and log file.
        
        Args:
            message: Message to log
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def run_quick_test(self):
        """
        Run a quick test to verify the algorithm works.
        """
        self.log("Running quick test...")
        
        result = run_test(
            algorithm_type='basic',
            data_distribution='zipf',
            domain_size=100,
            dataset_size=1000,
            epsilon=1.0,
            threshold=0.01
        )
        
        self.log(f"Quick test completed. F1 score: {result['metrics']['f1_score']:.4f}")
        return result
    
    def run_all_experiments(self):
        """
        Run all configured experiments.
        """
        start_time = time.time()
        self.log("Starting all experiments...")
        
        # Run quick test first
        self.run_quick_test()
        
        # Run parameter study if configured
        if self.config['tests']['parameter_study']:
            self.log("Running parameter study...")
            run_parameter_study(self.results_dir)
            self.log("Parameter study complete.")
        
        # Run adversarial scenarios if configured
        if self.config['tests']['adversarial']:
            self.log("Running adversarial testing...")
            adversarial_results = test_adversarial_scenarios()
            self.log("Adversarial testing complete.")
        
        # Generate visualizations
        self.log("Generating visualizations...")
        create_all_visualizations(self.results_dir, self.visualizations_dir)
        self.log("Visualizations complete.")
        
        # Generate summary report
        self.log("Generating summary report...")
        generate_summary_report(self.results_dir, self.visualizations_dir)
        self.log("Summary report complete.")
        
        end_time = time.time()
        total_time = end_time - start_time
        self.log(f"All experiments completed in {total_time:.2f} seconds.")
    
    def run_custom_experiment(self, params: Dict[str, Any]):
        """
        Run a custom experiment with specified parameters.
        
        Args:
            params: Dictionary of test parameters
        """
        self.log(f"Running custom experiment with parameters: {params}")
        
        # Extract parameters
        algorithm_type = params.get('algorithm_type', 'basic')
        data_distribution = params.get('data_distribution', 'zipf')
        domain_size = params.get('domain_size', 1000)
        dataset_size = params.get('dataset_size', 10000)
        epsilon = params.get('epsilon', 1.0)
        threshold = params.get('threshold', 0.01)
        hash_range = params.get('hash_range', None)
        num_hash_functions = params.get('num_hash_functions', 5)
        
        # Run the test
        result = run_test(
            algorithm_type=algorithm_type,
            data_distribution=data_distribution,
            domain_size=domain_size,
            dataset_size=dataset_size,
            epsilon=epsilon,
            threshold=threshold,
            hash_range=hash_range,
            num_hash_functions=num_hash_functions
        )
        
        # Log and save results
        self.log(f"Custom experiment completed. Results:")
        self.log(f"  Precision: {result['metrics']['precision']:.4f}")
        self.log(f"  Recall: {result['metrics']['recall']:.4f}")
        self.log(f"  F1 Score: {result['metrics']['f1_score']:.4f}")
        self.log(f"  Average Error: {result['metrics']['average_error']:.4f}")
        self.log(f"  Runtime: {result['metrics']['runtime']:.4f} seconds")
        
        # Save the result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(self.results_dir, f"custom_experiment_{timestamp}.json")
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.log(f"Results saved to {result_file}")
        
        return result


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Heavy Hitters Algorithm Experiment Runner")
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--quick-test', action='store_true', help='Run a quick test only')
    parser.add_argument('--custom', action='store_true', help='Run a custom experiment')
    parser.add_argument('--algorithm', type=str, choices=['basic', 'cms'], help='Algorithm type for custom experiment')
    parser.add_argument('--distribution', type=str, choices=['uniform', 'zipf', 'normal'], help='Data distribution for custom experiment')
    parser.add_argument('--domain-size', type=int, help='Domain size for custom experiment')
    parser.add_argument('--dataset-size', type=int, help='Dataset size for custom experiment')
    parser.add_argument('--epsilon', type=float, help='Privacy parameter for custom experiment')
    parser.add_argument('--threshold', type=float, help='Threshold for custom experiment')
    
    return parser.parse_args()


def main():
    """
    Main function to run experiments.
    """
    args = parse_args()
    
    # Initialize experiment runner
    runner = ExperimentRunner(args.config)
    
    if args.quick_test:
        # Run only a quick test
        runner.run_quick_test()
    elif args.custom:
        # Run a custom experiment
        params = {}
        if args.algorithm:
            params['algorithm_type'] = args.algorithm
        if args.distribution:
            params['data_distribution'] = args.distribution
        if args.domain_size:
            params['domain_size'] = args.domain_size
        if args.dataset_size:
            params['dataset_size'] = args.dataset_size
        if args.epsilon:
            params['epsilon'] = args.epsilon
        if args.threshold:
            params['threshold'] = args.threshold
        
        runner.run_custom_experiment(params)
    else:
        # Run all experiments
        runner.run_all_experiments()


if __name__ == "__main__":
    main()