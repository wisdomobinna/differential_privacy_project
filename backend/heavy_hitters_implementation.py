"""
heavy_hitters_implementation.py
Core implementation of the local model algorithm for heavy hitters with differential privacy
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any, Callable
import hashlib
import math

class LocalModelHeavyHitters:
    """
    Implementation of a local model algorithm for identifying heavy hitters with
    differential privacy guarantees.
    """
    
    def __init__(self, 
                 domain_size: int, 
                 epsilon: float, 
                 threshold: float,
                 hash_range: int = None):
        """
        Initialize the heavy hitters algorithm.
        
        Args:
            domain_size: Number of possible unique elements in the data
            epsilon: Privacy parameter (higher values mean less privacy but better utility)
            threshold: Fraction of the population required to be considered a heavy hitter
            hash_range: Size of the hash output space (defaults to domain_size * 10)
        """
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        self.hash_range = hash_range if hash_range else domain_size * 10
        
        # Initialize data structures
        self.frequency_table = np.zeros(self.hash_range)
        self.total_items = 0
        
    def _hash_element(self, element: Any) -> int:
        """
        Hash an element to a fixed range.
        
        Args:
            element: The item to hash
            
        Returns:
            An integer hash value in the range [0, hash_range - 1]
        """
        # Use SHA-256 for good distribution
        hash_obj = hashlib.sha256(str(element).encode())
        hash_value = int(hash_obj.hexdigest(), 16)
        return hash_value % self.hash_range
    
    def _add_noise(self, value: float) -> float:
        """
        Add Laplace noise calibrated to the sensitivity and privacy parameter.
        
        Args:
            value: The true value to which noise will be added
            
        Returns:
            Value with differential privacy noise added
        """
        # Scale parameter for Laplace distribution is 1/epsilon for count queries
        scale = 1.0 / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_element(self, element: Any) -> None:
        """
        Add an element to the frequency estimation structure with privacy guarantees.
        
        Args:
            element: The item to add
        """
        # Random response mechanism for local differential privacy
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        
        # With probability p, report true hash; otherwise, report random hash
        if np.random.random() < p:
            hash_idx = self._hash_element(element)
        else:
            hash_idx = np.random.randint(0, self.hash_range)
            
        self.frequency_table[hash_idx] += 1
        self.total_items += 1
    
    def add_elements(self, elements: List[Any]) -> None:
        """
        Add multiple elements to the frequency estimation structure.
        
        Args:
            elements: List of items to add
        """
        for element in elements:
            self.add_element(element)
    
    def get_estimated_frequency(self, element: Any) -> float:
        """
        Get the privacy-preserving estimated frequency of an element.
        
        Args:
            element: The item whose frequency is being estimated
            
        Returns:
            Estimated frequency (between 0 and 1)
        """
        hash_idx = self._hash_element(element)
        raw_count = self.frequency_table[hash_idx]
        
        # Correct for random responses in local differential privacy
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        corrected_count = (raw_count - (self.total_items * (1-p) / self.hash_range)) / p
        
        # Ensure non-negative values
        corrected_count = max(0, corrected_count)
        
        # Return frequency as a fraction of total items
        return corrected_count / self.total_items if self.total_items > 0 else 0
    
    def identify_heavy_hitters(self, candidate_set: List[Any] = None) -> Dict[Any, float]:
        """
        Identify elements exceeding the frequency threshold.
        
        Args:
            candidate_set: Optional list of elements to check (if None, requires domain enumeration)
            
        Returns:
            Dictionary mapping heavy hitter elements to their estimated frequencies
        """
        heavy_hitters = {}
        
        if candidate_set is None:
            # If no candidate set is provided, check elements with high counts in the hash table
            # This is a simplified approach and may miss some heavy hitters due to hash collisions
            for idx in range(self.hash_range):
                if self.frequency_table[idx] / self.total_items > self.threshold:
                    # We'd need to map back from hash to element, which isn't possible directly
                    # In practice, we'd need to maintain a reverse mapping or provide a candidate set
                    heavy_hitters[f"hash_{idx}"] = self.frequency_table[idx] / self.total_items
        else:
            # Check each candidate element
            for element in candidate_set:
                freq = self.get_estimated_frequency(element)
                if freq > self.threshold:
                    heavy_hitters[element] = freq
        
        return heavy_hitters
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the algorithm.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_items': self.total_items,
            'hash_range': self.hash_range,
            'average_load': self.total_items / self.hash_range if self.hash_range > 0 else 0,
            'max_bucket': np.max(self.frequency_table) if self.total_items > 0 else 0,
            'non_empty_buckets': np.count_nonzero(self.frequency_table),
            'privacy_parameter': self.epsilon,
            'threshold': self.threshold
        }


# Alternative implementation using Count-Min Sketch for better accuracy
class CountMinSketchHeavyHitters:
    """
    Implementation of heavy hitters algorithm using Count-Min Sketch with differential privacy.
    """
    
    def __init__(self, 
                 domain_size: int, 
                 epsilon: float, 
                 threshold: float,
                 num_hash_functions: int = 5,
                 width: int = None):
        """
        Initialize the Count-Min Sketch heavy hitters algorithm.
        
        Args:
            domain_size: Number of possible unique elements in the data
            epsilon: Privacy parameter (higher = less privacy)
            threshold: Fraction of population required to be considered a heavy hitter
            num_hash_functions: Number of hash functions to use
            width: Width of each hash table (defaults to e/threshold)
        """
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        self.num_hash_functions = num_hash_functions
        
        # Set width based on error bounds if not specified
        self.width = width if width else int(math.ceil(math.e / threshold))
        
        # Initialize sketch matrix
        self.sketch = np.zeros((self.num_hash_functions, self.width))
        self.total_items = 0
        
        # Generate random seeds for hash functions
        self.hash_seeds = np.random.randint(0, 2**32, size=self.num_hash_functions)
    
    def _hash_element(self, element: Any, seed: int) -> int:
        """
        Hash an element using a specific seed.
        
        Args:
            element: The item to hash
            seed: Seed for the hash function
            
        Returns:
            An integer hash value in the range [0, width - 1]
        """
        hash_obj = hashlib.sha256(f"{seed}:{element}".encode())
        hash_value = int(hash_obj.hexdigest(), 16)
        return hash_value % self.width
    
    def add_element(self, element: Any) -> None:
        """
        Add an element to the Count-Min Sketch with privacy guarantees.
        
        Args:
            element: The item to add
        """
        # Apply local differential privacy: randomized response
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        
        if np.random.random() < p:
            # Report true element
            for i in range(self.num_hash_functions):
                idx = self._hash_element(element, self.hash_seeds[i])
                self.sketch[i, idx] += 1
        else:
            # Report random positions
            for i in range(self.num_hash_functions):
                idx = np.random.randint(0, self.width)
                self.sketch[i, idx] += 1
                
        self.total_items += 1
    
    def add_elements(self, elements: List[Any]) -> None:
        """
        Add multiple elements to the sketch.
        
        Args:
            elements: List of items to add
        """
        for element in elements:
            self.add_element(element)
    
    def get_estimated_frequency(self, element: Any) -> float:
        """
        Get the privacy-preserving estimated frequency of an element.
        
        Args:
            element: The item whose frequency is being estimated
            
        Returns:
            Estimated frequency (between 0 and 1)
        """
        # Calculate raw estimates from all hash functions
        estimates = []
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        
        for i in range(self.num_hash_functions):
            idx = self._hash_element(element, self.hash_seeds[i])
            raw_count = self.sketch[i, idx]
            
            # Correct for random responses
            corrected_count = (raw_count - (self.total_items * (1-p) / self.width)) / p
            corrected_count = max(0, corrected_count)
            estimates.append(corrected_count)
        
        # Take the minimum estimate to reduce the effect of collisions
        best_estimate = min(estimates) if estimates else 0
        
        # Return as frequency
        return best_estimate / self.total_items if self.total_items > 0 else 0
    
    def identify_heavy_hitters(self, candidate_set: List[Any]) -> Dict[Any, float]:
        """
        Identify elements exceeding the frequency threshold.
        
        Args:
            candidate_set: List of elements to check
            
        Returns:
            Dictionary mapping heavy hitter elements to their estimated frequencies
        """
        heavy_hitters = {}
        
        for element in candidate_set:
            freq = self.get_estimated_frequency(element)
            if freq > self.threshold:
                heavy_hitters[element] = freq
        
        return heavy_hitters
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the algorithm.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'total_items': self.total_items,
            'sketch_width': self.width,
            'num_hash_functions': self.num_hash_functions,
            'average_load': self.total_items / (self.width * self.num_hash_functions),
            'max_bucket': np.max(self.sketch),
            'privacy_parameter': self.epsilon,
            'threshold': self.threshold
        }