"""
heavy_hitters_implementation.py
Implementation of multiple local differential privacy algorithms for heavy hitters
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any, Callable
import hashlib
import math
import mmh3  # MurmurHash3 for faster, high-quality hashing
import random

class LocalModelHeavyHitters:
    """
    Implementation of a local model algorithm for identifying heavy hitters with
    differential privacy guarantees using Randomized Response.
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
        self.algorithm_name = "Randomized Response"
        
    def _hash_element(self, element: Any) -> int:
        """
        Hash an element to a fixed range.
        
        Args:
            element: The item to hash
            
        Returns:
            An integer hash value in the range [0, hash_range - 1]
        """
        # Use MurmurHash3 for better performance
        hash_value = mmh3.hash(str(element).encode(), seed=42)
        return abs(hash_value) % self.hash_range
    
    def add_element(self, element: Any) -> None:
        """
        Add an element to the frequency estimation structure with privacy guarantees.
        
        Args:
            element: The item to add
        """
        # Randomized response mechanism for local differential privacy
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
            'algorithm_name': self.algorithm_name,
            'total_items': self.total_items,
            'hash_range': self.hash_range,
            'average_load': self.total_items / self.hash_range if self.hash_range > 0 else 0,
            'max_bucket': np.max(self.frequency_table) if self.total_items > 0 else 0,
            'non_empty_buckets': np.count_nonzero(self.frequency_table),
            'privacy_parameter': self.epsilon,
            'threshold': self.threshold
        }


class OLHHeavyHitters:
    """
    Complete implementation of Optimal Local Hashing (OLH) with all required methods
    """
    
    def __init__(self, domain_size: int, epsilon: float, threshold: float):
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        
        # Improved g calculation
        self.g = max(round(np.exp(epsilon) + 1), int(2/threshold))
        
        self.frequency_table = np.zeros(self.g)
        self.hash_seeds = {}
        self.total_items = 0
        self.algorithm_name = "Optimal Local Hashing (OLH)"

    def _hash_element(self, element: Any, seed: int) -> int:
        element_str = str(element).encode()
        return mmh3.hash(element_str, seed=seed) % self.g

    def _perturb_response(self, hash_val: int) -> int:
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + self.g - 1)
        if random.random() < p:
            return hash_val
        return random.randint(0, self.g - 1)

    def add_element(self, element: Any, client_id: int = None) -> None:
        if client_id is None:
            client_id = self.total_items
            
        if client_id not in self.hash_seeds:
            self.hash_seeds[client_id] = random.randint(0, 2**32)
            
        seed = self.hash_seeds[client_id]
        h = self._hash_element(element, seed)
        y = self._perturb_response(h)
        self.frequency_table[y] += 1
        self.total_items += 1

    def add_elements(self, elements: List[Any]) -> None:
        for i, element in enumerate(elements):
            self.add_element(element, client_id=i)

    def get_estimated_frequency(self, element: Any) -> float:
        if self.total_items == 0:
            return 0
            
        p = np.exp(self.epsilon) / (np.exp(self.epsilon) + self.g - 1)
        q = 1 / (self.g - 1) * (1 - p)
        
        sum_corrected = 0
        valid_seeds = 0
        
        for seed in self.hash_seeds.values():
            h = self._hash_element(element, seed)
            C_y = self.frequency_table[h]
            corrected = (C_y - self.total_items * q) / (p - q)
            sum_corrected += max(0, corrected)
            valid_seeds += 1
            
        return sum_corrected / (valid_seeds * self.total_items)

    def identify_heavy_hitters(self, candidate_set: List[Any]) -> Dict[Any, float]:
        """Identify elements exceeding the frequency threshold"""
        heavy_hitters = {}
        for element in candidate_set:
            freq = self.get_estimated_frequency(element)
            if freq > self.threshold:
                heavy_hitters[element] = freq
        return heavy_hitters

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current state"""
        return {
            'algorithm_name': self.algorithm_name,
            'total_items': self.total_items,
            'g_value': self.g,
            'num_clients': len(self.hash_seeds),
            'average_load': self.total_items / self.g if self.g > 0 else 0,
            'max_bucket': np.max(self.frequency_table),
            'privacy_parameter': self.epsilon,
            'threshold': self.threshold
        }


class RAPPORHeavyHitters:
    """
    Implementation of Google's Randomized Aggregatable Privacy-Preserving Ordinal Response (RAPPOR)
    algorithm for identifying heavy hitters with strong differential privacy guarantees.
    """
    def __init__(self, domain_size: int, epsilon: float, threshold: float, num_hashes: int = 2):
        """
        Initialize the RAPPOR algorithm.
        
        Args:
            domain_size: Number of possible unique elements in the data
            epsilon: Privacy parameter (higher values mean less privacy but better utility)
            threshold: Fraction of the population required to be considered a heavy hitter
            num_hashes: Number of hash functions to use in the Bloom filter
        """
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        self.num_hashes = num_hashes
        # Set the bit array size based on the threshold
        self.bit_array_size = int(2 / threshold)
        self.bloom_filter = np.zeros(self.bit_array_size)
        self.total_reports = 0
        self.algorithm_name = "RAPPOR"

    def _hash_element(self, element: Any) -> List[int]:
        """
        Hash an element to multiple positions using different hash seeds.
        
        Args:
            element: The item to hash
            
        Returns:
            List of hash positions in the Bloom filter
        """
        element_str = str(element).encode()
        return [mmh3.hash(element_str, seed=i) % self.bit_array_size 
                for i in range(self.num_hashes)]

    def add_element(self, element: Any) -> None:
        """
        Add an element to the Bloom filter with privacy guarantees.
        
        Args:
            element: The item to add
        """
        bits = self._hash_element(element)
        p = np.exp(self.epsilon/2) / (np.exp(self.epsilon/2) + 1)
        
        for bit in bits:
            if random.random() < p:
                self.bloom_filter[bit] += 1
        self.total_reports += 1
    
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
        if self.total_reports == 0:
            return 0
            
        bits = self._hash_element(element)
        p = np.exp(self.epsilon/2) / (np.exp(self.epsilon/2) + 1)
        
        sum_bits = sum(self.bloom_filter[bit] for bit in bits)
        corrected = (sum_bits/self.num_hashes - self.total_reports*(1-p)) / (2*p-1)
        return max(0, corrected) / self.total_reports if self.total_reports > 0 else 0
    
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
            'algorithm_name': self.algorithm_name,
            'total_items': self.total_reports,
            'bit_array_size': self.bit_array_size,
            'num_hashes': self.num_hashes,
            'bloom_filter_density': np.mean(self.bloom_filter > 0),
            'max_bit_count': np.max(self.bloom_filter) if self.total_reports > 0 else 0,
            'privacy_parameter': self.epsilon,
            'threshold': self.threshold
        }