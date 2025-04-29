"""
heavy_hitters_implementation.py
Implementation of local differential privacy algorithms for heavy hitters using IBM's diffprivlib
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any, Callable
import mmh3  # MurmurHash3 for faster, high-quality hashing
import random
from typing import List, Dict, Any, Optional, Tuple

# Import our bit-by-bit implementation
from bit_by_bit_heavyhitters import BitByBitHeavyHitters

# Import IBM's diffprivlib
try:
    import diffprivlib.mechanisms as mechanisms
    from diffprivlib.mechanisms import Laplace, Binary, RandomizedResponse
    import diffprivlib.tools as tools
    DIFFPRIVLIB_AVAILABLE = True
    print("All imports successful ✅")
except ImportError:
    DIFFPRIVLIB_AVAILABLE = False
    print("IBM's diffprivlib not found. Install with: pip install diffprivlib")



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
        
        # Initialize IBM's randomized response mechanism if available
        if DIFFPRIVLIB_AVAILABLE:
            self.rand_mechanism = mechanisms.RandomizedResponse(epsilon=epsilon, 
                                                              values=list(range(self.hash_range)))
        
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
        # Get true hash value
        hash_idx = self._hash_element(element)
        
        # Apply differential privacy using IBM's library if available
        if DIFFPRIVLIB_AVAILABLE:
            # Use IBM's randomized response mechanism
            privatized_hash = self.rand_mechanism.randomise(hash_idx)
        else:
            # Fallback to basic randomized response
            p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
            if np.random.random() < p:
                privatized_hash = hash_idx
            else:
                privatized_hash = np.random.randint(0, self.hash_range)
                
        self.frequency_table[privatized_hash] += 1
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
        if self.total_items == 0:
            return 0
            
        hash_idx = self._hash_element(element)
        raw_count = self.frequency_table[hash_idx]
        
        # Correct for random responses in local differential privacy
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        corrected_count = (raw_count - (self.total_items * (1-p) / self.hash_range)) / p
        
        # Ensure non-negative values
        corrected_count = max(0, corrected_count)
        
        # Return frequency as a fraction of total items
        return corrected_count / self.total_items
    
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
            'threshold': self.threshold,
            'using_diffprivlib': DIFFPRIVLIB_AVAILABLE
        }



class OLHHeavyHitters:
    """
    Optimized implementation of the OLH (Optimized Local Hashing) algorithm
    for heavy hitter detection with local differential privacy guarantees.
    
    Key Features:
    - Correct parameter selection following theoretical guidelines
    - Vectorized operations for better performance
    - Memory-efficient data structures
    - Numerically stable calculations
    - Batch processing support
    """
    
    def __init__(self, domain_size: int, epsilon: float, threshold: float):
        """
        Initialize the OLH heavy hitter detector.
        
        Args:
            domain_size: Estimated number of unique items in the domain
            epsilon: Privacy budget (ε > 0)
            threshold: Frequency threshold for heavy hitters (0 < threshold < 1)
        """
        # Validate inputs
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if not (0 < threshold < 1):
            raise ValueError("Threshold must be between 0 and 1")
        
        self.domain_size = domain_size
        self.epsilon = float(epsilon)
        self.threshold = float(threshold)
        
        # Optimal parameter selection (follows original OLH paper)
        self.g = max(2, int(np.exp(epsilon)) + 1)  # Hash range
        self.p = np.exp(epsilon) / (np.exp(epsilon) + self.g - 1)  # Probability
        
        # Data structures
        self.client_data = {}  # Maps client_id -> (seed, report)
        self.frequency_table = np.zeros(self.g, dtype=np.int32)  # Counts per hash
        self.total_items = 0
        
        # Cache for frequent elements
        self._frequency_cache = {}
        self._cache_valid = False
        
    def _hash_element(self, element: Any, seed: int) -> int:
        """
        Hash an element to a value in [0, g-1] using MurmurHash3.
        """
        element_str = str(element).encode('utf-8')
        # Convert numpy.uint32 to int explicitly
        seed_int = int(seed)
        return mmh3.hash(element_str, seed=seed_int) % self.g
    
    def add_element(self, element: Any, client_id: Optional[int] = None) -> None:
        """
        Add a single element with privacy protection.
        
        Args:
            element: The item to add
            client_id: Optional unique identifier for the client
        """
        if client_id is None:
            client_id = self.total_items
            
        if client_id not in self.client_data:
            # Generate random seed for this client
            seed = random.randint(0, 2**32 - 1)
            
            # Compute true hash
            true_hash = self._hash_element(element, seed)
            
            # Apply randomized response
            if random.random() < self.p:
                report = true_hash
            else:
                report = random.randint(0, self.g - 1)
            
            # Store the report
            self.client_data[client_id] = (seed, report)
            self.frequency_table[report] += 1
            self.total_items += 1
            self._cache_valid = False
    
    def add_elements(self, elements: List[Any]) -> None:
        """
        Add multiple elements efficiently.
        
        Args:
            elements: List of items to add
        """
        new_ids = range(self.total_items, self.total_items + len(elements))
        seeds = np.random.randint(0, 2**32, size=len(elements), dtype=np.uint32)
        
        # Vectorized hashing - convert seeds to int explicitly
        hashes = np.array([self._hash_element(el, int(seed)) for el, seed in zip(elements, seeds)])
        
        # Vectorized randomization
        rand_mask = np.random.random(size=len(elements)) < self.p
        reports = np.where(
            rand_mask,
            hashes,
            np.random.randint(0, self.g, size=len(elements), dtype=np.int32)
        )
        
        # Update data structures
        for client_id, seed, report in zip(new_ids, seeds, reports):
            self.client_data[client_id] = (int(seed), int(report))
            self.frequency_table[report] += 1
        
        self.total_items += len(elements)
        self._cache_valid = False
    
    def get_estimated_frequency(self, element: Any) -> float:
        """
        Estimate the true frequency of an element.
        
        Args:
            element: The item to estimate frequency for
            
        Returns:
            Estimated frequency in [0, 1]
        """
        if self.total_items == 0:
            return 0.0
        
        # Check cache first
        element_str = str(element)
        if self._cache_valid and element_str in self._frequency_cache:
            return self._frequency_cache[element_str]
        
        # Count matches
        matches = 0
        for seed, report in self.client_data.values():
            if report == self._hash_element(element, seed):
                matches += 1
        
        # Compute estimate
        n = self.total_items
        denominator = n * (self.p - 1 / self.g)
        
        # Handle numerical stability
        if denominator < 1e-10:
            return 0.0
            
        est = (matches - n / self.g) / denominator
        
        # Cache the result
        if self._cache_valid:
            self._frequency_cache[element_str] = np.clip(est, 0.0, 1.0)
        
        return np.clip(est, 0.0, 1.0)
    
    def identify_heavy_hitters(self, candidate_set: List[Any]) -> Dict[Any, float]:
        """
        Identify all heavy hitters from a candidate set.
        
        Args:
            candidate_set: List of potential heavy hitters to check
            
        Returns:
            Dictionary of {item: estimated_frequency} for heavy hitters
        """
        self._frequency_cache = {}
        heavy_hitters = {}
        
        for element in candidate_set:
            freq = self.get_estimated_frequency(element)
            if freq >= self.threshold:
                heavy_hitters[element] = freq
        
        self._cache_valid = True
        return heavy_hitters
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the current state of the algorithm.
        
        Returns:
            Dictionary of statistics including privacy parameters,
            data sizes, and performance metrics
        """
        return {
            'algorithm': 'Optimized Local Hashing (OLH)',
            'domain_size': self.domain_size,
            'epsilon': self.epsilon,
            'threshold': self.threshold,
            'hash_range_g': self.g,
            'probability_p': self.p,
            'total_clients': self.total_items,
            'unique_clients': len(self.client_data),
            'frequency_table_nonzero': np.count_nonzero(self.frequency_table),
            'frequency_table_max': np.max(self.frequency_table),
            'frequency_table_mean': np.mean(self.frequency_table),
        }
    
    def reset(self) -> None:
        """Reset all collected data while keeping parameters."""
        self.client_data.clear()
        self.frequency_table.fill(0)
        self.total_items = 0
        self._frequency_cache.clear()
        self._cache_valid = False
    

class RAPPORHeavyHitters:
    """
    Implementation of RAPPOR using IBM's diffprivlib mechanisms.
    Based on Google's Randomized Aggregatable Privacy-Preserving Ordinal Response.
    """
    def __init__(self, domain_size: int, epsilon: float, threshold: float, num_hashes: int = 4):
        """
        Initialize the RAPPOR algorithm with IBM's diffprivlib components.
        
        Args:
            domain_size: Number of possible unique elements in the data
            epsilon: Privacy parameter (higher values mean less privacy but better utility)
            threshold: Fraction of the population required to be considered a heavy hitter
            num_hashes: Number of hash functions to use in the Bloom filter (default: 4)
        """
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        self.num_hashes = num_hashes
        
        # Calculate optimal bit array size based on domain and threshold
        self.bit_array_size = max(int(1.44 * domain_size * np.log(1/threshold)), 
                                 domain_size)
        
        # Initialize data structures
        self.bloom_filter_counts = np.zeros(self.bit_array_size)
        self.total_reports = 0
        self.algorithm_name = "IBM RAPPOR"
        
        # Storage for client reports
        self.client_reports = []
        
        # Initialize IBM's mechanisms if available
        if DIFFPRIVLIB_AVAILABLE:
            # For permanent randomization
            self.perm_mechanism = mechanisms.Binary(epsilon=epsilon/2)
            
            # For instantaneous randomization 
            self.inst_mechanism = mechanisms.Binary(epsilon=epsilon/2)
        else:
            self.perm_mechanism = None
            self.inst_mechanism = None
        
    def _hash_element(self, element: Any) -> List[int]:
        """
        Hash an element to multiple positions in the Bloom filter.
        
        Args:
            element: The item to hash
            
        Returns:
            List of hash positions (bit indices)
        """
        element_str = str(element).encode()
        return [mmh3.hash(element_str, seed=i) % self.bit_array_size 
                for i in range(self.num_hashes)]
                
    def add_element(self, element: Any) -> None:
        """
        Add an element using the RAPPOR algorithm.
        
        Args:
            element: The item to add
        """
        # Create initial Bloom filter
        bloom_filter = np.zeros(self.bit_array_size, dtype=bool)
        
        # Set bits for the element
        for bit_pos in self._hash_element(element):
            bloom_filter[bit_pos] = True
            
        # Apply RAPPOR's two-phase randomization
        privatized_bits = np.zeros(self.bit_array_size, dtype=bool)
        
        # Probability parameters
        if DIFFPRIVLIB_AVAILABLE:
            # Use IBM's binary mechanisms for better guarantees
            for i in range(self.bit_array_size):
                privatized_bits[i] = self.perm_mechanism.randomise(bloom_filter[i])
                privatized_bits[i] = self.inst_mechanism.randomise(privatized_bits[i])
        else:
            # Fallback to standard RAPPOR implementation
            p = np.exp(self.epsilon/2) / (np.exp(self.epsilon/2) + 1)
            
            for i in range(self.bit_array_size):
                # Apply randomization
                if bloom_filter[i]:
                    privatized_bits[i] = np.random.random() < p
                else:
                    privatized_bits[i] = np.random.random() < (1-p)
        
        # Store the privatized report
        self.client_reports.append(privatized_bits)
        
        # Update counts in the aggregate Bloom filter
        for i in range(self.bit_array_size):
            if privatized_bits[i]:
                self.bloom_filter_counts[i] += 1
                
        self.total_reports += 1
        
    def add_elements(self, elements: List[Any]) -> None:
        """
        Add multiple elements using the RAPPOR algorithm.
        
        Args:
            elements: List of items to add
        """
        for element in elements:
            self.add_element(element)
            
    def get_estimated_frequency(self, element: Any) -> float:
        """
        Estimate the frequency of an element using RAPPOR.
        
        Args:
            element: The item whose frequency is being estimated
            
        Returns:
            Estimated frequency (between 0 and 1)
        """
        if self.total_reports == 0:
            return 0
            
        # Get bit positions for this element
        bit_positions = self._hash_element(element)
        
        # Calculate parameters for correction
        p = np.exp(self.epsilon/2) / (np.exp(self.epsilon/2) + 1)
        q = 0.5  # Probability of reporting 1 for a 0-bit
        
        # Calculate average observed frequency for these bits
        bit_freqs = [self.bloom_filter_counts[pos] / self.total_reports for pos in bit_positions]
        avg_freq = np.mean(bit_freqs)
        
        # Apply correction formula
        corrected = (avg_freq - q) / (p - q)
        
        # Adjust for false positives in Bloom filter (approximation)
        false_pos_prob = (1 - np.exp(-self.num_hashes * self.domain_size / self.bit_array_size)) ** self.num_hashes
        corrected = min(1.0, max(0.0, (corrected - false_pos_prob) / (1 - false_pos_prob)))
        
        return corrected
        
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
        if self.total_reports > 0:
            bloom_density = np.sum(self.bloom_filter_counts > 0) / self.bit_array_size
            max_count = np.max(self.bloom_filter_counts)
        else:
            bloom_density = 0.0
            max_count = 0
            
        return {
            'algorithm_name': self.algorithm_name,
            'total_items': self.total_reports,
            'bit_array_size': self.bit_array_size,
            'num_hashes': self.num_hashes,
            'bloom_filter_density': bloom_density,
            'max_bit_count': max_count,
            'privacy_parameter': self.epsilon,
            'threshold': self.threshold,
            'using_diffprivlib': DIFFPRIVLIB_AVAILABLE
        }