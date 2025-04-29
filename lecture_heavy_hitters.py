"""
lecture_heavy_hitters.py
Implementation of the bit-by-bit heavy hitters algorithm from lecture 6 on local differential privacy
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Any, Callable, Optional
import mmh3  # MurmurHash3 for faster, high-quality hashing
import random
import math
from collections import defaultdict

class LectureLDPFrequencyOracle:
    """
    Frequency Oracle implementation following the lecture's approach
    """
    def __init__(self, epsilon: float, hash_range: int):
        """
        Initialize a frequency oracle with the given privacy parameter
        
        Args:
            epsilon: Privacy parameter (higher values mean less privacy but better utility)
            hash_range: Size of the hash output space
        """
        self.epsilon = epsilon
        self.hash_range = hash_range
        
        # For randomized response
        self.p = np.exp(epsilon) / (1 + np.exp(epsilon))  # Prob of keeping true value
        
        # Matrix Z as described in the lecture (random {-1, 1} matrix)
        # In practice, we'll compute this on the fly using hash functions
        
        # Storage for user responses
        self.user_responses = []
        self.total_users = 0
    
    def randomized_response(self, value: int) -> int:
        """
        Apply randomized response to a value
        
        Args:
            value: True value (0 or 1)
            
        Returns:
            Randomized value (0 or 1)
        """
        if random.random() < self.p:
            return value
        else:
            return 1 - value
    
    def get_matrix_value(self, x: Any, i: int) -> int:
        """
        Get the value of Z[x,i] from the random matrix
        Using a hash function to simulate a random {-1, 1} matrix
        
        Args:
            x: Element (row index)
            i: Column index
            
        Returns:
            -1 or 1
        """
        # Use hash function to deterministically generate "random" matrix values
        hash_val = mmh3.hash(f"{str(x)}:{i}", seed=42)
        return 1 if hash_val % 2 == 0 else -1
    
    def add_user_data(self, x: Any) -> None:
        """
        Add a user's data point to the frequency oracle
        
        Args:
            x: User's data point
        """
        # Generate random index i for the user
        i = self.total_users
        
        # Get Z[x,i]
        z_val = self.get_matrix_value(x, i)
        
        # Apply randomized response to Z[x,i]
        rr_val = self.randomized_response(1 if z_val == 1 else 0)
        
        # Store the response (i, RR(Z[x,i]))
        self.user_responses.append((i, rr_val))
        self.total_users += 1
    
    def estimate_frequency(self, x: Any) -> float:
        """
        Estimate the frequency of an element
        
        Args:
            x: Element to estimate
            
        Returns:
            Estimated frequency (between 0 and 1)
        """
        if self.total_users == 0:
            return 0
        
        # Sum of user responses multiplied by Z[x,i]
        sum_val = 0
        for i, rr_val in self.user_responses:
            z_val = self.get_matrix_value(x, i)
            # Convert back from {0,1} to {-1,1}
            user_val = 2 * rr_val - 1
            sum_val += user_val * z_val
        
        # Apply correction formula from lecture
        epsilon_prime = np.log((self.p) / (1 - self.p))  # Equivalent to log((e^ε)/(1))
        correction_factor = 1 / (2 * self.p - 1)
        
        # Apply correction (sum_val/n is the biased estimate)
        corrected = correction_factor * (sum_val / self.total_users)
        
        # Ensure result is in [0,1]
        return max(0.0, min(1.0, (corrected + 1) / 2))  # Convert from [-1,1] to [0,1]


class LectureLDPHeavyHitters:
    """
    Implementation of the exact bit-by-bit heavy hitters algorithm from lecture 6
    """
    def __init__(self, 
                domain_size: int, 
                epsilon: float, 
                threshold: float,
                element_bit_size: Optional[int] = None):
        """
        Initialize the heavy hitters algorithm following the lecture's approach
        
        Args:
            domain_size: Size of the input domain |X|
            epsilon: Privacy parameter
            threshold: Threshold to be considered a heavy hitter
            element_bit_size: Number of bits needed to represent elements (default: log2(domain_size))
        """
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        
        # Number of bits needed to represent elements in the domain
        self.element_bit_size = element_bit_size or math.ceil(math.log2(domain_size))
        
        # Setting hash range to O(n²) as mentioned in the lecture to avoid collisions
        # Will be adjusted once we know the number of users
        self.hash_range = None
        
        # Per-bit datasets as described in the lecture
        self.bit_datasets = [[] for _ in range(self.element_bit_size)]
        
        # Store original data (for evaluation purposes)
        self.original_data = []
        
        # Frequency oracles for each bit position
        self.frequency_oracles = None
        
        # Final heavy hitters
        self.heavy_hitters = {}
        
        # Algorithm name for dashboard
        self.algorithm_name = "Lecture Bit-by-Bit"
    
    def _hash_element(self, element: Any) -> int:
        """
        Hash an element to the range [0, hash_range-1]
        
        Args:
            element: Element to hash
            
        Returns:
            Hash value
        """
        return mmh3.hash(str(element).encode(), seed=42) % self.hash_range
    
    def _get_bit(self, element: Any, bit_position: int) -> int:
        """
        Get the specific bit of an element
        
        Args:
            element: The element
            bit_position: Bit position (0 to element_bit_size-1)
            
        Returns:
            0 or 1
        """
        if isinstance(element, (int, float)):
            # Extract bit from numeric value
            element_int = int(element)
            return 1 if element_int & (1 << bit_position) else 0
        else:
            # For non-numeric types, use hash value
            hash_val = mmh3.hash(str(element).encode(), seed=bit_position)
            return hash_val % 2
    
    def add_element(self, element: Any) -> None:
        """
        Add an element to the algorithm, creating derived datasets as in the lecture
        
        Args:
            element: Element to add
        """
        self.original_data.append(element)
        
        # Get hash value
        if self.hash_range is None:
            # Initialize hash range as O(n²) for first element
            # Will be adjusted for each new element
            n = 1
            self.hash_range = n * n * 2
        else:
            n = len(self.original_data)
            # Update hash range dynamically as O(n²)
            self.hash_range = n * n * 2
        
        hash_val = self._hash_element(element)
        
        # For each bit position, create an entry in the derived dataset
        for bit_pos in range(self.element_bit_size):
            bit_val = self._get_bit(element, bit_pos)
            # Store the pair (hash_val, bit_val) in the bit-specific dataset
            self.bit_datasets[bit_pos].append((hash_val, bit_val))
    
    def add_elements(self, elements: List[Any]) -> None:
        """
        Add multiple elements
        
        Args:
            elements: List of elements to add
        """
        for element in elements:
            self.add_element(element)
    
    def _prepare_frequency_oracles(self) -> None:
        """
        Create frequency oracles for each bit position dataset
        """
        # Divide epsilon by element_bit_size for composition
        bit_epsilon = self.epsilon / self.element_bit_size
        
        self.frequency_oracles = []
        
        # Create a frequency oracle for each bit position
        for bit_pos in range(self.element_bit_size):
            oracle = LectureLDPFrequencyOracle(epsilon=bit_epsilon, hash_range=self.hash_range)
            
            # Add user data for this bit position
            for hash_val, bit_val in self.bit_datasets[bit_pos]:
                # Create a composite value that encodes both hash and bit
                oracle.add_user_data((hash_val, bit_val))
            
            self.frequency_oracles.append(oracle)
    
    def get_estimated_frequency(self, element: Any) -> float:
        """
        Get privacy-preserving estimated frequency for an element
        
        Args:
            element: Element to estimate
            
        Returns:
            Estimated frequency
        """
        if self.frequency_oracles is None:
            self._prepare_frequency_oracles()
            
        # For the dashboard compatibility, we'll use a verification oracle
        verification_oracle = LectureLDPFrequencyOracle(
            epsilon=self.epsilon, 
            hash_range=self.hash_range
        )
        
        # Add original data
        for data_element in self.original_data:
            verification_oracle.add_user_data(data_element)
            
        # Estimate frequency
        return verification_oracle.estimate_frequency(element)
    
    def identify_heavy_hitters(self, candidate_set: List[Any] = None) -> Dict[Any, float]:
        """
        Identify heavy hitters using the bit-by-bit reconstruction approach from the lecture
        
        Args:
            candidate_set: Optional list of candidates to check (not used in the lecture approach)
            
        Returns:
            Dictionary of heavy hitters and their estimated frequencies
        """
        if self.frequency_oracles is None:
            self._prepare_frequency_oracles()
        
        # Get hash values of potential heavy hitters
        potential_hh_hashes = self._identify_hash_heavy_hitters()
        
        # Reconstruct elements from hash values
        candidates = self._reconstruct_elements(potential_hh_hashes)
        
        # Verify heavy hitters from candidates
        self._verify_heavy_hitters(candidates)
        
        return self.heavy_hitters
    
    def _identify_hash_heavy_hitters(self) -> Set[int]:
        """
        Identify hash values that may correspond to heavy hitters
        
        Returns:
            Set of hash values
        """
        # Count hash values in the original data
        hash_counts = defaultdict(int)
        for element in self.original_data:
            hash_counts[self._hash_element(element)] += 1
        
        # Identify hash values that occur frequently
        potential_hash_values = set()
        n = len(self.original_data)
        for hash_val, count in hash_counts.items():
            if count / n >= self.threshold * 0.5:  # Lower threshold to catch all candidates
                potential_hash_values.add(hash_val)
        
        return potential_hash_values
    
    def _reconstruct_elements(self, hash_values: Set[int]) -> Dict[int, List[int]]:
        """
        Reconstruct potential elements from their hash values
        
        Args:
            hash_values: Set of hash values to reconstruct
            
        Returns:
            Dictionary mapping hash value to reconstructed bit patterns
        """
        candidates = {}
        
        for hash_val in hash_values:
            # For each bit position, determine the most likely bit value
            reconstructed_bits = []
            for bit_pos in range(self.element_bit_size):
                oracle = self.frequency_oracles[bit_pos]
                
                # Estimate frequencies for both possible bit values
                freq_0 = oracle.estimate_frequency((hash_val, 0))
                freq_1 = oracle.estimate_frequency((hash_val, 1))
                
                # Choose the bit value with higher estimated frequency
                bit_val = 1 if freq_1 > freq_0 else 0
                reconstructed_bits.append(bit_val)
            
            candidates[hash_val] = reconstructed_bits
            
        return candidates
    
    def _verify_heavy_hitters(self, candidates: Dict[int, List[int]]) -> None:
        """
        Verify which reconstructed candidates are actually heavy hitters
        
        Args:
            candidates: Dictionary of hash value to bit patterns
        """
        self.heavy_hitters = {}
        
        # Create frequency oracle for verification
        verification_oracle = LectureLDPFrequencyOracle(epsilon=self.epsilon, hash_range=self.hash_range)
        
        # Add original data to the verification oracle
        for element in self.original_data:
            verification_oracle.add_user_data(element)
        
        # For each candidate, convert bit pattern to an element and check frequency
        for hash_val, bit_pattern in candidates.items():
            # Convert bit pattern to integer (representative element)
            candidate_element = 0
            for bit_pos, bit_val in enumerate(bit_pattern):
                if bit_val == 1:
                    candidate_element |= (1 << bit_pos)
            
            # Estimate frequency using the verification oracle
            estimated_freq = verification_oracle.estimate_frequency(candidate_element)
            
            # Check if it's a heavy hitter
            if estimated_freq >= self.threshold:
                self.heavy_hitters[candidate_element] = estimated_freq
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the algorithm
        
        Returns:
            Dictionary of statistics
        """
        return {
            "algorithm_name": self.algorithm_name,
            "domain_size": self.domain_size,
            "privacy_parameter": self.epsilon,
            "threshold": self.threshold,
            "element_bit_size": self.element_bit_size,
            "hash_range": self.hash_range,
            "total_items": len(self.original_data),
            "num_heavy_hitters": len(self.heavy_hitters)
        }