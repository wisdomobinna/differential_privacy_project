import numpy as np
import mmh3
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math

class FrequencyOracle:
    """A modular frequency oracle implementation for LDP."""
    
    def __init__(self, epsilon: float):
        """
        Initialize the frequency oracle.
        
        Args:
            epsilon: Privacy parameter (ε > 0)
        """
        self.epsilon = epsilon
        self.p = np.exp(epsilon) / (1 + np.exp(epsilon))  # Probability of keeping true value
        self.counts = defaultdict(int)
        self.total = 0
    
    def add_element(self, element: Any) -> None:
        """
        Add an element with randomized response.
        
        Args:
            element: The element to add (must be hashable)
        """
        # Apply randomized response
        if np.random.random() < self.p:
            # Keep the true value
            privatized = element
        else:
            # For tuples, create an alternative value instead of trying to negate
            if isinstance(element, tuple):
                # For a tuple (hash, bit_pos), generate a different random hash value
                hash_val, bit_pos = element
                alternative_hash = (hash_val + 1) % 1000  # Simple alternative
                privatized = (alternative_hash, bit_pos)
            elif isinstance(element, bool):
                privatized = not element
            elif isinstance(element, (int, float)):
                privatized = 1 - element
            else:
                # For other types, use a hash function to create an alternative
                privatized = str(hash(str(element)))
        
        self.counts[privatized] += 1
        self.total += 1
    
    def estimate_frequency(self, element: Any) -> float:
        """
        Estimate the frequency of an element.
        
        Args:
            element: The element to estimate frequency for
            
        Returns:
            Estimated frequency in [0,1]
        """
        if self.total == 0:
            return 0.0
        
        # Get observed count
        observed = self.counts.get(element, 0)
        
        # Apply correction for randomized response
        corrected = (observed - self.total * (1 - self.p)) / (2 * self.p - 1)
        
        # Clip to valid range and normalize
        return max(0.0, corrected) / self.total

class BitByBitHeavyHitters:
    """
    Improved implementation of the bit-by-bit heavy hitters algorithm with:
    - Modular frequency oracles
    - Better hash handling
    - Optimized reconstruction
    - Comprehensive documentation
    """
    
    def __init__(self, 
                 domain_size: int, 
                 epsilon: float, 
                 threshold: float,
                 num_users: Optional[int] = None,
                 collision_prob: float = 0.01):
        """
        Initialize the heavy hitters algorithm.
        
        Args:
            domain_size: Number of possible unique elements
            epsilon: Privacy parameter (ε > 0)
            threshold: Fraction threshold for heavy hitters (0 < threshold < 1)
            num_users: Expected number of users (for optimizing hash range)
            collision_prob: Allowed probability of hash collisions
        """
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < threshold < 1):
            raise ValueError("threshold must be between 0 and 1")
        
        self.domain_size = domain_size
        self.epsilon = epsilon
        self.threshold = threshold
        self.collision_prob = collision_prob
        
        # Calculate bit length needed to represent elements
        self.bit_length = max(1, math.ceil(math.log2(domain_size)))
        
        # Initialize frequency oracles for each bit position
        self.bit_oracles = [FrequencyOracle(epsilon) for _ in range(self.bit_length)]
        
        # Hash function parameters
        self.hash_seed = 42  # Could be made configurable
        self.hash_range = self._calculate_hash_range(num_users) if num_users else 1000
        
        # Tracking
        self.total_users = 0
        self.unique_elements = set()
        
        # Results
        self.heavy_hitters = {}
        
        # Algorithm metadata
        self.algorithm_name = "Improved Bit-by-Bit Heavy Hitters"
    
    def _calculate_hash_range(self, n: int) -> int:
        """Calculate appropriate hash range to minimize collisions."""
        # Using birthday problem approximation: p_collision ≈ n²/(2T)
        # Solving for T: T ≈ n²/(2*p_collision)
        return math.ceil((n ** 2) / (2 * self.collision_prob))
    
    def _element_to_bits(self, element: Any) -> Tuple[int, List[int]]:
        """
        Convert an element to its binary representation.
        
        Args:
            element: Input element
            
        Returns:
            Tuple of (hashed_value, list_of_bits)
        """
        # Convert element to consistent integer
        try:
            element_int = int(element)
        except (ValueError, TypeError):
            element_int = mmh3.hash(str(element), self.hash_seed) % self.domain_size
        
        # Store original element
        self.unique_elements.add(element_int)
        
        # Get binary representation
        binary_str = format(element_int, f'0{self.bit_length}b')
        bits = [int(b) for b in binary_str]
        
        # Hash the element
        hashed = mmh3.hash(str(element_int), self.hash_seed) % self.hash_range
        
        return hashed, bits
    
    def add_element(self, element: Any) -> None:
        """Add a single element to the dataset."""
        hashed, bits = self._element_to_bits(element)
        
        # Add each bit to corresponding frequency oracle
        for bit_pos, bit in enumerate(bits):
            # Create unique identifier combining hash and bit position
            identifier = (hashed, bit_pos)
            self.bit_oracles[bit_pos].add_element(identifier)
        
        self.total_users += 1
    
    def add_elements(self, elements: List[Any]) -> None:
        """Add multiple elements to the dataset."""
        for element in elements:
            self.add_element(element)
    
    def _reconstruct_element(self, hashed_value: int) -> Tuple[int, float]:
        """
        Reconstruct an element from its hashed value.
        
        Args:
            hashed_value: The hash to reconstruct
            
        Returns:
            Tuple of (reconstructed_value, confidence_score)
        """
        reconstructed_bits = []
        min_confidence = 1.0  # Track minimum bit confidence
        
        for bit_pos in range(self.bit_length):
            # Create identifiers for both possible bit values
            id_0 = (hashed_value, bit_pos)
            
            # Get frequency estimate for this bit position
            freq = self.bit_oracles[bit_pos].estimate_frequency(id_0)
            
            # Determine bit value and confidence
            # If freq > 0.5, bit is likely 1, otherwise 0
            # Confidence is how far from 0.5 in either direction
            if freq > 0.5:
                bit = 1
                confidence = freq - 0.5
            else:
                bit = 0
                confidence = 0.5 - freq
            
            reconstructed_bits.append(str(bit))
            min_confidence = min(min_confidence, confidence)
        
        # Convert bits to integer
        reconstructed_int = int(''.join(reconstructed_bits), 2)
        
        # Return both the value and our confidence in it
        return reconstructed_int, min_confidence * 2  # Scale confidence to [0,1]
    
    def identify_heavy_hitters(self, candidate_set: List[Any] = None) -> Dict[Any, float]:
        """
        Identify heavy hitters in the dataset.
        
        Args:
            candidate_set: Optional list of candidate elements to check
            
        Returns:
            Dictionary of heavy hitters with their estimated frequencies
        """
        self.heavy_hitters = {}
        
        if candidate_set is not None:
            # Check frequencies of provided candidates
            for element in candidate_set:
                freq = self.get_estimated_frequency(element)
                if freq >= self.threshold:
                    self.heavy_hitters[element] = freq
        else:
            # Generate candidate hashes from seen elements
            candidate_hashes = set()
            for element in self.unique_elements:
                hashed = mmh3.hash(str(element), self.hash_seed) % self.hash_range
                candidate_hashes.add(hashed)
            
            # Also consider some random hashes to catch potentially missed elements
            num_random_checks = min(100, self.hash_range - len(candidate_hashes))
            for _ in range(num_random_checks):
                candidate_hashes.add(np.random.randint(0, self.hash_range))
            
            # Evaluate each candidate
            for hashed in candidate_hashes:
                value, confidence = self._reconstruct_element(hashed)
                
                # Only consider values within our domain
                if 0 <= value < self.domain_size:
                    # Use confidence as a proxy for frequency
                    estimated_freq = confidence
                    if estimated_freq >= self.threshold:
                        self.heavy_hitters[value] = estimated_freq
        
        return self.heavy_hitters
    
    def get_estimated_frequency(self, element: Any) -> float:
        """Estimate frequency of a specific element."""
        hashed, bits = self._element_to_bits(element)
        confidences = []
        
        for bit_pos, expected_bit in enumerate(bits):
            identifier = (hashed, bit_pos)
            freq = self.bit_oracles[bit_pos].estimate_frequency(identifier)
            
            # Calculate confidence based on expected bit value
            if expected_bit == 1:
                confidence = freq
            else:
                confidence = 1 - freq
                
            confidences.append(confidence)
        
        # Return minimum confidence as frequency estimate
        return min(confidences) if confidences else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current state."""
        return {
            'algorithm': self.algorithm_name,
            'total_users': self.total_users,
            'unique_elements': len(self.unique_elements),
            'domain_size': self.domain_size,
            'bit_length': self.bit_length,
            'hash_range': self.hash_range,
            'epsilon': self.epsilon,
            'threshold': self.threshold,
            'heavy_hitters_count': len(self.heavy_hitters),
            'estimated_precision': self._estimate_precision(),
            'estimated_recall': self._estimate_recall()
        }
    
    def _estimate_precision(self) -> float:
        """Estimate precision of heavy hitters (for evaluation)."""
        if not self.heavy_hitters:
            return 0.0
        
        # Count how many heavy hitters are actually frequent
        true_positives = 0
        for element in self.heavy_hitters:
            if self.get_estimated_frequency(element) >= self.threshold:
                true_positives += 1
        
        return true_positives / len(self.heavy_hitters)
    
    def _estimate_recall(self) -> float:
        """Estimate recall of heavy hitters (for evaluation)."""
        if not self.unique_elements:
            return 0.0
        
        # Count how many truly frequent elements were detected
        true_frequent = 0
        detected_frequent = 0
        
        for element in self.unique_elements:
            freq = sum(1 for x in self.unique_elements if x == element) / self.total_users
            if freq >= self.threshold:
                true_frequent += 1
                if element in self.heavy_hitters:
                    detected_frequent += 1
        
        return detected_frequent / true_frequent if true_frequent > 0 else 0.0