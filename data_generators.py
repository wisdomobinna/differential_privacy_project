"""
data_generators.py
Functions to generate data distributions for testing heavy hitters algorithms
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any

# Basic statistical distributions
def generate_zipf_data(domain_size: int, num_elements: int, alpha: float = 1.07) -> List[int]:
    """Generate data following a Zipf distribution (power law)."""
    probs = np.array([1/(i**alpha) for i in range(1, domain_size + 1)])
    probs /= probs.sum()
    return np.random.choice(domain_size, size=num_elements, p=probs).tolist()

def generate_uniform_data(domain_size: int, num_elements: int) -> List[int]:
    """Generate uniformly distributed data."""
    return np.random.randint(0, domain_size, size=num_elements).tolist()

def generate_normal_data(domain_size: int, num_elements: int, 
                        mean_factor: float = 0.5, std_factor: float = 0.15) -> List[int]:
    """Generate normally distributed data centered around the domain."""
    mean = domain_size * mean_factor
    std = domain_size * std_factor
    data = np.random.normal(mean, std, num_elements)
    # Clip to domain range and convert to integers
    data = np.clip(data, 0, domain_size - 1).astype(int)
    return data.tolist()

# More realistic data distributions
def create_zipf_distributed_job_titles(num_samples: int, alpha: float = 1.5) -> List[str]:
    """Create job titles following a Zipf distribution."""
    job_titles = [
        "Software Engineer", "Data Scientist", "Manager", "Director", 
        "Analyst", "Designer", "Developer", "Coordinator", "Specialist", 
        "Consultant", "Engineer", "Technician", "Administrator", "Assistant",
        "Supervisor", "Associate", "Representative", "Architect", "Lead",
        "Intern", "VP", "CEO", "CTO", "Researcher", "Tester", "Accountant",
        "HR Specialist", "Marketing Manager", "Product Manager", "Sales Rep"
    ]
    
    # Generate Zipf probabilities
    ranks = np.arange(1, len(job_titles) + 1)
    weights = 1.0 / (ranks ** alpha)
    weights[:3] *= 3.0  # Amplify top 3
    weights /= weights.sum()
    
    # Sample job titles
    indices = np.random.choice(len(job_titles), size=num_samples, p=weights)
    
    # Add some noise
    noisy_titles = []
    for idx in indices:
        if np.random.random() < 0.05:  # 5% chance to swap
            noisy_titles.append(np.random.choice(job_titles))
        else:
            noisy_titles.append(job_titles[idx])
    
    return noisy_titles

def create_bell_curve_ages(num_samples: int, mean: int = 35, std: int = 8) -> List[int]:
    """Create ages following a bell curve distribution."""
    ages = np.random.normal(mean, std, num_samples)
    ages = np.clip(ages, 18, 70).astype(int)
    return ages.tolist()

def create_skewed_departments(num_samples: int) -> List[str]:
    """Create department names with a skewed distribution."""
    departments = [
        "Engineering", "Marketing", "Sales", "Finance", 
        "HR", "Product", "Customer Support", "Operations",
        "Legal", "Research", "Administration", "Executive"
    ]
    
    probabilities = [0.30, 0.15, 0.15, 0.10, 0.08, 0.08, 0.05, 0.04, 0.02, 0.01, 0.01, 0.01]
    indices = np.random.choice(len(departments), size=num_samples, p=probabilities)
    
    return [departments[i] for i in indices]

def create_bimodal_salary_distribution(num_samples: int) -> List[int]:
    """Create salaries following a bimodal distribution."""
    mode_1_prob = 0.7  # 70% in lower mode
    modes = np.random.choice([0, 1], size=num_samples, p=[mode_1_prob, 1-mode_1_prob])
    
    lower_mode = np.random.normal(60000, 10000, num_samples)
    higher_mode = np.random.normal(120000, 15000, num_samples)
    
    salaries = np.where(modes == 0, lower_mode, higher_mode)
    salaries = np.clip(salaries, 30000, 200000)
    salaries = np.round(salaries, -3).astype(int)
    
    return salaries.tolist()

def create_realistic_example_data(num_samples: int = 1000) -> pd.DataFrame:
    """Create a realistic example dataset with various distributions."""
    user_ids = list(range(1001, 1001 + num_samples))
    
    first_names = [
        "John", "Jane", "Michael", "Emily", "David", "Sarah", "Robert", "Lisa",
        "William", "Mary", "James", "Jennifer", "Thomas", "Elizabeth", "Daniel",
        "Jessica", "Matthew", "Susan", "Christopher", "Karen"
    ]
    first_name_indices = np.random.choice(len(first_names), size=num_samples, p=[
        0.10, 0.10, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05,
        0.04, 0.04, 0.03, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02
    ])
    first_names_data = [first_names[i] for i in first_name_indices]
    
    last_names = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
    ]
    last_names_data = [np.random.choice(last_names) for _ in range(num_samples)]
    
    sex_data = np.random.choice(["F", "M"], size=num_samples, p=[0.52, 0.48]).tolist()
    
    emails = [f"{first.lower()}.{last.lower()}@example.com" 
              for first, last in zip(first_names_data, last_names_data)]
    
    job_titles = create_zipf_distributed_job_titles(num_samples)
    ages = create_bell_curve_ages(num_samples)
    departments = create_skewed_departments(num_samples)
    salaries = create_bimodal_salary_distribution(num_samples)
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'first_name': first_names_data,
        'last_name': last_names_data,
        'sex': sex_data,
        'email': emails,
        'age': ages,
        'job_title': job_titles,
        'department': departments,
        'salary': salaries
    })
    
    return df

def convert_dataframe_to_integer_domain(df: pd.DataFrame, column: str) -> List[int]:
    """
    Convert a categorical column to integer domain for heavy hitters algorithm.
    
    Args:
        df: DataFrame with data
        column: Column name to convert
        
    Returns:
        List of integers representing the original values
    """
    unique_values = df[column].unique()
    value_to_int = {val: i for i, val in enumerate(unique_values)}
    domain_size = len(unique_values)
    
    return [value_to_int[val] for val in df[column]], domain_size, value_to_int