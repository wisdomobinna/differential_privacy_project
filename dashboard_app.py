"""
dashboard_app.py
Interactive dashboard for comparing Heavy Hitters algorithms
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Dict, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import our algorithms
from heavy_hitters_implementation import (
    LocalModelHeavyHitters, 
    OLHHeavyHitters,
    RAPPORHeavyHitters
)
from bit_by_bit_heavyhitters import BitByBitHeavyHitters
from data_generators import (
    generate_zipf_data, 
    generate_uniform_data, 
    generate_normal_data,
    create_realistic_example_data,
    convert_dataframe_to_integer_domain
)

# Set page configuration
st.set_page_config(
    page_title="Heavy Hitters Algorithm Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS for better styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 0.15rem 0.3rem rgba(0,0,0,0.1);
    }
    .algorithm-section {
        border-left: 3px solid #4e8df5;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    h1, h2, h3 {
        margin-bottom: 0.5rem;
    }
    .stPlotlyChart {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def calculate_true_frequencies(data: List[int]) -> Dict[int, float]:
    """Calculate true frequencies of elements in the data."""
    counts = {}
    for element in data:
        counts[element] = counts.get(element, 0) + 1
    
    total = len(data)
    frequencies = {element: count/total for element, count in counts.items()}
    
    return frequencies

def get_true_heavy_hitters(data: List[int], threshold: float) -> Dict[int, float]:
    """Get elements that exceed the threshold (true heavy hitters)."""
    freqs = calculate_true_frequencies(data)
    return {elem: freq for elem, freq in freqs.items() if freq > threshold}

def calculate_metrics(true_hh: Dict[int, float], 
                     estimated_hh: Dict[int, float], 
                     all_elements: List[int]) -> Dict[str, float]:
    """Calculate performance metrics for the algorithm."""
    true_set = set(true_hh.keys())
    est_set = set(estimated_hh.keys())
    
    # Handle edge cases
    if not true_set and not est_set:
        precision = 1.0
        recall = 1.0
    elif not est_set:
        precision = 0.0
        recall = 0.0
    elif not true_set:
        precision = 0.0
        recall = 1.0
    else:
        precision = len(true_set.intersection(est_set)) / len(est_set)
        recall = len(true_set.intersection(est_set)) / len(true_set)
    
    # Calculate F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate average frequency error on all elements
    true_freqs = calculate_true_frequencies(all_elements)
    
    # Candidates for error calculation (true or estimated heavy hitters)
    candidates = true_set.union(est_set)
    error_sum = 0
    
    for element in candidates:
        true_freq = true_freqs.get(element, 0)
        est_freq = estimated_hh.get(element, 0)
        error_sum += abs(true_freq - est_freq)
    
    avg_error = error_sum / len(candidates) if candidates else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_error': avg_error,
        'num_true_hh': len(true_set),
        'num_estimated_hh': len(est_set)
    }

def run_algorithm(algorithm_type: str,
                 data: List[int],
                 domain_size: int,
                 epsilon: float,
                 threshold: float) -> Tuple[Dict[int, float], Dict[str, Any], Dict[str, float], float]:
    """Run the specified algorithm and return results."""
    start_time = time.time()
    
    # Initialize algorithm
    if algorithm_type == "Randomized Response":
        algorithm = LocalModelHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold
        )
    elif algorithm_type == "OLH":
        algorithm = OLHHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold
        )
    elif algorithm_type == "RAPPOR":
        algorithm = RAPPORHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold,
            num_hash_functions=5
        )
    elif algorithm_type == "Bit-by-Bit":  # Add new algorithm type
        algorithm = BitByBitHeavyHitters(
            domain_size=domain_size,
            epsilon=epsilon,
            threshold=threshold
        )
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    # Add data
    algorithm.add_elements(data)
    
    # Get true heavy hitters for comparison
    true_heavy_hitters = get_true_heavy_hitters(data, threshold)
    
    # Identify estimated heavy hitters
    candidate_set = list(range(domain_size))
    estimated_heavy_hitters = algorithm.identify_heavy_hitters(candidate_set)
    
    # Calculate metrics
    metrics = calculate_metrics(true_heavy_hitters, estimated_heavy_hitters, data)
    
    # Get algorithm statistics
    stats = algorithm.get_statistics()
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return estimated_heavy_hitters, stats, metrics, runtime

def compare_frequencies(data: List[int], 
                       results: Dict[str, Dict[int, float]],
                       threshold: float,
                       top_n: int = 20,
                       value_map: Dict[int, Any] = None) -> go.Figure:
    """Create a comparison plot of true vs estimated frequencies."""
    # Get true frequencies
    true_freqs = calculate_true_frequencies(data)
    
    # Find elements to display (union of heavy hitters from all algorithms)
    all_elements = set()
    for algo_name, hh_dict in results.items():
        all_elements.update(hh_dict.keys())
    
    # Also include true heavy hitters that might have been missed
    for elem, freq in true_freqs.items():
        if freq > threshold:
            all_elements.add(elem)
    
    # Sort elements by true frequency and take top N
    top_elements = sorted(all_elements, key=lambda x: true_freqs.get(x, 0), reverse=True)[:top_n]
    
    # Create dataframe for plotting
    plot_data = []
    
    # Add true frequencies
    for elem in top_elements:
        label = value_map.get(elem, elem) if value_map else elem
        plot_data.append({
            'Element': str(label),
            'Frequency': true_freqs.get(elem, 0),
            'Algorithm': 'True Frequency'
        })
    
    # Add estimated frequencies from each algorithm
    for algo_name, hh_dict in results.items():
        for elem in top_elements:
            label = value_map.get(elem, elem) if value_map else elem
            plot_data.append({
                'Element': str(label),
                'Frequency': hh_dict.get(elem, 0),
                'Algorithm': algo_name
            })
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    fig = px.bar(
        df, 
        x='Element', 
        y='Frequency', 
        color='Algorithm',
        barmode='group',
        title=f'Comparison of Top {len(top_elements)} Elements by Frequency',
        labels={'Element': 'Element', 'Frequency': 'Frequency (%)'},
        height=500
    )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=threshold,
        x1=len(top_elements)-0.5,
        y1=threshold,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=len(top_elements)//2,
        y=threshold*1.1,
        text=f"Threshold: {threshold:.4f}",
        showarrow=False,
        font=dict(color="red")
    )
    
    # Improve layout
    fig.update_layout(
        xaxis=dict(tickmode='linear'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create a radar chart comparing algorithm metrics."""
    # Create a list of algorithms and metrics to compare
    algorithms = list(metrics_dict.keys())
    metrics_to_plot = ['precision', 'recall', 'f1_score']
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each algorithm
    for algo in algorithms:
        fig.add_trace(go.Scatterpolar(
            r=[metrics_dict[algo][metric] for metric in metrics_to_plot],
            theta=metrics_to_plot,
            fill='toself',
            name=algo
        ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Algorithm Performance Comparison",
        height=400
    )
    
    return fig

def plot_error_vs_privacy(domain_size: int, 
                         data_distribution: str, 
                         num_elements: int, 
                         threshold: float,
                         algorithms: List[str]) -> go.Figure:
    """Create a plot showing error vs privacy parameter for each algorithm."""
    # Generate data
    if data_distribution == "zipf":
        data = generate_zipf_data(domain_size, num_elements)
    elif data_distribution == "uniform":
        data = generate_uniform_data(domain_size, num_elements)
    else:  # normal
        data = generate_normal_data(domain_size, num_elements)
    
    # Privacy parameters to test
    epsilons = [0.1, 0.5, 1.0, 2.0, 4.0]
    
    # Results storage
    results = {algo: {'avg_error': [], 'f1_score': []} for algo in algorithms}
    
    # Run algorithms with different epsilon values
    for eps in epsilons:
        for algo_type in algorithms:
            # Run algorithm with this epsilon
            _, _, metrics, _ = run_algorithm(
                algorithm_type=algo_type,
                data=data,
                domain_size=domain_size,
                epsilon=eps,
                threshold=threshold
            )
            
            # Store results
            results[algo_type]['avg_error'].append(metrics['avg_error'])
            results[algo_type]['f1_score'].append(metrics['f1_score'])
    
    # Create subplot with two y-axes
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=("Average Error vs Privacy Parameter", 
                                        "F1 Score vs Privacy Parameter"))
    
    # Add traces for average error
    for algo_type, data_dict in results.items():
        fig.add_trace(
            go.Scatter(
                x=epsilons, 
                y=data_dict['avg_error'],
                mode='lines+markers',
                name=f"{algo_type} (Error)",
                line=dict(width=2)
            ),
            row=1, col=1
        )
    
    # Add traces for F1 score
    for algo_type, data_dict in results.items():
        fig.add_trace(
            go.Scatter(
                x=epsilons, 
                y=data_dict['f1_score'],
                mode='lines+markers',
                name=f"{algo_type} (F1)",
                line=dict(width=2, dash='dot')
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Privacy-Utility Tradeoff",
        xaxis_title="Privacy Parameter (ε)",
        yaxis_title="Average Error",
        xaxis2_title="Privacy Parameter (ε)",
        yaxis2_title="F1 Score",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500
    )
    
    # Update x-axis scale to log scale
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_xaxes(type="log", row=1, col=2)
    
    return fig

def main():
    """Main dashboard function."""
    # App title
    st.title("Heavy Hitters Algorithm Dashboard")
    st.markdown("Compare Local Differential Privacy algorithms for identifying heavy hitters")
    
    # Sidebar for parameters
    st.sidebar.header("Parameters")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Synthetic Distributions", "Realistic Dataset"],
        help="Choose between simple statistical distributions or realistic data patterns"
    )
    
    # Data parameters
    if data_source == "Synthetic Distributions":
        st.sidebar.subheader("Synthetic Data Generation")
        distribution = st.sidebar.selectbox(
            "Data Distribution", 
            ["zipf", "uniform", "normal"],
            help="Zipf follows a power law common in many real-world scenarios"
        )
        
        domain_size = st.sidebar.slider(
            "Domain Size", 
            min_value=100, 
            max_value=10000, 
            value=1000,
            step=100,
            help="Number of possible unique elements"
        )
        
        num_elements = st.sidebar.slider(
            "Dataset Size", 
            min_value=1000, 
            max_value=100000, 
            value=10000,
            step=1000,
            help="Total number of data points"
        )
    else:  # Realistic Dataset
        st.sidebar.subheader("Realistic Data Generation")
        num_samples = st.sidebar.slider(
            "Number of Employees", 
            min_value=100, 
            max_value=10000, 
            value=1000,
            step=100,
            help="Number of employees in the dataset"
        )
        
        column_to_analyze = st.sidebar.selectbox(
            "Column to Analyze",
            ["job_title", "department", "first_name", "age"],
            help="Select which column to identify heavy hitters in"
        )
    
    # Algorithm parameters
    st.sidebar.subheader("Algorithm Settings")
    epsilon = st.sidebar.slider(
        "Privacy Parameter (ε)", 
        min_value=0.1, 
        max_value=5.0, 
        value=1.0,
        step=0.1,
        help="Higher values mean less privacy but better utility"
    )
    
    threshold = st.sidebar.slider(
        "Frequency Threshold", 
        min_value=0.001, 
        max_value=0.1, 
        value=0.01,
        format="%.3f",
        help="Minimum frequency required to be considered a heavy hitter"
    )
    
    # Algorithm selection
    algorithms = st.sidebar.multiselect(
        "Algorithms to Compare",
        ["Randomized Response", "OLH", "RAPPOR", "Bit-by-Bit"],
        default=["Randomized Response", "OLH"],
        help="Select at least one algorithm"
    )
    
    if not algorithms:
        st.warning("Please select at least one algorithm to run.")
        return
    
    # Run button
    run_button = st.sidebar.button("Run Comparison", type="primary")
    
    # About section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        This dashboard compares different local differential privacy algorithms 
        for identifying heavy hitters (frequently occurring elements) in a dataset.
        
        **Algorithms:**
        - **Randomized Response**: Basic local model algorithm using randomized response
        - **OLH (Optimal Local Hashing)**: Enhanced algorithm with optimized hash size
        - **RAPPOR**: Uses Bloom filters with randomization for better privacy
        - **Bit-by-Bit**: Lecture algorithm that reconstructs elements bit by bit
        """
    )
    
    # Main content area
    if run_button:
        with st.spinner("Generating data and running algorithms..."):
            # Generate data based on selection
            value_map = None
            
            if data_source == "Synthetic Distributions":
                # Generate data based on selected distribution
                if distribution == "zipf":
                    data = generate_zipf_data(domain_size, num_elements)
                    st.write("✅ Generated Zipf-distributed data")
                elif distribution == "uniform":
                    data = generate_uniform_data(domain_size, num_elements)
                    st.write("✅ Generated Uniformly-distributed data")
                else:  # normal
                    data = generate_normal_data(domain_size, num_elements)
                    st.write("✅ Generated Normally-distributed data")
            else:  # Realistic Dataset
                # Generate realistic dataset
                df = create_realistic_example_data(num_samples)
                st.write(f"✅ Generated realistic dataset with {num_samples} employees")
                
                # Display sample of the data
                st.write("Sample of generated data:")
                st.dataframe(df.head())
                
                # Convert to integer domain for algorithm processing
                data, domain_size, value_map = convert_dataframe_to_integer_domain(df, column_to_analyze)
                st.write(f"✅ Analyzing '{column_to_analyze}' column (domain size: {domain_size})")
            
            # Get true frequencies and heavy hitters
            true_freqs = calculate_true_frequencies(data)
            true_heavy_hitters = get_true_heavy_hitters(data, threshold)
            
            # Show data distribution info
            st.subheader("Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Data Points", len(data))
            
            with col2:
                st.metric("Unique Elements", len(true_freqs))
            
            with col3:
                st.metric("True Heavy Hitters", len(true_heavy_hitters))
            
            # Visualize data distribution
            top_elements = sorted(true_freqs.items(), key=lambda x: x[1], reverse=True)[:50]
            
            # Map back to original values if using realistic data
            if value_map:
                top_elements = [(value_map.get(elem, elem), freq) for elem, freq in top_elements]
                
            df_dist = pd.DataFrame(top_elements, columns=['Element', 'Frequency'])
            
            fig_dist = px.bar(
                df_dist, 
                x='Element', 
                y='Frequency',
                title=f'Top 50 Most Frequent Elements (Threshold: {threshold:.4f})',
                labels={'Element': 'Element', 'Frequency': 'True Frequency'}
            )
            
            fig_dist.add_shape(
                type="line",
                x0=-0.5,
                y0=threshold,
                x1=49.5,
                y1=threshold,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Results dictionary to store heavy hitters from each algorithm
            heavy_hitters_results = {}
            metrics_results = {}
            all_stats = {}
            runtimes = {}
            
            # Run each selected algorithm
            progress_bar = st.progress(0)
            
            for i, algo_type in enumerate(algorithms):
                st.markdown(f"### Running {algo_type} Algorithm")
                
                # Run the algorithm
                estimated_hh, stats, metrics, runtime = run_algorithm(
                    algorithm_type=algo_type,
                    data=data,
                    domain_size=domain_size,
                    epsilon=epsilon,
                    threshold=threshold
                )
                
                # Store results
                heavy_hitters_results[algo_type] = estimated_hh
                metrics_results[algo_type] = metrics
                all_stats[algo_type] = stats
                runtimes[algo_type] = runtime
                
                # Display metrics for this algorithm
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Precision", f"{metrics['precision']:.4f}")
                
                with col2:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                
                with col3:
                    st.metric("F1 Score", f"{metrics['f1_score']:.4f}")
                
                with col4:
                    st.metric("Average Error", f"{metrics['avg_error']:.4f}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(algorithms))
            
            # Clear progress bar
            progress_bar.empty()
            
            # Comparison visualizations
            st.subheader("Algorithm Comparison")
            
            # Compare metrics
            fig_metrics = plot_metrics_comparison(metrics_results)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Compare frequencies
            fig_freqs = compare_frequencies(data, heavy_hitters_results, threshold, value_map=value_map)
            st.plotly_chart(fig_freqs, use_container_width=True)
            
            # Privacy-utility tradeoff visualization
            st.subheader("Privacy-Utility Tradeoff")
            st.markdown("""
            This chart shows how the performance of each algorithm changes with different
            privacy parameter (ε) values. Lower ε means stronger privacy but typically
            worse utility (higher error, lower F1 score).
            """)
            
            # Create the tradeoff visualization with a button to refresh
            if st.button("Generate Privacy-Utility Tradeoff Chart"):
                with st.spinner("Running algorithms with different privacy parameters..."):
                    if data_source == "Synthetic Distributions":
                        fig_tradeoff = plot_error_vs_privacy(
                            domain_size=domain_size,
                            data_distribution=distribution,
                            num_elements=num_elements,
                            threshold=threshold,
                            algorithms=algorithms
                        )
                    else:
                        # Use the same data for privacy-utility tradeoff
                        fig_tradeoff = plot_error_vs_privacy(
                            domain_size=domain_size,
                            data_distribution="custom",  # Not used but needed for parameter
                            num_elements=len(data),
                            threshold=threshold,
                            algorithms=algorithms
                        )
                    st.plotly_chart(fig_tradeoff, use_container_width=True)
            
            # Detailed statistics table
            st.subheader("Algorithm Statistics")
            
            # Create a dataframe for algorithm stats
            stats_data = []
            for algo_type, stats in all_stats.items():
                row = {'Algorithm': algo_type, 'Runtime (s)': runtimes[algo_type]}
                for key, value in stats.items():
                    if key != 'algorithm_name':  # Skip redundant name
                        row[key] = value
                stats_data.append(row)
            
            df_stats = pd.DataFrame(stats_data)
            st.dataframe(df_stats, use_container_width=True)
            
            # Success message
            st.success("Analysis complete! You can adjust parameters and run again.")
    
    else:
        # Initial information when dashboard loads
        st.info("Set parameters in the sidebar and click 'Run Comparison' to start")
        
        # Add explanation about the algorithms
        st.markdown("## About the Algorithms")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Randomized Response")
            st.markdown("""
            The basic local model algorithm uses randomized response to achieve 
            differential privacy:
            
            1. With probability p = e^ε / (1 + e^ε), report true hash
            2. With probability 1-p, report random hash
            3. Correct estimations for the added noise
            
            **Best for**: Simpler scenarios with moderate accuracy requirements
            """)
        
        with col2:
            st.markdown("### Optimal Local Hashing (OLH)")
            st.markdown("""
            OLH optimizes the hash range based on the privacy parameter:
            
            1. Uses optimal hash range g = e^ε + 1
            2. Each client gets a consistent hash function
            3. More efficient estimation with optimized parameters
            
            **Best for**: Better utility with the same privacy guarantees
            """)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("### RAPPOR")
            st.markdown("""
            Uses Bloom filters with permanent and instantaneous randomization:
            
            1. Maps elements to Bloom filter bits
            2. Applies two-phase randomization for stronger privacy
            3. Estimates frequencies through matrix operations
            
            **Best for**: Applications requiring strong privacy guarantees
            """)
            
        with col4:
            st.markdown("### Bit-by-Bit Heavy Hitters")
            st.markdown("""
            The bit-by-bit algorithm from the lecture reconstructs heavy hitters one bit at a time:
            
            1. Uses O(n²) hash range to avoid collisions with high probability
            2. Maintains frequency oracles for each bit position (derived datasets)
            3. Reconstructs potential heavy hitters by comparing bit frequencies
            4. Exact match to the algorithm presented in the lecture
            
            **Best for**: Following the theoretical approach with strong collision avoidance
            """)
        
        # Privacy-utility explanation
        st.markdown("## Privacy-Utility Tradeoff")
        st.markdown("""
        Local differential privacy provides strong privacy guarantees but introduces
        a fundamental tradeoff with utility:
        
        - **Lower ε**: Stronger privacy, but worse accuracy
        - **Higher ε**: Better accuracy, but weaker privacy
        
        The dashboard helps visualize this tradeoff and compare algorithm performance
        across different parameters.
        """)
        
        # Realistic data explanation
        st.markdown("## Realistic Data Option")
        st.markdown("""
        The dashboard allows you to test algorithms on realistic data patterns typical in real-world scenarios:
        
        - **Job Titles**: Follow a Zipf distribution (power law) where a few titles dominate
        - **Departments**: Highly skewed distribution with Engineering/Sales/Marketing being much larger
        - **Ages**: Bell curve (normal) distribution centered around 35
        - **Salaries**: Bimodal distribution with peaks at junior and senior levels
        
        These realistic patterns allow more meaningful comparison of algorithm performance
        compared to pure statistical distributions.
        """)

if __name__ == "__main__":
    main()