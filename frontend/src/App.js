import React, { useState, useEffect } from 'react';
import Dashboard from './components/Dashboard';
import AlgorithmSelector from './components/AlgorithmSelector';
import ParameterControls from './components/ParameterControls';
import DataDistributionSelector from './components/DataDistributionSelector';
import ResultsVisualization from './components/ResultsVisualization';
import MetricsDisplay from './components/MetricsDisplay';
import PrivacyImpactChart from './components/PrivacyImpactChart';
import HeavyHittersTable from './components/HeavyHittersTable';
import { fetchPrivacyImpact, fetchDistributionImpact } from './services/api';
import './App.css';

function App() {
  // Algorithm parameters
  const [params, setParams] = useState({
    algorithm_type: 'basic',
    distribution: 'zipf',
    domain_size: 100,
    num_elements: 10000,
    epsilon: 1.0,
    threshold: 0.01,
    num_hash_functions: 5
  });

  // Results state
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Additional data for charts
  const [privacyImpactData, setPrivacyImpactData] = useState([]);
  const [distributionImpactData, setDistributionImpactData] = useState([]);
  const [activeTab, setActiveTab] = useState('run'); // 'run', 'privacy', 'distribution'

  // Load comparison data on mount
  useEffect(() => {
    const loadComparisonData = async () => {
      try {
        const privacyData = await fetchPrivacyImpact();
        setPrivacyImpactData(privacyData);
        
        const distributionData = await fetchDistributionImpact();
        setDistributionImpactData(distributionData);
      } catch (err) {
        console.error("Error fetching comparison data:", err);
        setError("Failed to load comparison data. Please try again later.");
      }
    };
    
    loadComparisonData();
  }, []);

  // Update parameter handler
  const handleParamChange = (name, value) => {
    setParams(prev => ({
      ...prev,
      [name]: value
    }));
  };

  // Run algorithm handler
  const handleRunAlgorithm = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/run-algorithm', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });
      
      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }
      
      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error("Error running algorithm:", err);
      setError("Failed to run the algorithm. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Heavy Hitters Algorithm Visualization</h1>
        <p>A visual exploration of local model algorithms for heavy hitters with differential privacy</p>
      </header>
      
      <main className="app-content">
        <div className="tabs">
          <button 
            className={activeTab === 'run' ? 'active' : ''} 
            onClick={() => setActiveTab('run')}
          >
            Run Algorithm
          </button>
          <button 
            className={activeTab === 'privacy' ? 'active' : ''} 
            onClick={() => setActiveTab('privacy')}
          >
            Privacy Impact
          </button>
          <button 
            className={activeTab === 'distribution' ? 'active' : ''} 
            onClick={() => setActiveTab('distribution')}
          >
            Distribution Impact
          </button>
        </div>
        
        {activeTab === 'run' && (
          <div className="run-algorithm-container">
            <div className="controls-panel">
              <h2>Algorithm Parameters</h2>
              <AlgorithmSelector 
                value={params.algorithm_type} 
                onChange={(value) => handleParamChange('algorithm_type', value)} 
              />
              
              <DataDistributionSelector 
                value={params.distribution} 
                onChange={(value) => handleParamChange('distribution', value)} 
              />
              
              <ParameterControls 
                params={params} 
                onChange={handleParamChange} 
              />
              
              <button 
                className="run-button" 
                onClick={handleRunAlgorithm}
                disabled={loading}
              >
                {loading ? 'Running...' : 'Run Algorithm'}
              </button>
              
              {error && <div className="error-message">{error}</div>}
            </div>
            
            <div className="results-panel">
              {results ? (
                <>
                  <MetricsDisplay metrics={results.metrics} statistics={results.statistics} runtime={results.runtime} />
                  
                  <ResultsVisualization 
                    visualizationData={results.visualization} 
                    threshold={params.threshold} 
                  />
                  
                  <HeavyHittersTable 
                    visualizationData={results.visualization} 
                    numTrueHeavyHitters={results.numTrueHeavyHitters}
                    numEstimatedHeavyHitters={results.numEstimatedHeavyHitters}
                    threshold={params.threshold}
                  />
                </>
              ) : (
                <div className="no-results">
                  <p>Set parameters and click "Run Algorithm" to see results</p>
                </div>
              )}
            </div>
          </div>
        )}
        
        {activeTab === 'privacy' && (
          <div className="privacy-impact-container">
            <h2>Privacy Parameter (ε) Impact</h2>
            <p>This chart shows how the privacy parameter affects algorithm accuracy.</p>
            <PrivacyImpactChart data={privacyImpactData} />
          </div>
        )}
        
        {activeTab === 'distribution' && (
          <div className="distribution-impact-container">
            <h2>Data Distribution Impact</h2>
            <p>This chart shows how different data distributions affect algorithm performance.</p>
            <Dashboard data={distributionImpactData} />
          </div>
        )}
      </main>
      
      <footer className="app-footer">
        <p>Heavy Hitters Algorithm Visualization Tool | Differential Privacy Project</p>
      </footer>
    </div>
  );
}

export default App;