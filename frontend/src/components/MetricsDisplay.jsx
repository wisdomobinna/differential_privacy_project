import React from 'react';

/**
 * Component for displaying algorithm performance metrics
 */
const MetricsDisplay = ({ metrics, statistics, runtime }) => {
  // Format a value to show 4 decimal places if it's a number
  const formatValue = (value) => {
    if (typeof value === 'number') {
      // If value is very small, show more decimal places
      if (Math.abs(value) < 0.0001) {
        return value.toExponential(4);
      }
      return value.toFixed(4);
    }
    return value;
  };
  
  // Convert runtime to appropriate units with good formatting
  const formatRuntime = (seconds) => {
    if (seconds < 0.001) {
      return `${(seconds * 1000000).toFixed(2)} μs`;
    } else if (seconds < 1) {
      return `${(seconds * 1000).toFixed(2)} ms`;
    } else if (seconds < 60) {
      return `${seconds.toFixed(2)} seconds`;
    } else {
      const minutes = Math.floor(seconds / 60);
      const remainingSeconds = seconds % 60;
      return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
    }
  };
  
  // Get health indicators for metrics
  const getHealthIndicator = (metric, value) => {
    let healthClass = 'neutral';
    
    switch (metric) {
      case 'precision':
      case 'recall':
      case 'f1_score':
        if (value >= 0.8) healthClass = 'excellent';
        else if (value >= 0.6) healthClass = 'good';
        else if (value >= 0.4) healthClass = 'moderate';
        else healthClass = 'poor';
        break;
      case 'runtime':
        if (value < 0.1) healthClass = 'excellent';
        else if (value < 0.5) healthClass = 'good';
        else if (value < 2) healthClass = 'moderate';
        else healthClass = 'poor';
        break;
      default:
        break;
    }
    
    return healthClass;
  };
  
  if (!metrics) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg shadow-sm text-center">
        <p className="text-gray-500 text-lg">No metrics available</p>
      </div>
    );
  }
  
  return (
    <div className="metrics-display">
      <h3>Algorithm Performance</h3>
      
      <div className="metrics-grid">
        <div className="metrics-section performance-metrics">
          <h4>Performance Metrics</h4>
          
          <div className="metric-item">
            <div className="metric-label">Precision</div>
            <div className={`metric-value health-${getHealthIndicator('precision', metrics.precision)}`}>
              {formatValue(metrics.precision)}
            </div>
            <div className="metric-description">
              Fraction of reported heavy hitters that are true heavy hitters
            </div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">Recall</div>
            <div className={`metric-value health-${getHealthIndicator('recall', metrics.recall)}`}>
              {formatValue(metrics.recall)}
            </div>
            <div className="metric-description">
              Fraction of true heavy hitters that are reported
            </div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">F1 Score</div>
            <div className={`metric-value health-${getHealthIndicator('f1_score', metrics.f1_score)}`}>
              {formatValue(metrics.f1_score)}
            </div>
            <div className="metric-description">
              Harmonic mean of precision and recall
            </div>
          </div>
          
          <div className="metric-item">
            <div className="metric-label">Runtime</div>
            <div className={`metric-value health-${getHealthIndicator('runtime', runtime)}`}>
              {formatRuntime(runtime)}
            </div>
            <div className="metric-description">
              Time taken to process the data
            </div>
          </div>
        </div>
        
        <div className="metrics-section algorithm-statistics">
          <h4>Algorithm Statistics</h4>
          
          {statistics && Object.entries(statistics).map(([key, value]) => (
            <div className="metric-item" key={key}>
              <div className="metric-label">{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</div>
              <div className="metric-value">{formatValue(value)}</div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="metrics-interpretation">
        <h4>Interpretation</h4>
        <ul>
          <li>
            <strong>Precision:</strong> {
              metrics.precision >= 0.8 ? 'Excellent. Almost all reported heavy hitters are correct.' :
              metrics.precision >= 0.6 ? 'Good. Most reported heavy hitters are correct.' :
              metrics.precision >= 0.4 ? 'Moderate. Only some reported heavy hitters are correct.' :
              'Poor. Many reported heavy hitters are incorrect.'
            }
          </li>
          <li>
            <strong>Recall:</strong> {
              metrics.recall >= 0.8 ? 'Excellent. Almost all true heavy hitters are found.' :
              metrics.recall >= 0.6 ? 'Good. Most true heavy hitters are found.' :
              metrics.recall >= 0.4 ? 'Moderate. Only some true heavy hitters are found.' :
              'Poor. Many true heavy hitters are missed.'
            }
          </li>
          <li>
            <strong>F1 Score:</strong> {
              metrics.f1_score >= 0.8 ? 'Excellent overall accuracy.' :
              metrics.f1_score >= 0.6 ? 'Good overall accuracy.' :
              metrics.f1_score >= 0.4 ? 'Moderate overall accuracy.' :
              'Poor overall accuracy.'
            }
          </li>
          <li>
            <strong>Privacy Impact:</strong> Higher epsilon values provide better accuracy but weaker privacy guarantees.
          </li>
        </ul>
      </div>
    </div>
  );
};

export default MetricsDisplay;