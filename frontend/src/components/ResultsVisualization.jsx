import React, { useRef, useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, ResponsiveContainer } from 'recharts';

/**
 * Component for visualizing the heavy hitters algorithm results
 */
const ResultsVisualization = ({ visualizationData, threshold }) => {
  const [activeTab, setActiveTab] = useState('chart');
  const [chartData, setChartData] = useState([]);
  const [filteredData, setFilteredData] = useState([]);
  const [showTooltip, setShowTooltip] = useState(false);
  const [tooltipData, setTooltipData] = useState(null);
  const chartRef = useRef(null);
  
  // Process visualization data for the chart
  useEffect(() => {
    if (visualizationData && visualizationData.length > 0) {
      // Limit to top 20 elements for better visualization
      const limitedData = visualizationData.slice(0, 20);
      
      // Transform data for the chart
      const formattedData = limitedData.map(item => ({
        element: item.element,
        trueFrequency: parseFloat(item.trueFrequency.toFixed(4)),
        estimatedFrequency: parseFloat(item.estimatedFrequency.toFixed(4)),
        isHeavyHitter: item.trueFrequency > threshold
      }));
      
      setChartData(formattedData);
      
      // Create filtered views
      const trueHeavyHitters = formattedData.filter(item => item.trueFrequency > threshold);
      setFilteredData(trueHeavyHitters);
    }
  }, [visualizationData, threshold]);
  
  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip">
          <p className="tooltip-label">{`Element: ${label}`}</p>
          <p className="tooltip-true">
            <span className="tooltip-dot true-dot"></span>
            {`True Frequency: ${payload[0].value.toFixed(4)}`}
          </p>
          <p className="tooltip-estimated">
            <span className="tooltip-dot estimated-dot"></span>
            {`Estimated Frequency: ${payload[1].value.toFixed(4)}`}
          </p>
          <p className="tooltip-error">
            {`Error: ${Math.abs(payload[0].value - payload[1].value).toFixed(4)}`}
          </p>
        </div>
      );
    }
    
    return null;
  };
  
  // If no data, show placeholder
  if (!visualizationData || visualizationData.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg shadow-sm text-center">
        <p className="text-gray-500 text-lg">No visualization data available</p>
      </div>
    );
  }
  
  return (
    <div className="results-visualization">
      <h3>Heavy Hitters Visualization</h3>
      
      <div className="visualization-tabs">
        <button 
          className={activeTab === 'chart' ? 'active' : ''}
          onClick={() => setActiveTab('chart')}
        >
          Frequency Chart
        </button>
        <button 
          className={activeTab === 'heatmap' ? 'active' : ''}
          onClick={() => setActiveTab('heatmap')}
        >
          Error Heatmap
        </button>
        <button 
          className={activeTab === 'comparison' ? 'active' : ''}
          onClick={() => setActiveTab('comparison')}
        >
          True vs. Estimated
        </button>
      </div>
      
      {activeTab === 'chart' && (
        <div className="chart-container">
          <div className="chart-controls">
            <button onClick={() => setChartData(visualizationData.slice(0, 20))}>
              All Elements
            </button>
            <button onClick={() => setChartData(filteredData)}>
              True Heavy Hitters Only
            </button>
          </div>
          
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={chartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              ref={chartRef}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="element" 
                label={{ value: 'Element', position: 'insideBottom', offset: -10 }}
                angle={-45}
                textAnchor="end"
              />
              <YAxis
                label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend verticalAlign="top" />
              <ReferenceLine y={threshold} stroke="red" strokeDasharray="3 3" label="Threshold" />
              <Bar 
                dataKey="trueFrequency" 
                name="True Frequency" 
                fill="#8884d8" 
                maxBarSize={50}
              />
              <Bar 
                dataKey="estimatedFrequency" 
                name="Estimated Frequency" 
                fill="#82ca9d" 
                maxBarSize={50}
              />
            </BarChart>
          </ResponsiveContainer>
          
          <div className="chart-legend">
            <div className="legend-item">
              <div className="legend-color true-color"></div>
              <span>True Frequency</span>
            </div>
            <div className="legend-item">
              <div className="legend-color estimated-color"></div>
              <span>Estimated Frequency</span>
            </div>
            <div className="legend-item">
              <div className="legend-color threshold-color"></div>
              <span>Threshold ({threshold.toFixed(4)})</span>
            </div>
          </div>
        </div>
      )}
      
      {activeTab === 'heatmap' && (
        <div className="heatmap-container">
          <h4>Error Heatmap</h4>
          <p>Visual representation of estimation errors across elements</p>
          
          <div className="heatmap">
            {chartData.map((item, index) => {
              const error = Math.abs(item.trueFrequency - item.estimatedFrequency);
              const errorPercentage = error / item.trueFrequency;
              const colorIntensity = Math.min(255, Math.floor(errorPercentage * 1000));
              const color = `rgb(${colorIntensity}, ${Math.max(0, 255 - colorIntensity)}, 0)`;
              
              return (
                <div 
                  key={index}
                  className="heatmap-cell"
                  style={{ 
                    backgroundColor: color,
                    width: `${100 / Math.min(20, chartData.length)}%`
                  }}
                  onMouseEnter={() => {
                    setTooltipData({
                      element: item.element,
                      trueFreq: item.trueFrequency,
                      estimatedFreq: item.estimatedFrequency,
                      error: error,
                      errorPercentage: errorPercentage * 100
                    });
                    setShowTooltip(true);
                  }}
                  onMouseLeave={() => {
                    setShowTooltip(false);
                  }}
                >
                  {item.element}
                </div>
              );
            })}
          </div>
          
          {showTooltip && tooltipData && (
            <div className="heatmap-tooltip">
              <p><strong>Element:</strong> {tooltipData.element}</p>
              <p><strong>True Frequency:</strong> {tooltipData.trueFreq.toFixed(4)}</p>
              <p><strong>Estimated Frequency:</strong> {tooltipData.estimatedFreq.toFixed(4)}</p>
              <p><strong>Absolute Error:</strong> {tooltipData.error.toFixed(4)}</p>
              <p><strong>Error Percentage:</strong> {tooltipData.errorPercentage.toFixed(2)}%</p>
            </div>
          )}
          
          <div className="heatmap-legend">
            <div className="heatmap-gradient"></div>
            <div className="heatmap-labels">
              <span>Low Error</span>
              <span>High Error</span>
            </div>
          </div>
        </div>
      )}
      
      {activeTab === 'comparison' && (
        <div className="comparison-container">
          <div className="scatter-container">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="trueFrequency"
                  label={{ value: 'True Frequency', position: 'bottom', offset: 0 }}
                />
                <YAxis
                  dataKey="estimatedFrequency"
                  label={{ value: 'Estimated Frequency', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <ReferenceLine y={threshold} stroke="red" strokeDasharray="3 3" />
                <ReferenceLine x={threshold} stroke="red" strokeDasharray="3 3" />
                <ReferenceLine y="x" stroke="blue" strokeDasharray="3 3" label="Perfect Estimation" />
                <Bar 
                  dataKey="estimatedFrequency" 
                  name="Elements" 
                  fill="#82ca9d" 
                  maxBarSize={10}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          
          <div className="comparison-explanation">
            <h4>Interpretation</h4>
            <ul>
              <li>Points on the blue line represent perfect estimation (estimated = true)</li>
              <li>Points above the line indicate overestimation</li>
              <li>Points below the line indicate underestimation</li>
              <li>Red lines show the threshold value</li>
              <li>The upper-right quadrant contains correctly identified heavy hitters</li>
              <li>The upper-left quadrant contains false positives</li>
              <li>The lower-right quadrant contains false negatives</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default ResultsVisualization;