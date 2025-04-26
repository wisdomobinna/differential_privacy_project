import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

/**
 * Component for visualizing the impact of privacy parameter on algorithm performance
 */
const PrivacyImpactChart = ({ data }) => {
  const [metric, setMetric] = useState('f1_score');
  
  if (!data || data.length === 0) {
    return (
      <div className="bg-gray-50 p-6 rounded-lg shadow-sm text-center">
        <p className="text-gray-500 text-lg">No privacy impact data available</p>
      </div>
    );
  }
  
  // Format data for the chart
  const chartData = data.map(item => ({
    epsilon: item.epsilon,
    basic: item.basic[metric],
    cms: item.cms[metric]
  }));
  
  const metricLabels = {
    precision: 'Precision',
    recall: 'Recall',
    f1_score: 'F1 Score'
  };
  
  return (
    <div className="privacy-impact-chart">
      <div className="chart-controls">
        <div className="metric-selector">
          <label>Metric:</label>
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            <option value="precision">Precision</option>
            <option value="recall">Recall</option>
            <option value="f1_score">F1 Score</option>
          </select>
        </div>
      </div>
      
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={400}>
          <LineChart
            data={chartData}
            margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="epsilon" 
              label={{ value: 'Privacy Parameter (ε)', position: 'bottom', offset: 0 }}
              domain={[0, 'dataMax']}
            />
            <YAxis
              label={{ value: metricLabels[metric], angle: -90, position: 'insideLeft' }}
              domain={[0, 1]}
            />
            <Tooltip />
            <Legend verticalAlign="top" />
            <Line 
              type="monotone" 
              dataKey="basic" 
              name="Basic Algorithm" 
              stroke="#8884d8" 
              activeDot={{ r: 8 }} 
              strokeWidth={2}
            />
            <Line 
              type="monotone" 
              dataKey="cms" 
              name="Count-Min Sketch" 
              stroke="#82ca9d" 
              activeDot={{ r: 8 }} 
              strokeWidth={2}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      
      <div className="privacy-explanation">
        <h4>Understanding the Privacy-Utility Tradeoff</h4>
        <p>
          The privacy parameter (ε) controls the tradeoff between privacy protection and algorithm accuracy:
        </p>
        <ul>
          <li><strong>Lower ε values</strong> (0.1-1.0) provide stronger privacy guarantees but typically result in lower accuracy.</li>
          <li><strong>Higher ε values</strong> (2.0-5.0) provide better accuracy but weaker privacy protection.</li>
          <li><strong>The "sweet spot"</strong> is often between 1.0-2.0, balancing privacy and utility.</li>
        </ul>
        
        <div className="privacy-scale-visualization">
          <div className="privacy-scale">
            <div className="scale-marker" style={{ left: '10%' }}>ε = 0.1</div>
            <div className="scale-marker" style={{ left: '30%' }}>ε = 1.0</div>
            <div className="scale-marker" style={{ left: '50%' }}>ε = 2.0</div>
            <div className="scale-marker" style={{ left: '70%' }}>ε = 5.0</div>
            <div className="scale-gradient"></div>
            <div className="scale-labels">
              <span>More Private</span>
              <span>More Accurate</span>
            </div>
          </div>
        </div>
        
        <h4>Key Observations</h4>
        <ul>
          <li>Count-Min Sketch consistently outperforms the basic algorithm across all privacy levels.</li>
          <li>The performance gap between algorithms is largest at moderate privacy levels (ε = 1.0-2.0).</li>
          <li>Both algorithms converge to similar performance at very high ε values, as privacy noise becomes minimal.</li>
        </ul>
      </div>
    </div>
  );
};

export default PrivacyImpactChart;