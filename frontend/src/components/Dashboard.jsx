import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

/**
 * Dashboard component for visualizing distribution impact on algorithm performance
 */
const Dashboard = ({ data }) => {
  const [metric, setMetric] = useState('f1_score');
  const [chartType, setChartType] = useState('bar');
  
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center p-10 bg-gray-50 rounded-lg shadow-sm">
        <p className="text-gray-500 text-lg">No distribution impact data available</p>
      </div>
    );
  }
  
  // Format data for different chart types
  const barChartData = data.map(item => ({
    distribution: item.distribution,
    basicAlgorithm: item.basic[metric],
    countMinSketch: item.cms[metric],
    numHeavyHitters: item.numTrueHeavyHitters
  }));
  
  // Format for radar chart
  const radarChartData = [
    { subject: 'Precision', basic: 0, cms: 0 },
    { subject: 'Recall', basic: 0, cms: 0 },
    { subject: 'F1 Score', basic: 0, cms: 0 }
  ];
  
  // Calculate averages for radar chart
  data.forEach(item => {
    radarChartData[0].basic += item.basic.precision / data.length;
    radarChartData[0].cms += item.cms.precision / data.length;
    radarChartData[1].basic += item.basic.recall / data.length;
    radarChartData[1].cms += item.cms.recall / data.length;
    radarChartData[2].basic += item.basic.f1_score / data.length;
    radarChartData[2].cms += item.cms.f1_score / data.length;
  });
  
  const metricLabels = {
    precision: 'Precision',
    recall: 'Recall',
    f1_score: 'F1 Score'
  };
  
  const distributionDescriptions = {
    uniform: "In uniform distribution, all elements appear with equal probability. The lack of clear heavy hitters makes detection more challenging.",
    zipf: "Zipf distribution (power law) has a few very frequent elements, making heavy hitters more distinct and easier to detect.",
    normal: "Normal distribution clusters elements around the center of the domain with symmetrically decreasing frequency toward the edges."
  };
  
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 bg-gray-50 rounded-lg shadow-md">
      <div className="flex flex-col md:flex-row gap-6 mb-8 items-center justify-between">
        <div className="w-full md:w-auto">
          <label className="block text-sm font-medium text-gray-700 mb-1">Metric:</label>
          <select 
            value={metric} 
            onChange={(e) => setMetric(e.target.value)}
            className="w-full md:w-48 px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          >
            <option value="precision">Precision</option>
            <option value="recall">Recall</option>
            <option value="f1_score">F1 Score</option>
          </select>
        </div>
        
        <div className="flex space-x-2">
          <button 
            className={`px-4 py-2 rounded-md font-medium transition ${
              chartType === 'bar' 
                ? 'bg-indigo-600 text-white shadow-md' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
            onClick={() => setChartType('bar')}
          >
            Bar Chart
          </button>
          <button 
            className={`px-4 py-2 rounded-md font-medium transition ${
              chartType === 'radar' 
                ? 'bg-indigo-600 text-white shadow-md' 
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
            onClick={() => setChartType('radar')}
          >
            Radar Chart
          </button>
        </div>
      </div>
      
      <div className="bg-white p-4 rounded-lg shadow-sm mb-8">
        {chartType === 'bar' ? (
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={barChartData}
              margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
            >
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="distribution" 
                label={{ value: 'Data Distribution', position: 'bottom', offset: 0 }}
              />
              <YAxis
                label={{ value: metricLabels[metric], angle: -90, position: 'insideLeft' }}
                domain={[0, 1]}
              />
              <Tooltip contentStyle={{ borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
              <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '10px' }} />
              <Bar 
                dataKey="basicAlgorithm" 
                name="Basic Algorithm" 
                fill="#8884d8" 
                maxBarSize={60}
              />
              <Bar 
                dataKey="countMinSketch" 
                name="Count-Min Sketch" 
                fill="#82ca9d" 
                maxBarSize={60}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height={400}>
            <RadarChart outerRadius={150} data={radarChartData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis domain={[0, 1]} />
              <Radar 
                name="Basic Algorithm" 
                dataKey="basic" 
                stroke="#8884d8" 
                fill="#8884d8" 
                fillOpacity={0.6} 
              />
              <Radar 
                name="Count-Min Sketch" 
                dataKey="cms" 
                stroke="#82ca9d" 
                fill="#82ca9d" 
                fillOpacity={0.6} 
              />
              <Legend />
              <Tooltip contentStyle={{ borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }} />
            </RadarChart>
          </ResponsiveContainer>
        )}
      </div>
      
      <div className="space-y-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">Distribution Characteristics</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {data.map(item => (
            <div className="bg-white p-5 rounded-lg shadow-sm hover:shadow-md transition" key={item.distribution}>
              <h4 className="text-lg font-semibold text-indigo-700 mb-3">
                {item.distribution.charAt(0).toUpperCase() + item.distribution.slice(1)} Distribution
              </h4>
              <div className="bg-gray-50 p-3 rounded-md mb-3">
                {item.distribution === 'uniform' && (
                  <div className="bars-container">
                    {[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4].map((height, i) => (
                      <div key={i} className="bar" style={{ height: `${height * 100}%` }}></div>
                    ))}
                  </div>
                )}
                {item.distribution === 'zipf' && (
                  <div className="bars-container">
                    {[0.9, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08].map((height, i) => (
                      <div key={i} className="bar" style={{ height: `${height * 100}%` }}></div>
                    ))}
                  </div>
                )}
                {item.distribution === 'normal' && (
                  <div className="bars-container">
                    {[0.1, 0.2, 0.4, 0.7, 0.9, 0.7, 0.4, 0.2].map((height, i) => (
                      <div key={i} className="bar" style={{ height: `${height * 100}%` }}></div>
                    ))}
                  </div>
                )}
              </div>
              <p className="text-sm text-gray-600 mb-4">{distributionDescriptions[item.distribution]}</p>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-gray-50 p-2 rounded">
                  <span className="text-xs text-gray-500 block">True Heavy Hitters</span>
                  <span className="font-semibold text-indigo-700">{item.numTrueHeavyHitters}</span>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <span className="text-xs text-gray-500 block">Best Algorithm</span>
                  <span className="font-semibold text-indigo-700">
                    {item.cms.f1_score > item.basic.f1_score ? 'CMS' : 'Basic'}
                  </span>
                </div>
                <div className="bg-gray-50 p-2 rounded col-span-2">
                  <span className="text-xs text-gray-500 block">F1 Score Difference</span>
                  <span className="font-semibold text-indigo-700">
                    {Math.abs(item.cms.f1_score - item.basic.f1_score).toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm mt-8">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Key Findings</h3>
          <ul className="space-y-3 text-gray-700">
            <li className="flex">
              <span className="text-indigo-500 mr-2">•</span>
              <div>
                <strong className="text-indigo-700">Zipf Distribution:</strong> Generally provides the best algorithm performance due to the
                clear distinction between frequent and infrequent elements.
              </div>
            </li>
            <li className="flex">
              <span className="text-indigo-500 mr-2">•</span>
              <div>
                <strong className="text-indigo-700">Uniform Distribution:</strong> Presents the biggest challenge for heavy hitter detection
                since all elements have similar frequencies.
              </div>
            </li>
            <li className="flex">
              <span className="text-indigo-500 mr-2">•</span>
              <div>
                <strong className="text-indigo-700">Count-Min Sketch Advantage:</strong> The performance gap between basic and CMS algorithms
                is most pronounced with uniform and normal distributions.
              </div>
            </li>
            <li className="flex">
              <span className="text-indigo-500 mr-2">•</span>
              <div>
                <strong className="text-indigo-700">Real-World Applicability:</strong> Since many real-world phenomena follow Zipf distributions
                (e.g., word frequencies, web traffic), the algorithm is well-suited for practical applications.
              </div>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;