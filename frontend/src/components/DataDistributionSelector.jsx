import React from 'react';

/**
 * Component for selecting the data distribution type
 */
const DataDistributionSelector = ({ value, onChange }) => {
  // Distribution visualizations (simplified representations)
  const distributions = {
    uniform: [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
    zipf: [0.9, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.08, 0.07, 0.06],
    normal: [0.1, 0.2, 0.4, 0.7, 0.9, 0.9, 0.7, 0.4, 0.2, 0.1]
  };
  
  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-3">Data Distribution</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <label className={`relative border-2 rounded-lg p-4 cursor-pointer transition-all ${value === 'uniform' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
          <input
            type="radio"
            name="distribution"
            value="uniform"
            checked={value === 'uniform'}
            onChange={() => onChange('uniform')}
            className="sr-only"
          />
          <div className="h-24 flex items-end justify-center space-x-1 mb-3 px-2">
            {distributions.uniform.map((height, i) => (
              <div 
                key={`uniform-${i}`} 
                className="w-full bg-blue-400" 
                style={{ height: `${height * 100}%` }}
              />
            ))}
          </div>
          <div className="text-center">
            <span className="font-medium block">Uniform</span>
            <span className="text-sm text-gray-500 block">
              Equal probability
            </span>
          </div>
        </label>
        
        <label className={`relative border-2 rounded-lg p-4 cursor-pointer transition-all ${value === 'zipf' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
          <input
            type="radio"
            name="distribution"
            value="zipf"
            checked={value === 'zipf'}
            onChange={() => onChange('zipf')}
            className="sr-only"
          />
          <div className="h-24 flex items-end justify-center space-x-1 mb-3 px-2">
            {distributions.zipf.map((height, i) => (
              <div 
                key={`zipf-${i}`} 
                className="w-full bg-green-400" 
                style={{ height: `${height * 100}%` }}
              />
            ))}
          </div>
          <div className="text-center">
            <span className="font-medium block">Zipf (Power Law)</span>
            <span className="text-sm text-gray-500 block">
              Few frequent elements
            </span>
          </div>
        </label>
        
        <label className={`relative border-2 rounded-lg p-4 cursor-pointer transition-all ${value === 'normal' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
          <input
            type="radio"
            name="distribution"
            value="normal"
            checked={value === 'normal'}
            onChange={() => onChange('normal')}
            className="sr-only"
          />
          <div className="h-24 flex items-end justify-center space-x-1 mb-3 px-2">
            {distributions.normal.map((height, i) => (
              <div 
                key={`normal-${i}`} 
                className="w-full bg-purple-400" 
                style={{ height: `${height * 100}%` }}
              />
            ))}
          </div>
          <div className="text-center">
            <span className="font-medium block">Normal (Gaussian)</span>
            <span className="text-sm text-gray-500 block">
              Elements cluster in middle
            </span>
          </div>
        </label>
      </div>
      
      <div className="mt-4 p-4 bg-gray-50 rounded-md">
        {value === 'uniform' && (
          <p className="text-sm">
            In a uniform distribution, all elements have an equal probability of appearing.
            This is the simplest distribution but rarely occurs in real-world scenarios.
            The algorithm typically performs worse with this distribution as there are no
            clear heavy hitters.
          </p>
        )}
        
        {value === 'zipf' && (
          <p className="text-sm">
            The Zipf distribution (power law) models many real-world phenomena where a small
            number of elements occur very frequently while most elements are rare. This
            distribution is common in natural language, web traffic, and social media.
            The algorithm typically performs best with this distribution.
          </p>
        )}
        
        {value === 'normal' && (
          <p className="text-sm">
            The normal (Gaussian) distribution has elements clustered around the middle of the
            domain with symmetrically decreasing frequency toward the edges. This distribution
            occurs in many natural and social phenomena. The algorithm typically performs
            moderately well with this distribution.
          </p>
        )}
      </div>
    </div>
  );
};

export default DataDistributionSelector;