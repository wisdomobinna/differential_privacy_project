import React from 'react';

/**
 * Component for controlling algorithm parameters with sliders and inputs
 */
const ParameterControls = ({ params, onChange }) => {
  const { domain_size, num_elements, epsilon, threshold, num_hash_functions, algorithm_type } = params;
  
  // Handle slider changes
  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    onChange(name, parseFloat(value));
  };
  
  // Handle direct input changes
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    let parsedValue;
    
    switch (name) {
      case 'domain_size':
      case 'num_elements':
        parsedValue = parseInt(value, 10);
        break;
      case 'epsilon':
      case 'threshold':
      case 'num_hash_functions':
        parsedValue = parseFloat(value);
        break;
      default:
        parsedValue = value;
    }
    
    if (!isNaN(parsedValue)) {
      onChange(name, parsedValue);
    }
  };
  
  return (
    <div className="space-y-6">
      <div className="parameter-group">
        <label htmlFor="domain_size" className="block text-sm font-medium text-gray-700 mb-1">
          Domain Size
        </label>
        <div className="flex items-center space-x-4">
          <input
            type="range"
            id="domain_size_slider"
            name="domain_size"
            min="10"
            max="10000"
            step="10"
            value={domain_size}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            id="domain_size"
            name="domain_size"
            min="10"
            max="10000"
            value={domain_size}
            onChange={handleInputChange}
            className="w-20 px-2 py-1 border border-gray-300 rounded"
          />
        </div>
        <p className="mt-1 text-xs text-gray-500">
          Number of possible unique elements (higher = more diverse data)
        </p>
      </div>
      
      <div className="parameter-group">
        <label htmlFor="num_elements" className="block text-sm font-medium text-gray-700 mb-1">
          Number of Elements
        </label>
        <div className="flex items-center space-x-4">
          <input
            type="range"
            id="num_elements_slider"
            name="num_elements"
            min="100"
            max="100000"
            step="100"
            value={num_elements}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            id="num_elements"
            name="num_elements"
            min="100"
            max="100000"
            value={num_elements}
            onChange={handleInputChange}
            className="w-20 px-2 py-1 border border-gray-300 rounded"
          />
        </div>
        <p className="mt-1 text-xs text-gray-500">
          Total number of data points (higher = better statistics, slower processing)
        </p>
      </div>
      
      <div className="parameter-group">
        <label htmlFor="epsilon" className="block text-sm font-medium text-gray-700 mb-1">
          Privacy Parameter (ε)
        </label>
        <div className="flex items-center space-x-4">
          <input
            type="range"
            id="epsilon_slider"
            name="epsilon"
            min="0.1"
            max="5"
            step="0.1"
            value={epsilon}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            id="epsilon"
            name="epsilon"
            min="0.1"
            max="5"
            step="0.1"
            value={epsilon}
            onChange={handleInputChange}
            className="w-20 px-2 py-1 border border-gray-300 rounded"
          />
        </div>
        <div className="flex justify-between mt-1 text-xs text-gray-500">
          <span>More Private</span>
          <span>More Accurate</span>
        </div>
        <p className="mt-1 text-xs text-gray-500">
          Controls the privacy-utility tradeoff (lower = more privacy, less accuracy)
        </p>
      </div>
      
      <div className="parameter-group">
        <label htmlFor="threshold" className="block text-sm font-medium text-gray-700 mb-1">
          Threshold
        </label>
        <div className="flex items-center space-x-4">
          <input
            type="range"
            id="threshold_slider"
            name="threshold"
            min="0.001"
            max="0.1"
            step="0.001"
            value={threshold}
            onChange={handleSliderChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            id="threshold"
            name="threshold"
            min="0.001"
            max="0.1"
            step="0.001"
            value={threshold}
            onChange={handleInputChange}
            className="w-20 px-2 py-1 border border-gray-300 rounded"
          />
        </div>
        <p className="mt-1 text-xs text-gray-500">
          Minimum frequency required to be considered a heavy hitter
        </p>
      </div>
      
      {algorithm_type === 'cms' && (
        <div className="parameter-group">
          <label htmlFor="num_hash_functions" className="block text-sm font-medium text-gray-700 mb-1">
            Number of Hash Functions
          </label>
          <div className="flex items-center space-x-4">
            <input
              type="range"
              id="num_hash_functions_slider"
              name="num_hash_functions"
              min="1"
              max="10"
              step="1"
              value={num_hash_functions}
              onChange={handleSliderChange}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <input
              type="number"
              id="num_hash_functions"
              name="num_hash_functions"
              min="1"
              max="10"
              step="1"
              value={num_hash_functions}
              onChange={handleInputChange}
              className="w-20 px-2 py-1 border border-gray-300 rounded"
            />
          </div>
          <p className="mt-1 text-xs text-gray-500">
            Number of hash functions used in Count-Min Sketch (higher = better accuracy, slower)
          </p>
        </div>
      )}
      
      <div className="parameter-info-box bg-blue-50 p-4 rounded-md text-sm">
        <h4 className="font-medium text-blue-800 mb-2">Parameter Guidelines</h4>
        <ul className="space-y-1 text-blue-700">
          <li><span className="font-medium">Privacy (ε):</span> 0.1-1.0 for high privacy, 1.0-2.0 for balanced, 2.0-5.0 for high accuracy</li>
          <li><span className="font-medium">Threshold:</span> 0.001-0.01 for more heavy hitters, 0.01-0.1 for fewer heavy hitters</li>
          <li><span className="font-medium">Domain Size:</span> Should reflect the diversity of your data</li>
          <li><span className="font-medium">Elements:</span> More elements generally provide better statistical results</li>
          {algorithm_type === 'cms' && (
            <li><span className="font-medium">Hash Functions:</span> 3-5 provides a good balance between accuracy and speed</li>
          )}
        </ul>
      </div>
    </div>
  );
};

export default ParameterControls;