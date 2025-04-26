import React from 'react';

/**
 * Component for selecting the heavy hitters algorithm type
 */
const AlgorithmSelector = ({ value, onChange }) => {
  return (
    <div className="mb-6">
      <h3 className="text-lg font-semibold mb-3">Algorithm</h3>
      
      <div className="flex gap-4">
        <label className={`relative flex flex-col items-center p-4 rounded-lg border-2 cursor-pointer transition-all ${value === 'basic' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
          <input
            type="radio"
            name="algorithm"
            value="basic"
            checked={value === 'basic'}
            onChange={() => onChange('basic')}
            className="sr-only"
          />
          <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
            </svg>
          </div>
          <div className="text-center">
            <span className="font-medium block">Basic Algorithm</span>
            <span className="text-sm text-gray-500 block">Single hash function</span>
          </div>
        </label>
        
        <label className={`relative flex flex-col items-center p-4 rounded-lg border-2 cursor-pointer transition-all ${value === 'cms' ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-blue-200'}`}>
          <input
            type="radio"
            name="algorithm"
            value="cms"
            checked={value === 'cms'}
            onChange={() => onChange('cms')}
            className="sr-only"
          />
          <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
            </svg>
          </div>
          <div className="text-center">
            <span className="font-medium block">Count-Min Sketch</span>
            <span className="text-sm text-gray-500 block">Multiple hash functions</span>
          </div>
        </label>
      </div>
      
      <div className="mt-4 p-4 bg-gray-50 rounded-md">
        {value === 'basic' ? (
          <div>
            <p className="mb-2 text-sm">
              The basic algorithm uses a single hash function to map elements to a fixed-size array
              and applies randomized response to achieve local differential privacy.
            </p>
            <ul className="list-disc pl-5 text-sm space-y-1">
              <li>Simpler implementation</li>
              <li>Faster processing</li>
              <li>Lower memory requirements</li>
              <li>More susceptible to hash collisions</li>
            </ul>
          </div>
        ) : (
          <div>
            <p className="mb-2 text-sm">
              The Count-Min Sketch algorithm uses multiple hash functions to reduce collision 
              probability and improve accuracy, while still maintaining privacy guarantees.
            </p>
            <ul className="list-disc pl-5 text-sm space-y-1">
              <li>Better accuracy</li>
              <li>More resilient to hash collisions</li>
              <li>Higher memory usage</li>
              <li>Slightly slower processing</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};

export default AlgorithmSelector;