import React, { useState, useEffect } from 'react';

/**
 * Component for displaying a table of heavy hitters with filtering and sorting
 */
const HeavyHittersTable = ({ visualizationData, numTrueHeavyHitters, numEstimatedHeavyHitters, threshold }) => {
  const [filter, setFilter] = useState('all'); // 'all', 'true', 'false-positive', 'false-negative'
  const [sortBy, setSortBy] = useState('trueFrequency');
  const [sortDirection, setSortDirection] = useState('desc');
  const [tableData, setTableData] = useState([]);
  const [searchTerm, setSearchTerm] = useState('');
  
  // Process visualization data for the table
  useEffect(() => {
    if (visualizationData && visualizationData.length > 0) {
      // Classify each element
      const processedData = visualizationData.map(item => ({
        ...item,
        isTrueHeavyHitter: item.trueFrequency > threshold,
        isEstimatedHeavyHitter: item.estimatedFrequency > threshold,
        error: Math.abs(item.trueFrequency - item.estimatedFrequency),
        errorPercentage: item.trueFrequency > 0 
          ? Math.abs(item.trueFrequency - item.estimatedFrequency) / item.trueFrequency * 100
          : 0,
        classification: 
          item.trueFrequency > threshold && item.estimatedFrequency > threshold ? 'true-positive' :
          item.trueFrequency <= threshold && item.estimatedFrequency > threshold ? 'false-positive' :
          item.trueFrequency > threshold && item.estimatedFrequency <= threshold ? 'false-negative' :
          'true-negative'
      }));
      
      setTableData(processedData);
    }
  }, [visualizationData, threshold]);
  
  // Apply filters and sorting
  const getDisplayData = () => {
    let filtered = [...tableData];
    
    // Apply search filter if any
    if (searchTerm) {
      filtered = filtered.filter(item => 
        item.element.toString().includes(searchTerm)
      );
    }
    
    // Apply classification filter
    switch (filter) {
      case 'true-positive':
        filtered = filtered.filter(item => 
          item.isTrueHeavyHitter && item.isEstimatedHeavyHitter
        );
        break;
      case 'false-positive':
        filtered = filtered.filter(item => 
          !item.isTrueHeavyHitter && item.isEstimatedHeavyHitter
        );
        break;
      case 'false-negative':
        filtered = filtered.filter(item => 
          item.isTrueHeavyHitter && !item.isEstimatedHeavyHitter
        );
        break;
      case 'true-negative':
        filtered = filtered.filter(item => 
          !item.isTrueHeavyHitter && !item.isEstimatedHeavyHitter
        );
        break;
      // No filter needed for 'all'
      default:
        break;
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      let comparison = 0;
      
      switch (sortBy) {
        case 'element':
          comparison = a.element - b.element;
          break;
        case 'trueFrequency':
          comparison = a.trueFrequency - b.trueFrequency;
          break;
        case 'estimatedFrequency':
          comparison = a.estimatedFrequency - b.estimatedFrequency;
          break;
        case 'error':
          comparison = a.error - b.error;
          break;
        case 'errorPercentage':
          comparison = a.errorPercentage - b.errorPercentage;
          break;
        default:
          break;
      }
      
      return sortDirection === 'asc' ? comparison : -comparison;
    });
    
    return filtered;
  };
  
  // Toggle sort direction when clicking on the same column
  const handleSortChange = (column) => {
    if (sortBy === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortDirection('desc'); // Default to descending for new column
    }
  };
  
  // Get sorted and filtered data
  const displayData = getDisplayData();
  
  // Get counts for different classifications
  const counts = {
    total: tableData.length,
    truePositive: tableData.filter(item => item.classification === 'true-positive').length,
    falsePositive: tableData.filter(item => item.classification === 'false-positive').length,
    falseNegative: tableData.filter(item => item.classification === 'false-negative').length,
    trueNegative: tableData.filter(item => item.classification === 'true-negative').length
  };
  
  return (
    <div className="w-full mt-8">
      <h3 className="text-xl font-semibold mb-4">Heavy Hitters Analysis</h3>
      
      <div className="flex justify-between items-center mb-4">
        <div className="stats-summary flex gap-4">
          <div className="stat-item px-3 py-2 bg-gray-100 rounded">
            <span className="block text-sm font-medium text-gray-600">True Heavy Hitters</span>
            <span className="block text-lg font-bold">{numTrueHeavyHitters}</span>
          </div>
          <div className="stat-item px-3 py-2 bg-gray-100 rounded">
            <span className="block text-sm font-medium text-gray-600">Estimated Heavy Hitters</span>
            <span className="block text-lg font-bold">{numEstimatedHeavyHitters}</span>
          </div>
          <div className="stat-item px-3 py-2 bg-gray-100 rounded">
            <span className="block text-sm font-medium text-gray-600">Threshold</span>
            <span className="block text-lg font-bold">{threshold.toFixed(4)}</span>
          </div>
        </div>
        
        <div className="search-input">
          <input
            type="text"
            placeholder="Search elements..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="border rounded px-3 py-2 w-48"
          />
        </div>
      </div>
      
      <div className="filter-tabs flex gap-2 mb-4">
        <button 
          className={`px-3 py-1 rounded ${filter === 'all' ? 'bg-blue-600 text-white' : 'bg-gray-200'}`}
          onClick={() => setFilter('all')}
        >
          All ({counts.total})
        </button>
        <button 
          className={`px-3 py-1 rounded ${filter === 'true-positive' ? 'bg-green-600 text-white' : 'bg-green-100'}`}
          onClick={() => setFilter('true-positive')}
        >
          True Positives ({counts.truePositive})
        </button>
        <button 
          className={`px-3 py-1 rounded ${filter === 'false-positive' ? 'bg-red-600 text-white' : 'bg-red-100'}`}
          onClick={() => setFilter('false-positive')}
        >
          False Positives ({counts.falsePositive})
        </button>
        <button 
          className={`px-3 py-1 rounded ${filter === 'false-negative' ? 'bg-yellow-600 text-white' : 'bg-yellow-100'}`}
          onClick={() => setFilter('false-negative')}
        >
          False Negatives ({counts.falseNegative})
        </button>
        <button 
          className={`px-3 py-1 rounded ${filter === 'true-negative' ? 'bg-gray-600 text-white' : 'bg-gray-100'}`}
          onClick={() => setFilter('true-negative')}
        >
          True Negatives ({counts.trueNegative})
        </button>
      </div>
      
      <div className="overflow-x-auto">
        <table className="min-w-full bg-white border">
          <thead>
            <tr className="bg-gray-100">
              <th className="py-2 px-4 border cursor-pointer" onClick={() => handleSortChange('element')}>
                Element
                {sortBy === 'element' && (
                  <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="py-2 px-4 border cursor-pointer" onClick={() => handleSortChange('trueFrequency')}>
                True Frequency
                {sortBy === 'trueFrequency' && (
                  <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="py-2 px-4 border cursor-pointer" onClick={() => handleSortChange('estimatedFrequency')}>
                Estimated Frequency
                {sortBy === 'estimatedFrequency' && (
                  <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="py-2 px-4 border cursor-pointer" onClick={() => handleSortChange('error')}>
                Error
                {sortBy === 'error' && (
                  <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="py-2 px-4 border cursor-pointer" onClick={() => handleSortChange('errorPercentage')}>
                Error %
                {sortBy === 'errorPercentage' && (
                  <span className="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                )}
              </th>
              <th className="py-2 px-4 border">
                Classification
              </th>
            </tr>
          </thead>
          <tbody>
            {displayData.map((item, index) => (
              <tr 
                key={index}
                className={
                  item.classification === 'true-positive' ? 'bg-green-50' :
                  item.classification === 'false-positive' ? 'bg-red-50' :
                  item.classification === 'false-negative' ? 'bg-yellow-50' :
                  ''
                }
              >
                <td className="py-2 px-4 border">{item.element}</td>
                <td className="py-2 px-4 border">
                  <div className="flex items-center">
                    <div 
                      className="w-16 h-2 bg-blue-200 mr-2 rounded-sm" 
                      style={{ width: `${Math.min(item.trueFrequency * 500, 100)}%` }}
                    />
                    <span>{item.trueFrequency.toFixed(4)}</span>
                    {item.isTrueHeavyHitter && (
                      <span className="ml-2 text-xs bg-blue-100 px-1 rounded">HH</span>
                    )}
                  </div>
                </td>
                <td className="py-2 px-4 border">
                  <div className="flex items-center">
                    <div 
                      className="w-16 h-2 bg-green-200 mr-2 rounded-sm" 
                      style={{ width: `${Math.min(item.estimatedFrequency * 500, 100)}%` }}
                    />
                    <span>{item.estimatedFrequency.toFixed(4)}</span>
                    {item.isEstimatedHeavyHitter && (
                      <span className="ml-2 text-xs bg-green-100 px-1 rounded">HH</span>
                    )}
                  </div>
                </td>
                <td className="py-2 px-4 border">
                  {item.error.toFixed(4)}
                </td>
                <td className="py-2 px-4 border">
                  <div className="flex items-center">
                    <div 
                      className={`w-16 h-2 mr-2 rounded-sm ${
                        item.errorPercentage < 10 ? 'bg-green-300' :
                        item.errorPercentage < 25 ? 'bg-yellow-300' :
                        'bg-red-300'
                      }`}
                      style={{ width: `${Math.min(item.errorPercentage, 100)}%` }}
                    />
                    <span>{item.errorPercentage.toFixed(2)}%</span>
                  </div>
                </td>
                <td className="py-2 px-4 border">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    item.classification === 'true-positive' ? 'bg-green-100 text-green-800' :
                    item.classification === 'false-positive' ? 'bg-red-100 text-red-800' :
                    item.classification === 'false-negative' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-gray-100 text-gray-800'
                  }`}>
                    {item.classification.split('-').map(word => 
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ')}
                  </span>
                </td>
              </tr>
            ))}
            {displayData.length === 0 && (
              <tr>
                <td className="py-4 px-4 border text-center text-gray-500" colSpan="6">
                  No matching elements found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      
      <div className="mt-4 bg-gray-50 p-4 rounded">
        <h4 className="font-medium mb-2">Classification Guide</h4>
        <ul className="text-sm">
          <li className="mb-1">
            <span className="inline-block w-28">
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 text-green-800">True Positive</span>
            </span>
            <span>: Correctly identified as a heavy hitter</span>
          </li>
          <li className="mb-1">
            <span className="inline-block w-28">
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800">False Positive</span>
            </span>
            <span>: Incorrectly identified as a heavy hitter</span>
          </li>
          <li className="mb-1">
            <span className="inline-block w-28">
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">False Negative</span>
            </span>
            <span>: Failed to identify a true heavy hitter</span>
          </li>
          <li>
            <span className="inline-block w-28">
              <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">True Negative</span>
            </span>
            <span>: Correctly identified as not a heavy hitter</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default HeavyHittersTable;