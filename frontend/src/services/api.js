/**
 * API service for communicating with the heavy hitters algorithm backend
 */

const API_BASE_URL = 'http://localhost:5000/api';

/**
 * Run the heavy hitters algorithm with specified parameters
 * @param {Object} params - Algorithm parameters
 * @returns {Promise} - Promise resolving to algorithm results
 */
export const runAlgorithm = async (params) => {
  try {
    const response = await fetch(`${API_BASE_URL}/run-algorithm`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(params),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `API returned status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error running algorithm:', error);
    throw error;
  }
};

/**
 * Fetch data about privacy parameter impact on algorithm performance
 * @returns {Promise} - Promise resolving to privacy impact data
 */
export const fetchPrivacyImpact = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/privacy-impact`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `API returned status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching privacy impact data:', error);
    throw error;
  }
};

/**
 * Fetch data about distribution impact on algorithm performance
 * @returns {Promise} - Promise resolving to distribution impact data
 */
export const fetchDistributionImpact = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/distribution-impact`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `API returned status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error fetching distribution impact data:', error);
    throw error;
  }
};