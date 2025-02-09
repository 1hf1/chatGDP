import React, { useState, useEffect } from 'react';
import { Settings } from 'lucide-react';
import type { ModelType } from '../types';

interface ModelConfigProps {
  variables: string[];
  onConfigure: (config: ModelConfiguration) => void;
  onCancel: () => void;
  darkMode: boolean;
}

export interface ModelConfiguration {
  selectedVariables: string[];
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  predictDifferences: boolean;
  modelType: ModelType;
  arLags: number;
  maPeriod: number;
  lstmLookback?: number;
}

const DEFAULT_VARIABLES = [
  'CPI_All_Items',
  'Real_Disposable_Income',
  'Avg_Hourly_Earnings',
  'Producer_Price_Index',
  'Consumer_Sentiment',
  'Trade_Balance'
];

export function ModelConfig({ variables, onConfigure, onCancel, darkMode }: ModelConfigProps) {
  const [selectedVars, setSelectedVars] = useState<string[]>([]);
  const [nEstimators, setNEstimators] = useState(10);
  const [maxDepth, setMaxDepth] = useState(3);
  const [minSamplesSplit, setMinSamplesSplit] = useState(5);
  const [predictDifferences, setPredictDifferences] = useState(false);
  const [modelType, setModelType] = useState<ModelType>('constrainedMA');
  const [arLags, setArLags] = useState(3);
  const [maPeriod, setMAPeriod] = useState(3);
  const [lstmLookback, setLSTMLookback] = useState(4);

  // Set default selected variables when component mounts
  useEffect(() => {
    const defaultSelected = variables.filter(v => DEFAULT_VARIABLES.includes(v));
    setSelectedVars(defaultSelected);
  }, [variables]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onConfigure({
      selectedVariables: selectedVars,
      nEstimators,
      maxDepth,
      minSamplesSplit,
      predictDifferences,
      modelType,
      arLags,
      maPeriod,
      lstmLookback
    });
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4">
      <div className={`${darkMode ? 'bg-gray-800' : 'bg-white'} rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto`}>
        <div className="p-6">
          <div className="flex items-center gap-2 mb-6">
            <Settings className={`w-6 h-6 ${darkMode ? 'text-blue-400' : 'text-blue-600'}`} />
            <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>Model Configuration</h2>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                Model Type
              </label>
              <select
                value={modelType}
                onChange={(e) => setModelType(e.target.value as ModelType)}
                className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                  darkMode 
                    ? 'bg-gray-700 border-gray-600 text-white' 
                    : 'border-gray-300 text-gray-900'
                }`}
              >
                <option value="constrainedMA">Constrained MA</option>
                <option value="randomForest">Random Forest</option>
                <option value="autoregressive">Autoregressive</option>
                <option value="lstm">LSTM (Beta Mode: won't train in time)</option>
              </select>
            </div>

            <div>
              <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                Select Variables to Include
              </label>
              <div className={`grid grid-cols-2 md:grid-cols-3 gap-3 p-4 border rounded-lg ${
                darkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-50 border-gray-200'
              }`}>
                {variables.map(variable => (
                  <label key={variable} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedVars.includes(variable)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedVars([...selectedVars, variable]);
                        } else {
                          setSelectedVars(selectedVars.filter(v => v !== variable));
                        }
                      }}
                      className={`rounded border-gray-300 text-blue-600 focus:ring-blue-500 ${
                        darkMode ? 'bg-gray-600' : ''
                      }`}
                    />
                    <span className={`text-sm ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                      {variable}
                    </span>
                  </label>
                ))}
              </div>
            </div>

            {modelType === 'constrainedMA' && (
              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                  MA Period
                </label>
                <input
                  type="number"
                  min="2"
                  max="10"
                  value={maPeriod}
                  onChange={(e) => setMAPeriod(Number(e.target.value))}
                  className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-white' 
                      : 'border-gray-300'
                  }`}
                />
                <p className={`mt-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Recommended: 3-5 periods
                </p>
              </div>
            )}

            {modelType === 'lstm' && (
              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                  Lookback Period
                </label>
                <input
                  type="number"
                  min="2"
                  max="12"
                  value={lstmLookback}
                  onChange={(e) => setLSTMLookback(Number(e.target.value))}
                  className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-white' 
                      : 'border-gray-300'
                  }`}
                />
                <p className={`mt-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Recommended: 4-8 periods
                </p>
              </div>
            )}

            {modelType === 'randomForest' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                    Number of Trees
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="100"
                    value={nEstimators}
                    onChange={(e) => setNEstimators(Number(e.target.value))}
                    className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                      darkMode 
                        ? 'bg-gray-700 border-gray-600 text-white' 
                        : 'border-gray-300'
                    }`}
                  />
                  <p className={`mt-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Recommended: 5-20 trees
                  </p>
                </div>

                <div>
                  <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                    Maximum Tree Depth
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(Number(e.target.value))}
                    className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                      darkMode 
                        ? 'bg-gray-700 border-gray-600 text-white' 
                        : 'border-gray-300'
                    }`}
                  />
                  <p className={`mt-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Recommended: 2-5 levels
                  </p>
                </div>

                <div>
                  <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                    Min Samples to Split
                  </label>
                  <input
                    type="number"
                    min="2"
                    max="20"
                    value={minSamplesSplit}
                    onChange={(e) => setMinSamplesSplit(Number(e.target.value))}
                    className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                      darkMode 
                        ? 'bg-gray-700 border-gray-600 text-white' 
                        : 'border-gray-300'
                    }`}
                  />
                  <p className={`mt-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    Recommended: 3-10 samples
                  </p>
                </div>
              </div>
            )}

            {modelType === 'autoregressive' && (
              <div>
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                  Number of Lags
                </label>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={arLags}
                  onChange={(e) => setArLags(Number(e.target.value))}
                  className={`w-full rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500 ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-white' 
                      : 'border-gray-300'
                  }`}
                />
                <p className={`mt-1 text-xs ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                  Recommended: 3-5 lags
                </p>
              </div>
            )}

            <div className="flex justify-end space-x-4 pt-4">
              <button
                type="button"
                onClick={onCancel}
                className={`px-4 py-2 text-sm font-medium rounded-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 ${
                  darkMode
                    ? 'text-gray-200 bg-gray-700 hover:bg-gray-600 border-gray-600'
                    : 'text-gray-700 bg-white border border-gray-300 hover:bg-gray-50'
                }`}
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={selectedVars.length === 0}
                className={`px-4 py-2 text-sm font-medium text-white bg-blue-600 border border-transparent rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed ${
                  darkMode ? 'focus:ring-offset-gray-800' : ''
                }`}
              >
                Run Analysis
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}