import React, { useState, useEffect } from 'react';
import { TrendingUp, ChevronDown, ChevronUp, ArrowRight, Beaker } from 'lucide-react';
import type { PredictionResult, EconomicData } from '../types';
import { VariableChart } from './VariableChart';
import { calculateRandomForest } from '../utils/statistics';
import { calculateConstrainedMA } from '../utils/constrainedMA';
import { isValidNumber } from '../utils/validation';
import { ProductTour } from './ProductTour';

interface PredictionResultsProps {
  results: PredictionResult[];
  historicalData: EconomicData[];
  predictDifferences: boolean;
}

export function PredictionResults({ results: initialResults, historicalData, predictDifferences }: PredictionResultsProps) {
  const [expandedVariable, setExpandedVariable] = useState<string | null>(null);
  const [currentData, setCurrentData] = useState<EconomicData[]>(historicalData);
  const [quarterOffset, setQuarterOffset] = useState(0);
  const [results, setResults] = useState<PredictionResult[]>(initialResults);
  const [trainedModels, setTrainedModels] = useState<Map<string, any>>(new Map());
  const [predictions, setPredictions] = useState<Map<string, number[]>>(new Map());
  const [experimentalValues, setExperimentalValues] = useState<Map<string, number>>(new Map());
  const [isProcessing, setIsProcessing] = useState(false);
  const [showTour, setShowTour] = useState(true);
  const [tourStep, setTourStep] = useState(1);
  const [currentQuarter, setCurrentQuarter] = useState(() => {
    const date = new Date();
    return {
      quarter: Math.floor((date.getMonth() + 3) / 3),
      year: date.getFullYear()
    };
  });

  // Auto-expand first variable when tour starts
  useEffect(() => {
    if (showTour && results.length > 0) {
      setExpandedVariable(results[0].variable);
    }
  }, [showTour, results]);

  // Get current quarter label
  const getCurrentQuarter = () => {
    return `Q${currentQuarter.quarter} ${currentQuarter.year}`;
  };

  const handleNextTourStep = () => {
    setTourStep(prev => prev + 1);
  };

  const handleTourComplete = () => {
    setShowTour(false);
    localStorage.setItem('tourCompleted', 'true');
  };

  useEffect(() => {
    const tourCompleted = localStorage.getItem('tourCompleted');
    if (tourCompleted) {
      setShowTour(false);
    }
  }, []);

  const predictNextQuarter = (data: EconomicData[], experimentalOverrides: Map<string, number> = new Map()) => {
    const newResults: PredictionResult[] = [];
    const shouldRetrain = quarterOffset % 4 === 0;
    
    for (const prevResult of results) {
      const targetVar = prevResult.variable;
      
      if (prevResult.modelType === 'constrainedMA') {
        const timeSeriesData = data.map(row => Number(row[targetVar]))
          .filter(isValidNumber);
        
        const result = calculateConstrainedMA(timeSeriesData, 3);
        
        if (result) {
          const prediction = result.prediction;
          if (isValidNumber(prediction)) {
            const prevPredictions = predictions.get(targetVar) || [];
            predictions.set(targetVar, [...prevPredictions, prediction]);

            newResults.push({
              variable: targetVar,
              prediction,
              details: [],
              modelType: 'constrainedMA',
              direction: result.direction,
              confidence: result.confidence,
              maChange: result.maChange
            });
          }
        }
      } else {
        const inputVars = prevResult.details.map(d => d.inputVar);
        let trainResult;
        
        if (shouldRetrain) {
          const trainingData = data.slice(0, -1)
            .map((row, index) => {
              const nextRow = data[index + 1];
              const inputValues = inputVars.map(v => Number(row[v]));
              const targetValue = predictDifferences
                ? Number(nextRow[targetVar]) - Number(row[targetVar])
                : Number(nextRow[targetVar]);

              if ([...inputValues, targetValue].every(isValidNumber)) {
                return [...inputValues, targetValue];
              }
              return null;
            })
            .filter((row): row is number[] => row !== null);

          if (trainingData.length >= 10) {
            trainResult = calculateRandomForest(
              trainingData,
              inputVars,
              10,
              3,
              5
            );
            
            if (trainResult) {
              trainedModels.set(targetVar, trainResult);
            }
          }
        } else {
          trainResult = trainedModels.get(targetVar);
        }
        
        if (trainResult) {
          const lastDataPoint = data[data.length - 1];
          const inputData = inputVars.map(v => {
            const value = experimentalOverrides.has(v) 
              ? experimentalOverrides.get(v)! 
              : Number(lastDataPoint[v]);
            return isValidNumber(value) ? value : 0;
          });
          
          const rawPrediction = trainResult.forest.predict(inputData);
          if (isValidNumber(rawPrediction)) {
            const prediction = predictDifferences
              ? Number(lastDataPoint[targetVar]) + rawPrediction
              : rawPrediction;

            if (isValidNumber(prediction)) {
              const details = inputVars.map((inputVar, idx) => ({
                inputVar,
                importance: trainResult.importance[idx],
                prediction: rawPrediction
              }));

              const prevPredictions = predictions.get(targetVar) || [];
              predictions.set(targetVar, [...prevPredictions, prediction]);

              newResults.push({
                variable: targetVar,
                prediction,
                details: details.sort((a, b) => b.importance - a.importance),
                modelType: prevResult.modelType
              });
            }
          }
        }
      }
    }
    
    setPredictions(new Map(predictions));
    return newResults;
  };

  const handleExperimentalValueChange = (variable: string, value: number) => {
    if (!isValidNumber(value)) return;
    
    const newExperimentalValues = new Map(experimentalValues);
    newExperimentalValues.set(variable, value);
    setExperimentalValues(newExperimentalValues);
    
    const newResults = predictNextQuarter(currentData, newExperimentalValues);
    setResults(newResults);
  };

  const advanceQuarter = async () => {
    if (isProcessing) return;
    setIsProcessing(true);

    const newDataPoint: EconomicData = { ...currentData[currentData.length - 1] };
    
    results.forEach(result => {
      const value = experimentalValues.has(result.variable) 
        ? experimentalValues.get(result.variable)! 
        : result.prediction;
      newDataPoint[result.variable] = value;
    });
    
    const newData = [...currentData, newDataPoint];
    setCurrentData(newData);
    setExperimentalValues(new Map());
    
    const newResults = predictNextQuarter(newData);
    setResults(newResults);
    setQuarterOffset(prev => prev + 1);

    // Update current quarter
    setCurrentQuarter(prev => {
      let newQuarter = prev.quarter + 1;
      let newYear = prev.year;
      
      if (newQuarter > 4) {
        newQuarter = 1;
        newYear++;
      }
      
      return {
        quarter: newQuarter,
        year: newYear
      };
    });
    
    setIsProcessing(false);
  };

  const resetSimulation = () => {
    setCurrentData(historicalData);
    setQuarterOffset(0);
    setResults(initialResults);
    setPredictions(new Map());
    setExperimentalValues(new Map());
    setIsProcessing(false);
    setExpandedVariable(null);
    // Reset quarter to current date
    const date = new Date();
    setCurrentQuarter({
      quarter: Math.floor((date.getMonth() + 3) / 3),
      year: date.getFullYear()
    });
  };

  const toggleExpand = (variable: string) => {
    setExpandedVariable(expandedVariable === variable ? null : variable);
  };

  return (
    <div className="w-full max-w-4xl">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
        <div className="flex flex-col">
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <TrendingUp className="w-6 h-6" />
            Simulate the US Economy!
          </h2>
          <p className="text-sm text-gray-600 mt-1">
            Current Quarter: {getCurrentQuarter()}
          </p>
          {quarterOffset % 4 === 0 && (
            <span className="text-sm text-blue-600 mt-1">
              (Models retrained this quarter)
            </span>
          )}
          {experimentalValues.size > 0 && (
            <span className="text-sm text-purple-600 mt-1 flex items-center gap-1">
              <Beaker className="w-4 h-4" />
              (Experimental Mode)
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={resetSimulation}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            Reset Simulation
          </button>
          <button
            onClick={advanceQuarter}
            disabled={isProcessing}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 whitespace-nowrap"
            data-tour="next-quarter"
          >
            Next Quarter <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-4">
        {results.map((result) => (
          <div
            key={result.variable}
            className="bg-white rounded-lg shadow-md overflow-hidden"
          >
            <div 
              className="p-4 cursor-pointer hover:bg-gray-50 flex justify-between items-center"
              onClick={() => !showTour && toggleExpand(result.variable)}
              data-tour={result.variable === results[0].variable ? "variable-expand" : undefined}
            >
              <div>
                <h3 className="font-medium text-gray-700">{result.variable}</h3>
                <p className="text-2xl font-bold text-blue-600">
                  {isValidNumber(result.prediction) ? result.prediction.toFixed(2) : 'N/A'}
                </p>
                {result.modelType === 'constrainedMA' && (
                  <div className="mt-2 space-y-1">
                    <p className={`text-sm ${
                      result.direction === 'up' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      Direction: {result.direction === 'up' ? '↑' : '↓'} 
                    </p>
                    <p className="text-sm text-gray-500">
                      MA Change: {result.maChange!.toFixed(2)}
                    </p>
                  </div>
                )}
                {experimentalValues.has(result.variable) && (
                  <p className="text-sm text-purple-600 flex items-center gap-1 mt-1">
                    <Beaker className="w-4 h-4" />
                    Experimental Value
                  </p>
                )}
              </div>
              {expandedVariable === result.variable ? 
                <ChevronUp className="w-5 h-5 text-gray-500" /> : 
                <ChevronDown className="w-5 h-5 text-gray-500" />
              }
            </div>
            
            {expandedVariable === result.variable && (
              <div className="border-t border-gray-200 p-4 space-y-4">
                <VariableChart
                  variable={result.variable}
                  historicalData={currentData}
                  prediction={result.prediction}
                  quarterOffset={quarterOffset}
                  predictions={predictions.get(result.variable) || []}
                  onExperimentalValueChange={(value) => handleExperimentalValueChange(result.variable, value)}
                />
                
                {result.details.length > 0 && (
                  <>
                    <h4 className="font-medium text-gray-700 mt-4 mb-2">Feature Importance:</h4>
                    <div className="space-y-4">
                      {result.details.map((detail) => (
                        <div key={detail.inputVar} className="bg-gray-50 p-3 rounded">
                          <h5 className="font-medium text-gray-700 mb-1">Input: {detail.inputVar}</h5>
                          <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>
                              <span className="text-gray-500">Importance:</span>{' '}
                              <span className="font-medium">{(detail.importance * 100).toFixed(1)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2.5">
                              <div 
                                className="bg-blue-600 h-2.5 rounded-full" 
                                style={{ width: `${detail.importance * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {showTour && (
        <ProductTour
          step={tourStep}
          onNext={handleNextTourStep}
          onComplete={handleTourComplete}
          darkMode={false}
        />
      )}
    </div>
  );
}