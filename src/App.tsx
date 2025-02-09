import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import { FileUpload } from './components/FileUpload';
import { ModelConfig } from './components/ModelConfig';
import { PredictionResults } from './components/PredictionResults';
import { LoadingScreen } from './components/LoadingScreen';
import { calculateRandomForest } from './utils/statistics';
import { calculateAutoregressive } from './utils/autoregressive';
import { calculateConstrainedMA } from './utils/constrainedMA';
import type { EconomicData, PredictionResult, ForestDetail, ModelConfiguration } from './types';
import { TrendingUp, Sun, Moon, Info, Code, PlayCircle, Lightbulb, ArrowRight } from 'lucide-react';
import { ProductTour } from './components/ProductTour';

function App() {
  const [predictions, setPredictions] = useState<PredictionResult[]>([]);
  const [rawData, setRawData] = useState<EconomicData[]>([]);
  const [variables, setVariables] = useState<string[]>([]);
  const [showConfig, setShowConfig] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [darkMode, setDarkMode] = useState(false);
  const [activeTab, setActiveTab] = useState<'about' | 'implementation' | 'findings' | 'try'>('about');
  const [showMainTour, setShowMainTour] = useState(true);
  const [mainTourStep, setMainTourStep] = useState(1);

  const handleNextMainTourStep = () => {
    setMainTourStep(prev => prev + 1);
  };

  const handleMainTourComplete = () => {
    setShowMainTour(false);
    localStorage.setItem('mainTourCompleted', 'true');
  };

  useEffect(() => {
    const mainTourCompleted = localStorage.getItem('mainTourCompleted');
    if (mainTourCompleted) {
      setShowMainTour(false);
    }
  }, []);

  const processData = async (config: ModelConfiguration) => {
    setIsLoading(true);
    setPredictions([]);
    const results: PredictionResult[] = [];
    
    const targetVars = config.selectedVariables;
    setProgress({ current: 0, total: targetVars.length });

    for (let i = 0; i < targetVars.length; i++) {
      const targetVar = targetVars[i];
      const result = await processVariable(targetVar, config);
      if (result) results.push(result);
      setProgress(prev => ({ ...prev, current: i + 1 }));
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    setPredictions(results);
    setShowConfig(false);
    setIsLoading(false);
  };

  const processVariable = async (targetVar: string, config: ModelConfiguration): Promise<PredictionResult | null> => {
    if (config.modelType === 'constrainedMA') {
      const data = rawData.map(row => Number(row[targetVar]));
      const result = calculateConstrainedMA(data, config.maPeriod);
      
      if (result) {
        return {
          variable: targetVar,
          prediction: result.prediction,
          details: [],
          modelType: 'constrainedMA',
          direction: result.direction,
          confidence: result.confidence,
          maChange: result.maChange
        };
      }
    } else if (config.modelType === 'randomForest') {
      const otherVars = config.selectedVariables.filter(v => v !== targetVar);
      if (otherVars.length === 0) return null;

      const forestData = rawData.slice(0, -1).map((row, index) => [
        ...otherVars.map(v => Number(row[v])),
        Number(rawData[index + 1][targetVar])
      ]);

      const result = calculateRandomForest(
        forestData,
        otherVars,
        config.nEstimators,
        config.maxDepth,
        config.minSamplesSplit
      );
      
      if (result) {
        const details: ForestDetail[] = otherVars.map((inputVar, idx) => ({
          inputVar,
          importance: result.importance[idx],
          prediction: result.prediction
        }));

        return {
          variable: targetVar,
          prediction: result.prediction,
          details: details.sort((a, b) => b.importance - a.importance),
          modelType: 'randomForest'
        };
      }
    } else {
      const data = rawData.map(row => Number(row[targetVar]));
      const result = calculateAutoregressive(data, config.arLags);
      
      if (result) {
        return {
          variable: targetVar,
          prediction: result.prediction,
          details: [],
          modelType: 'autoregressive',
          coefficients: result.coefficients
        };
      }
    }
    return null;
  };

  const handleFileUpload = (file: File) => {
    Papa.parse(file, {
      complete: (results) => {
        const data = results.data as EconomicData[];
        setRawData(data);
        setVariables(Object.keys(data[0]));
        setShowConfig(true);
        setActiveTab('try');
      },
      header: true,
      dynamicTyping: true,
    });
  };

  return (
    <div className={`min-h-screen transition-colors duration-200 ${darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'}`}>
      <div className="container mx-auto px-4 py-8">
        <header className="mb-12 text-center pt-6">
          <div className="flex justify-center items-center mb-4 relative">
            <h1 className="text-4xl font-bold flex items-center gap-3">
              <TrendingUp className="w-10 h-10 text-blue-600" />
              chatGDP
            </h1>
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`absolute right-0 p-2 rounded-full ${darkMode ? 'bg-gray-800 text-yellow-400' : 'bg-gray-100 text-gray-600'}`}
              data-tour="theme-toggle"
            >
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
          <p className={`text-lg max-w-2xl mx-auto ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Building foundation models for economic forecasting and scientific discovery
          </p>
        </header>

        <div className="flex justify-center mb-8">
          <nav className="inline-flex gap-4 border-b border-gray-200">
            <button
              onClick={() => setActiveTab('about')}
              className={`flex items-center gap-2 px-4 py-2 font-medium ${
                activeTab === 'about' 
                  ? 'border-b-2 border-blue-600 text-blue-600' 
                  : darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}
              data-tour="about-tab"
            >
              <Info className="w-4 h-4" /> About
            </button>
            <button
              onClick={() => setActiveTab('implementation')}
              className={`flex items-center gap-2 px-4 py-2 font-medium ${
                activeTab === 'implementation'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}
              data-tour="implementation-tab"
            >
              <Code className="w-4 h-4" /> Implementation
            </button>
            <button
              onClick={() => setActiveTab('findings')}
              className={`flex items-center gap-2 px-4 py-2 font-medium ${
                activeTab === 'findings'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}
              data-tour="findings-tab"
            >
              <Lightbulb className="w-4 h-4" /> Findings & Next Steps
            </button>
            <button
              onClick={() => setActiveTab('try')}
              className={`flex items-center gap-2 px-4 py-2 font-medium ${
                activeTab === 'try'
                  ? 'border-b-2 border-blue-600 text-blue-600'
                  : darkMode ? 'text-gray-400' : 'text-gray-600'
              }`}
              data-tour="try-tab"
            >
              <PlayCircle className="w-4 h-4" /> Try It
            </button>
          </nav>
        </div>

        <div className="max-w-4xl mx-auto">
          {activeTab === 'about' && (
            <div className={`prose ${darkMode ? 'prose-invert' : ''} max-w-none`}>
              <div className="space-y-8">
                <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
                  <h3 className={`text-lg font-semibold mb-2 ${darkMode ? 'text-white' : 'text-gray-900'}`}>TL;DR</h3>
                  <p className="leading-relaxed">
                    We need generative models of the economy because traditional economic models fail to capture complex, cascading effects of policy changes. 
                    By leveraging AI to model intricate economic relationships, we can better predict how changes ripple through communities, leading to more 
                    effective and equitable policy decisions that support entire economic ecosystems, not just primary stakeholders.
                  </p>
                </div>

                <article className="space-y-6">
                  <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    The Limitations of Current Policy Analysis
                  </h2>
                  <p>
                    Consider a steel mill affected by trade policy changes. Current programs, like the Trade Adjustment Assistance (TAA), 
                    focus on supporting displaced steel workers. But what about the restaurant server whose income depends on steel workers' 
                    lunch breaks? Or the local real estate agent whose market crashes when the community's primary employer downsizes? 
                    These secondary and tertiary effects often go unaccounted for in policy planning, leading to incomplete support systems 
                    and underestimated social costs.
                  </p>

                  <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    The Promise of Generative Economic Models
                  </h2>
                  <p>
                    This is where generative models of the economy could revolutionize policy design. These models would function as 
                    sophisticated next-state predictors, simulating not just direct economic impacts but entire chains of consequences 
                    across communities. By leveraging massive datasets about economic interconnections, employment patterns, and historical 
                    policy outcomes, these models could generate detailed predictions about how policy changes might cascade through local 
                    and regional economies.
                  </p>

                  <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    Beyond Simple Cause and Effect
                  </h2>
                  <p>
                    Traditional economic models often rely on simplified assumptions and linear relationships. But real economies are 
                    dynamic, adaptive systems where changes in one sector can trigger unexpected responses in others. Generative models 
                    could capture these complex dynamics by modeling multi-step causal chains, accounting for behavioral adaptations, 
                    identifying unexpected beneficiaries and victims of policy changes, and simulating long-term structural changes in 
                    local economies.
                  </p>

                  <h2 className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                    The Path to Better Policy Design
                  </h2>
                  <p>
                    Imagine running a proposed tariff policy through a generative model and receiving a detailed map of potential impacts: 
                    not just job losses at the steel mill, but predicted changes in local business revenues, housing prices, school 
                    enrollment, and even community health metrics. This comprehensive view would allow policymakers to design more nuanced, 
                    effective interventions that support entire communities through economic transitions.
                  </p>
                </article>
              </div>
            </div>
          )}

          {activeTab === 'implementation' && (
            <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg space-y-8`}>
              <h2 className={`text-2xl font-bold mb-6 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Model Architecture</h2>
              
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Random Forest Model</h3>
                <p className={`mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  Our random forest implementation uses an ensemble of decision trees to capture complex relationships between economic variables:
                </p>
                <ul className={`list-disc pl-6 space-y-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  <li>Multiple decision trees trained on bootstrapped samples</li>
                  <li>Feature importance calculation for interpretability</li>
                  <li>Adaptive depth based on data complexity</li>
                  <li>Weighted voting system for predictions</li>
                </ul>
              </div>

              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Autoregressive Model</h3>
                <p className={`mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  The autoregressive model specializes in time-series prediction:
                </p>
                <ul className={`list-disc pl-6 space-y-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  <li>Variable lag selection for temporal dependencies</li>
                  <li>Robust coefficient estimation</li>
                  <li>Adaptive to seasonal patterns</li>
                  <li>Error correction mechanisms</li>
                </ul>
              </div>

              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Constrained Moving Average Model</h3>
                <p className={`mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  A specialized model that combines traditional moving averages with machine learning:
                </p>
                <ul className={`list-disc pl-6 space-y-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  <li>Adaptive window size for trend detection</li>
                  <li>Confidence-based prediction intervals</li>
                  <li>Trend direction classification</li>
                  <li>Volatility-aware adjustments</li>
                </ul>
              </div>

              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
                <h3 className={`text-xl font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Data Sources</h3>
                <p className={`mb-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  The model is trained on data from the Federal Reserve Economic Data (FRED) API, provided by the Federal Reserve Bank of St. Louis. 
                  The dataset includes approximately 30 key economic indicators, such as:
                </p>
                <ul className={`list-disc pl-6 space-y-2 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  <li>GDP and its components</li>
                  <li>Employment statistics and labor market indicators</li>
                  <li>Inflation measures (CPI, PCE)</li>
                  <li>Interest rates and monetary policy metrics</li>
                  <li>Industrial production and capacity utilization</li>
                  <li>Housing market indicators</li>
                </ul>
                <p className={`mt-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  While our default dataset focuses on the US economy, the model architecture is designed to work with any economic dataset. 
                  Users can upload their own CSV files containing different variables and economies, and the models will adapt accordingly.
                </p>
              </div>
            </div>
          )}

          {activeTab === 'findings' && (
            <div className={`space-y-8 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                <h2 className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Key Findings</h2>
                <p className="mb-4">
                  The economy is remarkably stable. So far experiments I've run have caused some discontinuity, but then mean reversion happens. 
                  This suggests that no key economic indicators alone can cause dramatic changes in the economy, without a major shift in a few at once.
                </p>
                <p>
                  Simpler models are needed to model sets of broad economic variables, but more detailed models including individual firms and 
                  individuals may benefit from more advanced models like the use of transformers.
                </p>
              </div>

              <div className={`p-6 rounded-lg ${darkMode ? 'bg-gray-800' : 'bg-white'} shadow-lg`}>
                <h2 className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Next Steps</h2>
                <ul className="space-y-4">
                  <li className="flex items-start gap-2">
                    <ArrowRight className="w-5 h-5 mt-1 flex-shrink-0" />
                    <span>Finding more microeconomic data to model individuals and firms.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="w-5 h-5 mt-1 flex-shrink-0" />
                    <span>Setting up a GPU-connected, python backend to make it possible to quickly train models like the LSTM or transformers.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="w-5 h-5 mt-1 flex-shrink-0" />
                    <span>Run real experiments being carried out in the US right now, including the new tariff policy, and validate results.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <ArrowRight className="w-5 h-5 mt-1 flex-shrink-0" />
                    <span>
                      Implement auto-validation. Right now the project takes in any data, so its difficult to automatically validate results with 
                      backtesting. Early testing suggests that the constrained MA is probabilistically pretty accurate, but since different variables 
                      and models can be used we need a more robust way to actually test.
                    </span>
                  </li>
                </ul>
              </div>
            </div>
          )}

          {activeTab === 'try' && (
            <div className="space-y-8">
              {!showConfig && !predictions.length && (
                <FileUpload onFileUpload={handleFileUpload} darkMode={darkMode} />
              )}
              
              {showConfig && (
                <ModelConfig
                  variables={variables}
                  onConfigure={processData}
                  onCancel={() => setShowConfig(false)}
                  darkMode={darkMode}
                />
              )}
              
              {isLoading && (
                <LoadingScreen progress={progress} />
              )}
              
              {predictions.length > 0 && (
                <PredictionResults
                  results={predictions}
                  historicalData={rawData}
                  predictDifferences={false}
                />
              )}
            </div>
          )}
        </div>
      </div>

      {showMainTour && (
        <ProductTour
          step={mainTourStep}
          onNext={handleNextMainTourStep}
          onComplete={handleMainTourComplete}
          darkMode={darkMode}
          tourType="main"
        />
      )}
    </div>
  );
}

export default App;