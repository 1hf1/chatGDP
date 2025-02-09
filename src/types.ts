export interface EconomicData {
  [key: string]: number;
}

export interface ForestDetail {
  inputVar: string;
  importance: number;
  prediction: number;
}

export interface PredictionResult {
  variable: string;
  prediction: number;
  details: ForestDetail[];
  targetVar?: string;
  modelType: 'randomForest' | 'autoregressive' | 'constrainedMA' | 'lstm';
  coefficients?: number[]; // For autoregressive model
  direction?: 'up' | 'down'; // For constrainedMA model
  confidence?: number; // For constrainedMA model
  maChange?: number; // For constrainedMA model
}

export type ModelType = 'randomForest' | 'autoregressive' | 'constrainedMA' | 'lstm';

export interface ModelConfiguration {
  selectedVariables: string[];
  nEstimators: number;
  maxDepth: number;
  minSamplesSplit: number;
  predictDifferences: boolean;
  modelType: ModelType;
  arLags?: number; // Number of lags for autoregressive model
  maPeriod?: number; // Period for constrained MA model
  lstmLookback?: number; // Lookback period for LSTM
}