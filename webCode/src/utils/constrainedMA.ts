import { RandomForest } from './randomForestClassifier';

export interface ConstrainedMAResult {
  prediction: number;
  direction: 'up' | 'down';
  confidence: number;
  maChange: number;
}

export class ConstrainedMAModel {
  private period: number;
  private classifier: RandomForest;

  constructor(period: number = 3) {
    this.period = period;
    this.classifier = new RandomForest(10, 3, 5); // nTrees, maxDepth, minSamplesSplit
  }

  private calculateMA(data: number[]): number {
    const window = data.slice(-this.period);
    if (window.length === 0) return 0;
    return window.reduce((a, b) => a + b, 0) / window.length;
  }

  private calculateChanges(data: number[]): number[] {
    const changes: number[] = [];
    for (let i = 1; i < data.length; i++) {
      changes.push(data[i] - data[i - 1]);
    }
    return changes;
  }

  private createFeatures(data: number[], index: number): number[] {
    const features: number[] = [];
    
    // Recent changes
    const recentChanges = this.calculateChanges(data.slice(Math.max(0, index - this.period), index + 1));
    features.push(...recentChanges);
    
    // Moving averages of different periods
    [this.period, this.period * 2].forEach(period => {
      const slice = data.slice(Math.max(0, index - period), index + 1);
      features.push(this.calculateMA(slice));
    });
    
    // Volatility (standard deviation of changes)
    const changes = this.calculateChanges(data.slice(Math.max(0, index - this.period), index + 1));
    const mean = changes.reduce((a, b) => a + b, 0) / changes.length;
    const variance = changes.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / changes.length;
    features.push(Math.sqrt(variance));

    return features;
  }

  fit(data: number[]): void {
    if (data.length < this.period + 1) {
      throw new Error('Not enough data points');
    }

    const X: number[][] = [];
    const y: number[] = [];

    // Create training data
    for (let i = this.period; i < data.length - 1; i++) {
      const features = this.createFeatures(data, i);
      const actualChange = data[i + 1] - data[i];
      
      X.push(features);
      y.push(actualChange > 0 ? 1 : 0); // 1 for up, 0 for down
    }

    this.classifier.fit(X, y);
  }

  predict(data: number[]): ConstrainedMAResult {
    if (data.length < this.period) {
      throw new Error('Not enough data points');
    }

    const features = this.createFeatures(data, data.length - 1);
    const [upProb, downProb] = this.classifier.predictProba(features);
    const direction = upProb > downProb ? 'up' : 'down';
    const confidence = Math.max(upProb, downProb);

    // Calculate MA of recent changes
    const recentChanges = this.calculateChanges(data.slice(-this.period));
    const maChange = Math.abs(this.calculateMA(recentChanges));

    // Use the MA of changes as the magnitude, with direction from classifier
    const change = direction === 'up' ? maChange : -maChange;
    const prediction = data[data.length - 1] + change;

    return {
      prediction,
      direction,
      confidence,
      maChange
    };
  }
}

export function calculateConstrainedMA(
  data: number[],
  period: number = 3
): ConstrainedMAResult | null {
  try {
    const validData = data.filter(val => 
      typeof val === 'number' && !isNaN(val) && isFinite(val)
    );

    if (validData.length < period + 1) {
      console.warn('Not enough valid data points');
      return null;
    }

    const model = new ConstrainedMAModel(period);
    model.fit(validData);
    return model.predict(validData);
  } catch (error) {
    console.error('Error in constrained MA calculation:', error);
    return null;
  }
}