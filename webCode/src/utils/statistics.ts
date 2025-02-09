// Custom Random Forest Implementation
interface TreeNode {
  feature?: number;
  threshold?: number;
  value?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export interface TrainedForest {
  forest: RandomForest;
  inputVars: string[];
  importance: number[];
}

interface RandomForestResult {
  prediction: number;
  importance: number[];
  forest: RandomForest;
}

// Helper function to check if a value is valid (not NaN or Infinity)
function isValidNumber(value: number): boolean {
  return typeof value === 'number' && !isNaN(value) && isFinite(value);
}

// Helper function to clean data row
function cleanDataRow(row: number[]): number[] | null {
  if (row.some(val => !isValidNumber(val))) {
    return null;
  }
  return row;
}

class DecisionTree {
  private maxDepth: number;
  private minSamplesSplit: number;
  private feature?: number;
  private threshold?: number;
  private value?: number;
  private left?: DecisionTree;
  private right?: DecisionTree;

  constructor(maxDepth: number = 5, minSamplesSplit: number = 5) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  private calculateVariance(y: number[]): number {
    if (y.length === 0) return 0;
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    return y.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / y.length;
  }

  private findBestSplit(X: number[][], y: number[], availableFeatures: number[]): { feature: number; threshold: number; gain: number } | null {
    let bestGain = -Infinity;
    let bestFeature = 0;
    let bestThreshold = 0;

    // Only consider a random subset of features
    for (const feature of availableFeatures) {
      const values = Array.from(new Set(X.map(row => row[feature]))).sort((a, b) => a - b);
      
      for (let i = 0; i < values.length - 1; i++) {
        const threshold = (values[i] + values[i + 1]) / 2;
        const [leftY, rightY] = this.splitData(X, y, feature, threshold);
        
        if (leftY.length >= this.minSamplesSplit && rightY.length >= this.minSamplesSplit) {
          const gain = this.calculateGain(y, leftY, rightY);
          if (gain > bestGain) {
            bestGain = gain;
            bestFeature = feature;
            bestThreshold = threshold;
          }
        }
      }
    }

    if (bestGain === -Infinity) return null;
    return { feature: bestFeature, threshold: bestThreshold, gain: bestGain };
  }

  private calculateGain(parent: number[], leftChild: number[], rightChild: number[]): number {
    const parentVar = this.calculateVariance(parent);
    const leftVar = this.calculateVariance(leftChild);
    const rightVar = this.calculateVariance(rightChild);
    
    const leftWeight = leftChild.length / parent.length;
    const rightWeight = rightChild.length / parent.length;
    
    return parentVar - (leftWeight * leftVar + rightWeight * rightVar);
  }

  private splitData(X: number[][], y: number[], feature: number, threshold: number): [number[], number[]] {
    const leftY: number[] = [];
    const rightY: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftY.push(y[i]);
      } else {
        rightY.push(y[i]);
      }
    }

    return [leftY, rightY];
  }

  private splitDataWithX(X: number[][], y: number[], feature: number, threshold: number): [number[][], number[], number[][], number[]] {
    const leftX: number[][] = [];
    const leftY: number[] = [];
    const rightX: number[][] = [];
    const rightY: number[] = [];

    for (let i = 0; i < X.length; i++) {
      if (X[i][feature] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }

    return [leftX, leftY, rightX, rightY];
  }

  fit(X: number[][], y: number[], depth: number = 0, availableFeatures?: number[]): void {
    // Clean input data
    const validIndices = X.map((row, idx) => ({
      row,
      y: y[idx],
      valid: row.every(isValidNumber) && isValidNumber(y[idx])
    }))
    .filter(item => item.valid)
    .map(item => ({ row: item.row, y: item.y }));

    if (validIndices.length === 0) {
      this.value = 0; // Default value when no valid data
      return;
    }

    const cleanX = validIndices.map(item => item.row);
    const cleanY = validIndices.map(item => item.y);

    // Continue with original fit logic using cleaned data
    if (!availableFeatures) {
      availableFeatures = Array.from({ length: cleanX[0].length }, (_, i) => i);
    }

    if (depth >= this.maxDepth || 
        cleanY.length < this.minSamplesSplit || 
        new Set(cleanY).size === 1 ||
        availableFeatures.length === 0) {
      const weights = cleanY.map((_, i) => Math.exp(i / cleanY.length));
      const weightSum = weights.reduce((a, b) => a + b, 0);
      this.value = cleanY.reduce((acc, val, i) => acc + val * weights[i], 0) / weightSum;
      return;
    }

    const numFeatures = Math.max(1, Math.floor(Math.sqrt(availableFeatures.length)));
    const selectedFeatures = availableFeatures
      .sort(() => Math.random() - 0.5)
      .slice(0, numFeatures);

    const bestSplit = this.findBestSplit(cleanX, cleanY, selectedFeatures);
    if (!bestSplit) {
      this.value = cleanY.reduce((a, b) => a + b, 0) / cleanY.length;
      return;
    }

    this.feature = bestSplit.feature;
    this.threshold = bestSplit.threshold;

    const [leftX, leftY, rightX, rightY] = this.splitDataWithX(cleanX, cleanY, bestSplit.feature, bestSplit.threshold);

    const remainingFeatures = availableFeatures.filter(f => f !== bestSplit.feature);

    this.left = new DecisionTree(this.maxDepth, this.minSamplesSplit);
    this.right = new DecisionTree(this.maxDepth, this.minSamplesSplit);

    this.left.fit(leftX, leftY, depth + 1, remainingFeatures);
    this.right.fit(rightX, rightY, depth + 1, remainingFeatures);
  }

  predict(x: number[]): number {
    // Handle invalid input
    if (x.some(val => !isValidNumber(val))) {
      return 0; // Default prediction for invalid input
    }

    if (this.value !== undefined) return this.value;
    if (this.feature === undefined || this.threshold === undefined) throw new Error("Tree not fitted");
    
    if (x[this.feature] <= this.threshold) {
      return this.left!.predict(x);
    } else {
      return this.right!.predict(x);
    }
  }
}

class RandomForest {
  private trees: DecisionTree[];
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private featureImportances: number[];

  constructor(nEstimators: number = 10, maxDepth: number = 5, minSamplesSplit: number = 5) {
    this.trees = [];
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.featureImportances = [];
  }

  private bootstrapSample(X: number[][], y: number[]): [number[][], number[]] {
    const sampleX: number[][] = [];
    const sampleY: number[] = [];
    const n = X.length;
    
    // Use weighted sampling to favor recent data
    const weights = Array.from({ length: n }, (_, i) => Math.exp((i / n) * 2));
    const weightSum = weights.reduce((a, b) => a + b, 0);
    const normalizedWeights = weights.map(w => w / weightSum);
    
    for (let i = 0; i < n; i++) {
      let r = Math.random();
      let sum = 0;
      let idx = 0;
      
      for (let j = 0; j < n; j++) {
        sum += normalizedWeights[j];
        if (r <= sum) {
          idx = j;
          break;
        }
      }
      
      sampleX.push(X[idx]);
      sampleY.push(y[idx]);
    }
    
    return [sampleX, sampleY];
  }

  fit(X: number[][], y: number[]): void {
    this.trees = [];
    this.featureImportances = new Array(X[0].length).fill(0);

    for (let i = 0; i < this.nEstimators; i++) {
      const tree = new DecisionTree(this.maxDepth, this.minSamplesSplit);
      const [sampleX, sampleY] = this.bootstrapSample(X, y);
      tree.fit(sampleX, sampleY);
      this.trees.push(tree);
    }

    // Calculate feature importance
    for (let feature = 0; feature < X[0].length; feature++) {
      const originalPredictions = this.predictAll(X);
      const permutedX = X.map(row => [...row]);
      
      for (let i = permutedX.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [permutedX[i][feature], permutedX[j][feature]] = [permutedX[j][feature], permutedX[i][feature]];
      }
      
      const permutedPredictions = this.predictAll(permutedX);
      
      const originalVar = this.calculateVariance(originalPredictions);
      const permutedVar = this.calculateVariance(permutedPredictions);
      this.featureImportances[feature] = Math.max(0, (originalVar - permutedVar) / originalVar);
    }
    
    const sum = this.featureImportances.reduce((a, b) => a + b, 0);
    if (sum > 0) {
      this.featureImportances = this.featureImportances.map(imp => imp / sum);
    }
  }

  private calculateVariance(predictions: number[]): number {
    const mean = predictions.reduce((a, b) => a + b, 0) / predictions.length;
    return predictions.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / predictions.length;
  }

  predict(x: number[]): number {
    const predictions = this.trees.map(tree => tree.predict(x));
    
    // Use weighted average of tree predictions
    const weights = predictions.map((_, i) => 1 + Math.random() * 0.2); // Add some randomness
    const weightSum = weights.reduce((a, b) => a + b, 0);
    return predictions.reduce((acc, pred, i) => acc + pred * weights[i], 0) / weightSum;
  }

  private predictAll(X: number[][]): number[] {
    return X.map(x => this.predict(x));
  }

  getFeatureImportance(): number[] {
    return this.featureImportances;
  }
}

export function calculateRandomForest(
  data: number[][],
  inputVars: string[],
  nEstimators: number = 10,
  maxDepth: number = 3,
  minSamplesSplit: number = 5
): RandomForestResult | null {
  try {
    // Clean data more thoroughly
    const validData = data
      .map(row => cleanDataRow(row))
      .filter((row): row is number[] => row !== null);

    if (validData.length < 10) {
      console.warn('Not enough valid data points after cleaning');
      return null;
    }

    const X = validData.map(row => row.slice(0, -1));
    const y = validData.map(row => row[row.length - 1]);

    // Additional validation
    if (X.some(row => row.length !== inputVars.length)) {
      console.error('Inconsistent feature dimensions after cleaning');
      return null;
    }

    const rf = new RandomForest(nEstimators, maxDepth, minSamplesSplit);
    rf.fit(X, y);

    const importance = rf.getFeatureImportance();
    const lastX = validData[validData.length - 1].slice(0, -1);

    // Validate prediction
    const prediction = rf.predict(lastX);
    if (!isValidNumber(prediction)) {
      console.warn('Invalid prediction generated');
      return null;
    }

    return {
      prediction,
      importance,
      forest: rf
    };
  } catch (error) {
    console.error('Error in random forest calculation:', error);
    return null;
  }
}