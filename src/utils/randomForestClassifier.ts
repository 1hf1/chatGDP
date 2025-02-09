interface TreeNode {
  feature?: number;
  threshold?: number;
  value?: number[];
  left?: TreeNode;
  right?: TreeNode;
}

class DecisionTree {
  private maxDepth: number;
  private minSamplesSplit: number;
  private feature?: number;
  private threshold?: number;
  private value?: number[];
  private left?: DecisionTree;
  private right?: DecisionTree;

  constructor(maxDepth: number = 5, minSamplesSplit: number = 5) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  private calculateGini(y: number[]): number {
    const counts = new Map<number, number>();
    for (const val of y) {
      counts.set(val, (counts.get(val) || 0) + 1);
    }
    
    let gini = 1;
    for (const count of counts.values()) {
      const p = count / y.length;
      gini -= p * p;
    }
    return gini;
  }

  private findBestSplit(X: number[][], y: number[]): { feature: number; threshold: number; gain: number } | null {
    let bestGain = -Infinity;
    let bestFeature = 0;
    let bestThreshold = 0;

    for (let feature = 0; feature < X[0].length; feature++) {
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
    const parentGini = this.calculateGini(parent);
    const leftGini = this.calculateGini(leftChild);
    const rightGini = this.calculateGini(rightChild);
    
    const leftWeight = leftChild.length / parent.length;
    const rightWeight = rightChild.length / parent.length;
    
    return parentGini - (leftWeight * leftGini + rightWeight * rightGini);
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

  fit(X: number[][], y: number[], depth: number = 0): void {
    if (depth >= this.maxDepth || 
        y.length < this.minSamplesSplit || 
        new Set(y).size === 1) {
      const classes = Array.from(new Set(y));
      this.value = classes.map(c => 
        y.filter(val => val === c).length / y.length
      );
      return;
    }

    const bestSplit = this.findBestSplit(X, y);
    if (!bestSplit) {
      const classes = Array.from(new Set(y));
      this.value = classes.map(c => 
        y.filter(val => val === c).length / y.length
      );
      return;
    }

    this.feature = bestSplit.feature;
    this.threshold = bestSplit.threshold;

    const [leftX, leftY, rightX, rightY] = this.splitDataWithX(X, y, bestSplit.feature, bestSplit.threshold);

    this.left = new DecisionTree(this.maxDepth, this.minSamplesSplit);
    this.right = new DecisionTree(this.maxDepth, this.minSamplesSplit);

    this.left.fit(leftX, leftY, depth + 1);
    this.right.fit(rightX, rightY, depth + 1);
  }

  predict(x: number[]): number[] {
    if (this.value !== undefined) return this.value;
    if (this.feature === undefined || this.threshold === undefined) throw new Error("Tree not fitted");
    
    if (x[this.feature] <= this.threshold) {
      return this.left!.predict(x);
    } else {
      return this.right!.predict(x);
    }
  }
}

export class RandomForest {
  private trees: DecisionTree[];
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;

  constructor(nEstimators: number = 10, maxDepth: number = 5, minSamplesSplit: number = 5) {
    this.trees = [];
    this.nEstimators = nEstimators;
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
  }

  private bootstrapSample(X: number[][], y: number[]): [number[][], number[]] {
    const sampleX: number[][] = [];
    const sampleY: number[] = [];
    const n = X.length;
    
    for (let i = 0; i < n; i++) {
      const idx = Math.floor(Math.random() * n);
      sampleX.push(X[idx]);
      sampleY.push(y[idx]);
    }
    
    return [sampleX, sampleY];
  }

  fit(X: number[][], y: number[]): void {
    this.trees = [];

    for (let i = 0; i < this.nEstimators; i++) {
      const tree = new DecisionTree(this.maxDepth, this.minSamplesSplit);
      const [sampleX, sampleY] = this.bootstrapSample(X, y);
      tree.fit(sampleX, sampleY);
      this.trees.push(tree);
    }
  }

  predict(x: number[]): number {
    const predictions = this.trees.map(tree => {
      const probs = tree.predict(x);
      return probs.indexOf(Math.max(...probs));
    });
    
    // Majority vote
    const counts = new Map<number, number>();
    for (const pred of predictions) {
      counts.set(pred, (counts.get(pred) || 0) + 1);
    }
    
    let maxCount = -1;
    let maxClass = -1;
    for (const [cls, count] of counts.entries()) {
      if (count > maxCount) {
        maxCount = count;
        maxClass = cls;
      }
    }
    
    return maxClass;
  }

  predictProba(x: number[]): number[] {
    const predictions = this.trees.map(tree => tree.predict(x));
    const avgPredictions = predictions[0].map((_, i) => 
      predictions.reduce((sum, p) => sum + p[i], 0) / predictions.length
    );
    return avgPredictions;
  }
}