import { isValidNumber } from './validation';

class AutoregressiveModel {
  private lags: number;
  private coefficients: number[];

  constructor(lags: number = 3) {
    this.lags = lags;
    this.coefficients = new Array(lags).fill(0);
  }

  private createLaggedData(data: number[]): [number[][], number[]] {
    const X: number[][] = [];
    const y: number[] = [];

    for (let i = this.lags; i < data.length; i++) {
      const laggedValues = data.slice(i - this.lags, i).reverse();
      if (laggedValues.every(isValidNumber)) {
        X.push(laggedValues);
        y.push(data[i]);
      }
    }

    return [X, y];
  }

  private normalEquation(X: number[][], y: number[]): number[] {
    // Add bias term
    const Xb = X.map(row => [1, ...row]);
    
    // Calculate X^T * X
    const XtX = Array(this.lags + 1).fill(0).map(() => Array(this.lags + 1).fill(0));
    for (let i = 0; i < this.lags + 1; i++) {
      for (let j = 0; j < this.lags + 1; j++) {
        XtX[i][j] = Xb.reduce((sum, row) => sum + row[i] * row[j], 0);
      }
    }
    
    // Calculate X^T * y
    const Xty = Array(this.lags + 1).fill(0);
    for (let i = 0; i < this.lags + 1; i++) {
      Xty[i] = Xb.reduce((sum, row, idx) => sum + row[i] * y[idx], 0);
    }
    
    // Solve using Gaussian elimination with regularization
    const lambda = 0.1; // Regularization parameter
    for (let i = 0; i < this.lags + 1; i++) {
      XtX[i][i] += lambda;
    }
    
    return this.solveLinearSystem(XtX, Xty);
  }

  private solveLinearSystem(A: number[][], b: number[]): number[] {
    const n = A.length;
    const augmented = A.map((row, i) => [...row, b[i]]);
    
    // Forward elimination
    for (let i = 0; i < n; i++) {
      let maxRow = i;
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(augmented[j][i]) > Math.abs(augmented[maxRow][i])) {
          maxRow = j;
        }
      }
      
      [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
      
      for (let j = i + 1; j < n; j++) {
        const factor = augmented[j][i] / augmented[i][i];
        for (let k = i; k <= n; k++) {
          augmented[j][k] -= factor * augmented[i][k];
        }
      }
    }
    
    // Back substitution
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) {
      let sum = augmented[i][n];
      for (let j = i + 1; j < n; j++) {
        sum -= augmented[i][j] * x[j];
      }
      x[i] = sum / augmented[i][i];
    }
    
    return x;
  }

  fit(data: number[]): void {
    if (data.length < this.lags + 1) {
      throw new Error('Not enough data points');
    }

    const [X, y] = this.createLaggedData(data);
    if (X.length === 0) {
      throw new Error('No valid training examples after creating lags');
    }

    this.coefficients = this.normalEquation(X, y);
  }

  predict(lastValues: number[]): number {
    if (lastValues.length !== this.lags) {
      throw new Error('Invalid number of input values');
    }

    if (!lastValues.every(isValidNumber)) {
      throw new Error('Invalid input values');
    }

    // Add bias term (1) and multiply with coefficients
    return this.coefficients[0] + lastValues.reduce(
      (sum, value, i) => sum + value * this.coefficients[i + 1],
      0
    );
  }

  getCoefficients(): number[] {
    return [...this.coefficients];
  }
}

export interface AutoregressiveResult {
  prediction: number;
  coefficients: number[];
}

export function calculateAutoregressive(
  data: number[],
  lags: number = 3
): AutoregressiveResult | null {
  try {
    // Clean data
    const validData = data.filter(isValidNumber);
    
    if (validData.length < lags + 1) {
      console.warn('Not enough valid data points');
      return null;
    }

    const model = new AutoregressiveModel(lags);
    model.fit(validData);

    const lastValues = validData.slice(-lags);
    const prediction = model.predict(lastValues);
    
    // Validate prediction
    if (!isValidNumber(prediction)) {
      console.warn('Invalid prediction generated');
      return null;
    }

    const coefficients = model.getCoefficients();
    if (coefficients.some(c => !isValidNumber(c))) {
      console.warn('Invalid coefficients generated');
      return null;
    }

    return {
      prediction,
      coefficients
    };
  } catch (error) {
    console.error('Error in autoregressive calculation:', error);
    return null;
  }
}