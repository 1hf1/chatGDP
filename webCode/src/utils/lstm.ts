// Matrix operations for LSTM
class Matrix {
  rows: number;
  cols: number;
  data: number[][];

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = Array(rows).fill(0).map(() => Array(cols).fill(0));
  }

  static multiply(a: Matrix, b: Matrix): Matrix {
    if (a.cols !== b.rows) {
      throw new Error(`Matrix dimensions don't match for multiplication: ${a.rows}x${a.cols} and ${b.rows}x${b.cols}`);
    }
    const result = new Matrix(a.rows, b.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        let sum = 0;
        for (let k = 0; k < a.cols; k++) {
          sum += a.data[i][k] * b.data[k][j];
        }
        result.data[i][j] = sum;
      }
    }
    return result;
  }

  static add(a: Matrix, b: Matrix): Matrix {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new Error(`Matrix dimensions don't match for addition: ${a.rows}x${a.cols} and ${b.rows}x${b.cols}`);
    }
    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] + b.data[i][j];
      }
    }
    return result;
  }

  static elementwiseMultiply(a: Matrix, b: Matrix): Matrix {
    if (a.rows !== b.rows || a.cols !== b.cols) {
      throw new Error(`Matrix dimensions don't match for elementwise multiplication`);
    }
    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        result.data[i][j] = a.data[i][j] * b.data[i][j];
      }
    }
    return result;
  }

  static fromArray(arr: number[]): Matrix {
    if (!Array.isArray(arr)) {
      throw new Error('Input must be an array');
    }
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i][0] = arr[i];
    }
    return m;
  }

  toArray(): number[] {
    const arr: number[] = [];
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        arr.push(this.data[i][j]);
      }
    }
    return arr;
  }

  map(func: (x: number) => number): Matrix {
    const result = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        result.data[i][j] = func(this.data[i][j]);
      }
    }
    return result;
  }

  randomize(): void {
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        // Xavier initialization
        const limit = Math.sqrt(6 / (this.rows + this.cols));
        this.data[i][j] = Math.random() * 2 * limit - limit;
      }
    }
  }

  clone(): Matrix {
    const m = new Matrix(this.rows, this.cols);
    for (let i = 0; i < this.rows; i++) {
      for (let j = 0; j < this.cols; j++) {
        m.data[i][j] = this.data[i][j];
      }
    }
    return m;
  }
}

// Activation functions
const sigmoid = (x: number): number => 1 / (1 + Math.exp(-x));
const tanh = (x: number): number => Math.tanh(x);
const dsigmoid = (y: number): number => y * (1 - y);
const dtanh = (y: number): number => 1 - y * y;

class LSTMCell {
  inputSize: number;
  hiddenSize: number;
  
  // Gates weights and biases
  Wf: Matrix; bf: Matrix;  // Forget gate
  Wi: Matrix; bi: Matrix;  // Input gate
  Wc: Matrix; bc: Matrix;  // Cell state
  Wo: Matrix; bo: Matrix;  // Output gate
  
  // States
  cellState: Matrix;
  hiddenState: Matrix;
  
  // Learning rate
  learningRate: number;

  constructor(inputSize: number, hiddenSize: number) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.learningRate = 0.01;

    // Initialize weights with correct dimensions for concatenated input [x_t, h_{t-1}]
    const totalInputSize = inputSize + hiddenSize;
    
    this.Wf = new Matrix(hiddenSize, totalInputSize);
    this.Wi = new Matrix(hiddenSize, totalInputSize);
    this.Wc = new Matrix(hiddenSize, totalInputSize);
    this.Wo = new Matrix(hiddenSize, totalInputSize);
    
    this.bf = new Matrix(hiddenSize, 1);
    this.bi = new Matrix(hiddenSize, 1);
    this.bc = new Matrix(hiddenSize, 1);
    this.bo = new Matrix(hiddenSize, 1);

    // Randomize weights
    this.Wf.randomize();
    this.Wi.randomize();
    this.Wc.randomize();
    this.Wo.randomize();

    // Initialize states
    this.cellState = new Matrix(hiddenSize, 1);
    this.hiddenState = new Matrix(hiddenSize, 1);
  }

  forward(input: Matrix): { output: Matrix; cache: any } {
    // Concatenate input with previous hidden state
    const combinedInput = new Matrix(this.inputSize + this.hiddenSize, 1);
    for (let i = 0; i < this.inputSize; i++) {
      combinedInput.data[i][0] = input.data[i][0];
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      combinedInput.data[this.inputSize + i][0] = this.hiddenState.data[i][0];
    }

    // Gates computation with concatenated input
    const forgetGate = Matrix.multiply(this.Wf, combinedInput);
    Matrix.add(forgetGate, this.bf);
    const forgetActivation = forgetGate.map(sigmoid);

    const inputGate = Matrix.multiply(this.Wi, combinedInput);
    Matrix.add(inputGate, this.bi);
    const inputActivation = inputGate.map(sigmoid);

    const candidateCell = Matrix.multiply(this.Wc, combinedInput);
    Matrix.add(candidateCell, this.bc);
    const candidateActivation = candidateCell.map(tanh);

    const outputGate = Matrix.multiply(this.Wo, combinedInput);
    Matrix.add(outputGate, this.bo);
    const outputActivation = outputGate.map(sigmoid);

    // Update cell state
    const newCellState = Matrix.add(
      Matrix.elementwiseMultiply(forgetActivation, this.cellState),
      Matrix.elementwiseMultiply(inputActivation, candidateActivation)
    );

    // Update hidden state
    const newHiddenState = Matrix.elementwiseMultiply(
      outputActivation,
      newCellState.map(tanh)
    );

    // Update states
    this.cellState = newCellState;
    this.hiddenState = newHiddenState;

    return {
      output: this.hiddenState,
      cache: {
        combinedInput,
        forgetActivation,
        inputActivation,
        candidateActivation,
        outputActivation
      }
    };
  }

  backward(gradOutput: Matrix, cache: any): void {
    const { combinedInput, forgetActivation, inputActivation, candidateActivation, outputActivation } = cache;

    // Gradients for gates
    const dforgetGate = Matrix.elementwiseMultiply(gradOutput, forgetActivation.map(dsigmoid));
    const dinputGate = Matrix.elementwiseMultiply(gradOutput, inputActivation.map(dsigmoid));
    const dcandidateCell = Matrix.elementwiseMultiply(gradOutput, candidateActivation.map(dtanh));
    const doutputGate = Matrix.elementwiseMultiply(gradOutput, outputActivation.map(dsigmoid));

    // Update weights and biases with gradient clipping
    const clipValue = 5.0;
    const clip = (x: number) => Math.max(Math.min(x, clipValue), -clipValue);

    // Update weights
    this.Wf = Matrix.add(this.Wf, Matrix.multiply(dforgetGate, combinedInput.map(clip)));
    this.Wi = Matrix.add(this.Wi, Matrix.multiply(dinputGate, combinedInput.map(clip)));
    this.Wc = Matrix.add(this.Wc, Matrix.multiply(dcandidateCell, combinedInput.map(clip)));
    this.Wo = Matrix.add(this.Wo, Matrix.multiply(doutputGate, combinedInput.map(clip)));

    // Update biases
    this.bf = Matrix.add(this.bf, dforgetGate.map(clip));
    this.bi = Matrix.add(this.bi, dinputGate.map(clip));
    this.bc = Matrix.add(this.bc, dcandidateCell.map(clip));
    this.bo = Matrix.add(this.bo, doutputGate.map(clip));
  }
}

export class LSTM {
  private cells: LSTMCell[];
  private lookback: number;
  private hiddenSize: number;
  private mean: number;
  private std: number;
  private trainingProgress: number;
  private isTraining: boolean;

  constructor(lookback: number = 4, hiddenSize: number = 32) {
    this.lookback = lookback;
    this.hiddenSize = hiddenSize;
    this.cells = Array.from(
      { length: lookback },
      () => new LSTMCell(1, hiddenSize)
    );
    this.mean = 0;
    this.std = 1;
    this.trainingProgress = 0;
    this.isTraining = false;
  }

  private normalize(data: number[]): number[] {
    this.mean = data.reduce((a, b) => a + b, 0) / data.length;
    this.std = Math.sqrt(
      data.reduce((a, b) => a + Math.pow(b - this.mean, 2), 0) / data.length
    ) || 1;
    return data.map(x => (x - this.mean) / this.std);
  }

  private denormalize(value: number): number {
    return value * this.std + this.mean;
  }

  private createSequences(data: number[]): number[][] {
    const sequences: number[][] = [];
    for (let i = this.lookback; i < data.length; i++) {
      sequences.push(data.slice(i - this.lookback, i + 1));
    }
    return sequences;
  }

  async fit(data: number[]): Promise<void> {
    if (this.isTraining) {
      console.warn('LSTM is already training');
      return;
    }

    this.isTraining = true;
    this.trainingProgress = 0;
    console.log('Starting LSTM training...');

    try {
      const normalizedData = this.normalize(data);
      const sequences = this.createSequences(normalizedData);
      const epochs = 100;
      const batchSize = 32;
      
      for (let epoch = 0; epoch < epochs; epoch++) {
        let epochLoss = 0;
        const batchCount = Math.ceil(sequences.length / batchSize);
        
        for (let batchIndex = 0; batchIndex < sequences.length; batchIndex += batchSize) {
          const batch = sequences.slice(batchIndex, batchIndex + batchSize);
          let batchLoss = 0;

          for (const sequence of batch) {
            const input = sequence.slice(0, -1);
            const target = sequence[sequence.length - 1];
            
            // Forward pass
            let currentState = Matrix.fromArray([input[0]]);
            const caches = [];
            
            for (let i = 0; i < this.cells.length; i++) {
              const { output, cache } = this.cells[i].forward(currentState);
              currentState = output;
              caches.push(cache);
            }
            
            // Calculate loss
            const prediction = currentState.data[0][0];
            const loss = Math.pow(prediction - target, 2);
            batchLoss += loss;
            
            // Backward pass
            let gradOutput = new Matrix(this.hiddenSize, 1);
            gradOutput.data[0][0] = 2 * (prediction - target);
            
            for (let i = this.cells.length - 1; i >= 0; i--) {
              this.cells[i].backward(gradOutput, caches[i]);
            }
          }

          epochLoss += batchLoss / batch.length;
          
          // Update progress
          const totalSteps = epochs * batchCount;
          const currentStep = epoch * batchCount + Math.floor(batchIndex / batchSize);
          this.trainingProgress = (currentStep / totalSteps) * 100;
        }

        // Log progress
        if ((epoch + 1) % 10 === 0) {
          console.log(
            `Epoch ${epoch + 1}/${epochs}:`,
            `Loss: ${(epochLoss / batchCount).toFixed(6)}`,
            `Progress: ${this.trainingProgress.toFixed(1)}%`
          );
        }

        // Add small delay to prevent UI blocking
        await new Promise(resolve => setTimeout(resolve, 10));
      }

      console.log('LSTM training completed successfully');
    } catch (error) {
      console.error('Error during LSTM training:', error);
      throw error;
    } finally {
      this.isTraining = false;
    }
  }

  predict(data: number[]): number | null {
    try {
      const validData = data.slice(-this.lookback);
      if (validData.length < this.lookback) return null;

      const normalizedData = this.normalize(validData);
      let currentState = Matrix.fromArray([normalizedData[0]]);

      // Forward pass through all cells
      for (let i = 0; i < this.cells.length; i++) {
        const { output } = this.cells[i].forward(currentState);
        currentState = output;
      }

      return this.denormalize(currentState.data[0][0]);
    } catch (error) {
      console.error('Error in LSTM prediction:', error);
      return null;
    }
  }
}