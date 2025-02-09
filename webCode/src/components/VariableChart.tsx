import React, { useRef, useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import type { EconomicData } from '../types';
import { BeakerIcon } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface VariableChartProps {
  variable: string;
  historicalData: EconomicData[];
  prediction: number;
  quarterOffset: number;
  predictions: number[];
  onExperimentalValueChange?: (value: number) => void;
}

export function VariableChart({ 
  variable, 
  historicalData, 
  prediction, 
  quarterOffset,
  predictions,
  onExperimentalValueChange 
}: VariableChartProps) {
  const chartRef = useRef<any>();
  const [experimentalValue, setExperimentalValue] = useState<number | null>(null);
  const [inputValue, setInputValue] = useState('');

  const labels = historicalData.map((d, i) => {
    const quarterField = Object.keys(d).find(key => 
      key.toLowerCase().includes('quarter') || 
      key.toLowerCase().includes('date') ||
      key.toLowerCase().includes('period')
    );
    
    return quarterField ? String(d[quarterField]) : `Q${i + 1}`;
  });
  
  labels.push(`Q${labels.length + 1}`);

  const historicalValues = historicalData.map(d => {
    const value = Number(d[variable]);
    return isNaN(value) ? null : value;
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      const value = parseFloat(inputValue);
      if (!isNaN(value)) {
        setExperimentalValue(value);
        if (onExperimentalValueChange) {
          onExperimentalValueChange(value);
        }
      }
    }
  };

  const handleBlur = () => {
    const value = parseFloat(inputValue);
    if (!isNaN(value)) {
      setExperimentalValue(value);
      if (onExperimentalValueChange) {
        onExperimentalValueChange(value);
      }
    }
  };

  const clearExperiment = () => {
    setExperimentalValue(null);
    setInputValue('');
    if (onExperimentalValueChange) {
      onExperimentalValueChange(prediction);
    }
  };

  const lightBlueColor = 'rgb(59, 130, 246)'; // Tailwind blue-500
  const lightBlueColorTransparent = 'rgba(59, 130, 246, 0.5)';

  const data = {
    labels,
    datasets: [
      {
        label: 'Historical Values',
        data: [...historicalValues, null],
        borderColor: 'rgb(156, 163, 175)',
        backgroundColor: 'rgba(156, 163, 175, 0.5)',
        borderWidth: 2,
        pointRadius: 3,
        pointBackgroundColor: 'rgb(156, 163, 175)',
        pointBorderColor: 'rgb(156, 163, 175)',
        spanGaps: true,
      },
      {
        label: 'New Prediction',
        data: [...new Array(historicalValues.length).fill(null), prediction],
        borderColor: lightBlueColor,
        backgroundColor: lightBlueColorTransparent,
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: lightBlueColor,
        pointBorderColor: lightBlueColor,
        spanGaps: true,
      }
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: `${variable} Over Time`,
      },
      tooltip: {
        callbacks: {
          label: (context: any) => {
            const value = context.raw;
            if (value === null) return '';
            return `${context.dataset.label}: ${value.toFixed(2)}`;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: false,
        ticks: {
          callback: (value: number) => value.toLocaleString()
        }
      },
      x: {
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      }
    },
    elements: {
      line: {
        tension: 0.3
      }
    }
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow-md relative">
      <div className="absolute top-2 right-2 flex items-center gap-3">
        <BeakerIcon className="w-4 h-4 text-purple-500" />
        <div className="flex items-center gap-2" data-tour="beaker-input">
          <input
            type="number"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleInputKeyDown}
            onBlur={handleBlur}
            placeholder={prediction.toFixed(2)}
            className="w-24 px-2 py-1 text-sm border rounded focus:outline-none focus:ring-1 focus:ring-purple-500"
          />
          {experimentalValue !== null && (
            <button
              onClick={clearExperiment}
              className="text-xs text-gray-500 hover:text-gray-700"
            >
              Reset
            </button>
          )}
        </div>
      </div>
      
      <div>
        <Line ref={chartRef} options={options} data={data} />
      </div>
    </div>
  );
}