import React from 'react';
import { Upload, ArrowRight, Beaker, PlayCircle } from 'lucide-react';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  darkMode: boolean;
}

export function FileUpload({ onFileUpload, darkMode }: FileUploadProps) {
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.name.endsWith('.csv')) {
      onFileUpload(file);
    } else {
      alert('Please upload a valid CSV file');
    }
  };

  return (
    <div className="w-full max-w-md mx-auto">
      <div className="text-center mb-8">
        <h2 className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
          Quick Start
        </h2>
        <div className={`space-y-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          <div className="flex items-center gap-4 justify-center">
            <Upload className="w-6 h-6" />
            <ArrowRight className="w-4 h-4" />
            <p>Upload your CSV data</p>
          </div>
          <div className="flex items-center gap-4 justify-center">
            <PlayCircle className="w-6 h-6" />
            <ArrowRight className="w-4 h-4" />
            <p>Configure model parameters</p>
          </div>
          <div className="flex items-center gap-4 justify-center">
            <Beaker className="w-6 h-6" />
            <ArrowRight className="w-4 h-4" />
            <p>Run predictions and experiment</p>
          </div>
        </div>
      </div>

      <label
        htmlFor="file-upload"
        className={`flex flex-col items-center justify-center w-full h-48 border-2 border-dashed rounded-lg cursor-pointer transition-colors duration-200 ${
          darkMode 
            ? 'border-gray-600 bg-gray-800 hover:bg-gray-700' 
            : 'border-gray-300 bg-gray-50 hover:bg-gray-100'
        }`}
      >
        <div className="flex flex-col items-center justify-center pt-5 pb-6">
          <Upload className={`w-12 h-12 mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} />
          <p className={`mb-2 text-lg font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
            Click to upload or drag and drop
          </p>
          <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>CSV file only</p>
        </div>
        <input
          id="file-upload"
          type="file"
          accept=".csv"
          className="hidden"
          onChange={handleFileChange}
        />
      </label>
    </div>
  );
}