import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingScreenProps {
  progress: {
    current: number;
    total: number;
  };
}

export function LoadingScreen({ progress }: LoadingScreenProps) {
  const percentage = progress.total > 0 ? Math.round((progress.current / progress.total) * 100) : 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-8 max-w-md w-full mx-4 text-center">
        <Loader2 className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-800 mb-2">
          Training Models
        </h3>
        <p className="text-gray-600 mb-4">
          Processing variable {progress.current} of {progress.total}
        </p>
        <div className="w-full bg-gray-200 rounded-full h-2.5 mb-2">
          <div
            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
            style={{ width: `${percentage}%` }}
          ></div>
        </div>
        <p className="text-sm text-gray-500">{percentage}% Complete</p>
      </div>
    </div>
  );
}