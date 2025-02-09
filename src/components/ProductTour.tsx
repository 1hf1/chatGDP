import React, { useEffect, useState } from 'react';

interface ProductTourProps {
  step: number;
  onNext: () => void;
  onComplete: () => void;
  darkMode: boolean;
  tourType: 'main' | 'simulation';
}

export function ProductTour({ step, onNext, onComplete, darkMode, tourType }: ProductTourProps) {
  const [targetRect, setTargetRect] = useState<DOMRect | null>(null);

  const getStepContent = () => {
    if (tourType === 'main') {
      switch (step) {
        case 1:
          return {
            selector: '[data-tour="about-tab"]',
            content: {
              question: "What's chatGDP all about?",
              answer: "Read up here! Learn about our mission to revolutionize economic modeling."
            },
            position: 'bottom'
          };
        case 2:
          return {
            selector: '[data-tour="implementation-tab"]',
            content: {
              question: "How was it made?",
              answer: "Glad you asked! Dive into the technical details of our models here."
            },
            position: 'bottom'
          };
        case 3:
          return {
            selector: '[data-tour="findings-tab"]',
            content: {
              question: "What can we learn from chatGDP? What's next?",
              answer: "That kind of stuff is right here!"
            },
            position: 'bottom'
          };
        case 4:
          return {
            selector: '[data-tour="try-tab"]',
            content: {
              question: "But can I actually use it?",
              answer: "Right here! Upload your data and start experimenting."
            },
            position: 'bottom'
          };
        case 5:
          return {
            selector: '[data-tour="theme-toggle"]',
            content: {
              question: "What kind of hackathon project uses light mode?",
              answer: "I was thinking the same thing. Click here to toggle!"
            },
            position: 'left'
          };
        default:
          return null;
      }
    } else {
      switch (step) {
        case 1:
          return {
            selector: '[data-tour="variable-expand"]',
            content: {
              question: "Want to see what's driving the predictions?",
              answer: "Click any variable to see detailed forecasts and understand which factors influence it the most."
            },
            position: 'bottom'
          };
        case 2:
          return {
            selector: '[data-tour="beaker-input"]',
            content: {
              question: "What if inflation suddenly spikes?",
              answer: "Use this experimental input to test different scenarios and see how they affect future predictions."
            },
            position: 'top'
          };
        case 3:
          return {
            selector: '[data-tour="next-quarter"]',
            content: {
              question: "Ready to see the future?",
              answer: "Click here to advance time and see how your predictions evolve. Models retrain every 4 quarters!"
            },
            position: 'left'
          };
        default:
          return null;
      }
    }
  };

  const stepContent = getStepContent();

  useEffect(() => {
    if (stepContent) {
      const element = document.querySelector(stepContent.selector);
      if (element) {
        setTargetRect(element.getBoundingClientRect());
      }
    }
  }, [step]);

  if (!stepContent || !targetRect) return null;

  const tooltipStyle = {
    position: 'fixed' as const,
    left: stepContent.position === 'left' 
      ? targetRect.left - 20 - 320 // width of tooltip + padding
      : targetRect.left,
    top: stepContent.position === 'top'
      ? targetRect.top - 20 - 100 // height of tooltip + padding
      : stepContent.position === 'bottom'
      ? targetRect.bottom + 20
      : targetRect.top,
    zIndex: 1000
  };

  return (
    <>
      {/* Dark overlay */}
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50" />
      
      {/* Highlight cutout */}
      <div
        className="fixed inset-0 z-50 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at center, transparent 0, rgba(0,0,0,0.5) 100%)',
        }}
      >
        <div
          className="absolute bg-transparent border-2 border-blue-500 shadow-lg"
          style={{
            left: targetRect.left - 4,
            top: targetRect.top - 4,
            width: targetRect.width + 8,
            height: targetRect.height + 8,
            borderRadius: '4px',
          }}
        />
      </div>

      {/* Tooltip */}
      <div
        className={`fixed z-[1000] p-4 rounded-lg shadow-lg w-80 ${
          darkMode ? 'bg-gray-800' : 'bg-white'
        }`}
        style={tooltipStyle}
      >
        <div className="space-y-2">
          <p className={`font-bold ${darkMode ? 'text-gray-100' : 'text-gray-900'}`}>
            {stepContent.content.question}
          </p>
          <p className="font-bold text-blue-600">
            {stepContent.content.answer}
          </p>
        </div>
        <div className="flex justify-between items-center mt-4">
          <div className="space-x-1">
            {Array.from({ length: tourType === 'main' ? 5 : 3 }).map((_, i) => (
              <span
                key={i}
                className={`inline-block w-2 h-2 rounded-full ${
                  i + 1 === step
                    ? 'bg-blue-500'
                    : darkMode
                    ? 'bg-gray-600'
                    : 'bg-gray-300'
                }`}
              />
            ))}
          </div>
          <button
            onClick={step === (tourType === 'main' ? 5 : 3) ? onComplete : onNext}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            {step === (tourType === 'main' ? 5 : 3) ? 'Get Started' : 'Next'}
          </button>
        </div>
      </div>
    </>
  );
}