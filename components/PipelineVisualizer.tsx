import React from 'react';
import { PipelineStage } from '../types';
import { STAGES } from '../constants';

interface PipelineVisualizerProps {
  currentStage: PipelineStage;
}

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({ currentStage }) => {
  return (
    <div className="w-full py-8">
      <div className="flex justify-between items-center relative">
        {/* Progress Bar Background */}
        <div className="absolute top-1/2 left-0 w-full h-1 bg-slate-200 -z-10 transform -translate-y-1/2 rounded-full"></div>
        
        {/* Progress Bar Fill */}
        <div 
          className="absolute top-1/2 left-0 h-1 bg-sky-500 -z-10 transform -translate-y-1/2 rounded-full transition-all duration-500 ease-in-out"
          style={{ width: `${Math.max(0, (currentStage / (STAGES.length - 1)) * 100)}%` }}
        ></div>

        {STAGES.map((stage) => {
          const isActive = currentStage === stage.id;
          const isCompleted = currentStage > stage.id;
          const isPending = currentStage < stage.id;

          return (
            <div key={stage.id} className="flex flex-col items-center group relative w-32">
              <div 
                className={`
                  w-12 h-12 rounded-full flex items-center justify-center text-lg shadow-sm border-4 transition-all duration-300
                  ${isActive ? 'bg-sky-500 border-sky-200 text-white scale-110 shadow-sky-200' : ''}
                  ${isCompleted ? 'bg-green-500 border-green-200 text-white' : ''}
                  ${isPending ? 'bg-white border-slate-200 text-slate-300' : ''}
                `}
              >
                {isCompleted ? <i className="fas fa-check"></i> : <i className={`fas ${stage.icon}`}></i>}
              </div>
              
              <div className={`mt-3 text-center transition-colors duration-300 ${isActive ? 'text-sky-700 font-bold' : 'text-slate-500'}`}>
                <div className="text-sm font-medium">{stage.label}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};