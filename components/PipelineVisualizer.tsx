import React from 'react';
import { PipelineStage } from '../types';
import { STAGES } from '../constants';

interface PipelineVisualizerProps {
  currentStage: PipelineStage;
}

export const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({ currentStage }) => {
  return (
    <div className="w-full py-6">
      <div className="flex items-center justify-between">
        {STAGES.map((stage, index) => {
          const isActive = currentStage === stage.id;
          const isCompleted = currentStage > stage.id;

          return (
            <React.Fragment key={stage.id}>
              {index > 0 && (
                <div className="flex-1 h-px mx-3">
                  <div className={`h-full ${isCompleted || isActive ? 'bg-neutral-900' : 'bg-neutral-200'} transition-colors duration-300`}></div>
                </div>
              )}
              <div className="flex flex-col items-center">
                <div
                  className={`
                    w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium transition-all duration-300
                    ${isActive ? 'bg-neutral-900 text-white ring-4 ring-neutral-200' : ''}
                    ${isCompleted ? 'bg-neutral-900 text-white' : ''}
                    ${!isActive && !isCompleted ? 'bg-white border border-neutral-200 text-neutral-400' : ''}
                  `}
                >
                  {isCompleted ? <i className="fas fa-check text-[10px]"></i> : index + 1}
                </div>
                <div className={`mt-2 text-center transition-colors duration-300 ${isActive ? 'text-neutral-900 font-medium' : 'text-neutral-400'}`}>
                  <div className="text-xs">{stage.label}</div>
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
