import React, { useState, useEffect } from 'react';
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, Tooltip as RechartsTooltip } from 'recharts';
import { AnalysisReport, FMCategory, ExpertCorrection, MovementMetrics } from '../types';
import { ConfidenceGauge } from './Charts';

interface ReportViewProps {
  report: AnalysisReport;
  onClose?: () => void;
  onSaveCorrection?: (correction: ExpertCorrection) => void;
  userRole?: string;
}

const MiniChart: React.FC<{ data: MovementMetrics[], dataKey: keyof MovementMetrics, color: string }> = ({ data, dataKey, color }) => {
  if (!data || data.length === 0) return <div className="h-16 flex items-center justify-center text-xs text-slate-300">No Data</div>;
  
  return (
    <div className="h-20 w-full mt-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`color-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.3}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis dataKey="timestamp" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <RechartsTooltip 
            contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '4px', color: '#fff', fontSize: '10px' }}
            itemStyle={{ color: '#fff' }}
            formatter={(value: number) => [value.toFixed(2), dataKey]}
            labelStyle={{ display: 'none' }}
          />
          <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#color-${dataKey})`} strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ReportView: React.FC<ReportViewProps> = ({ report, onClose, onSaveCorrection, userRole = 'Specialist' }) => {
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<FMCategory>(report.classification);
  const [notes, setNotes] = useState('');

  // Reset local state if report changes
  useEffect(() => {
    setSelectedCategory(report.classification);
    setIsCorrecting(false);
    setNotes('');
  }, [report]);

  // Determine which classification to use (AI or Expert)
  const internalClassification = report.expertCorrection ? report.expertCorrection.correctClassification : report.classification;
  
  // LOGIC: Map internal classification to Display Label
  let displayLabel: string = internalClassification;
  if (internalClassification === 'Seizures') {
      displayLabel = "Possible Rhythmic activity detected";
  } else if (internalClassification === 'Sarnat Stage II' || internalClassification === 'Sarnat Stage III') {
      displayLabel = "Possible encephalopathy detected";
  }

  // Status Logic (Uses Internal Classification for logic checks)
  const isSarnat = internalClassification.includes('Sarnat');
  const isSeizure = internalClassification === 'Seizures';
  const displayIsNormal = internalClassification === 'Normal';
  
  let statusColor = 'bg-amber-100 text-amber-800 border-amber-500'; // Default Warning

  if (displayIsNormal) {
      statusColor = 'bg-green-100 text-green-800 border-green-500';
  } else if (isSeizure) {
      statusColor = 'bg-red-100 text-red-800 border-red-500';
  } else if (isSarnat) {
      if (internalClassification === 'Sarnat Stage I') statusColor = 'bg-amber-100 text-amber-800 border-amber-500';
      else statusColor = 'bg-purple-100 text-purple-800 border-purple-500'; // Stage II & III
  }

  const handleExportCSV = () => {
    if (!report.timelineData) return;
    const headers = ['timestamp', 'entropy', 'fluency_velocity', 'fluency_jerk', 'fractal_dim', 'phase_x', 'phase_v', 'kinetic_energy', 'angular_jerk', 'root_stress'];
    const rows = report.timelineData.map(row => [
        row.timestamp,
        row.entropy.toFixed(4),
        row.fluency_velocity.toFixed(4),
        row.fluency_jerk.toFixed(4),
        row.fractal_dim.toFixed(4),
        row.phase_x.toFixed(4),
        row.phase_v.toFixed(4),
        row.kinetic_energy.toFixed(4),
        (row.angular_jerk ?? 0).toFixed(4),
        row.root_stress.toFixed(4)
    ].join(','));
    const csvContent = "data:text/csv;charset=utf-8," + headers.join(',') + "\n" + rows.join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `neuromotion_raw_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const submitCorrection = () => {
    if (onSaveCorrection) {
        onSaveCorrection({
            correctClassification: selectedCategory,
            notes: notes,
            timestamp: new Date().toISOString(),
            clinicianName: userRole
        });
    }
  };

  const posture = report.rawData.posture;
  const toneColor = posture?.tone_label === 'Normal' ? 'text-green-600 bg-green-50' : posture?.tone_label === 'Hypotonic' ? 'text-purple-600 bg-purple-50' : 'text-amber-600 bg-amber-50';

  return (
    <div className="animate-fade-in space-y-8 report-container">
      <style>{`
        @media print {
            .no-print { display: none !important; }
            header, nav, .pipeline-visualizer { display: none !important; } 
            body { background-color: white; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            main { padding-top: 0 !important; margin: 0 !important; }
            .report-container { margin: 0; padding: 0; }
            .bg-slate-50 { background-color: #f8fafc !important; }
            .bg-white { box-shadow: none !important; border: 1px solid #e2e8f0 !important; }
            .shadow-lg, .shadow-xl { box-shadow: none !important; }
        }
      `}</style>

      {/* RESULT TYPE 1: CATEGORICAL CLASSIFICATION */}
      <div className={`p-8 rounded-2xl border-l-8 shadow-lg bg-white flex flex-col md:flex-row items-center justify-between gap-6 ${statusColor.replace('bg-', 'border-')}`}>
        <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
                {report.expertCorrection && (
                    <span className="bg-purple-600 text-white text-xs px-2 py-0.5 rounded font-bold shadow-sm animate-pulse print:animate-none">
                        <i className="fas fa-brain mr-1"></i> Expert Memory Recall
                    </span>
                )}
            </div>
            <h2 className={`text-3xl font-extrabold ${displayIsNormal ? 'text-slate-800' : 'text-slate-900'} leading-tight`}>
                {displayLabel}
            </h2>
            
            {/* NEW: Seizure Details Sub-header */}
            {report.seizureDetected && report.seizureType !== 'None' && (
                <div className="mt-2 inline-flex items-center bg-red-100 text-red-700 px-3 py-1 rounded-full text-sm font-bold border border-red-200">
                    <i className="fas fa-bolt mr-2"></i>
                    {report.seizureType} Type Detected
                </div>
            )}
        </div>
        <div className="w-32 h-32">
             <ConfidenceGauge value={report.expertCorrection ? 100 : report.confidence} />
        </div>
      </div>
      
      {/* NEW: Differential Diagnosis Alert (Mimics) */}
      {report.differentialAlert && (
         <div className="bg-amber-50 border-l-4 border-amber-500 p-4 rounded-r-lg shadow-sm flex items-start gap-3">
             <i className="fas fa-exclamation-triangle text-amber-500 mt-1"></i>
             <div>
                 <h4 className="font-bold text-amber-800 text-sm">Differential Diagnosis Alert</h4>
                 <p className="text-amber-700 text-sm">{report.differentialAlert}</p>
             </div>
         </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 break-inside-avoid">
        
        {/* RESULT TYPE 2: RAW DATA SUMMARY (VISUALIZED) */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
           <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center justify-between">
             <span className="flex items-center"><i className="fas fa-microchip mr-2 text-purple-500"></i> Raw Data Analysis</span>
             <span className="text-xs bg-slate-100 text-slate-500 px-2 py-1 rounded">Physics Engine v6.0 (UE5 Algorithms)</span>
           </h3>
           
           <div className="space-y-6">
             {/* Posture & Tone Card */}
             {posture && (
                <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                    <div className="flex justify-between items-center mb-3">
                        <span className="text-sm font-bold text-slate-700">Posture & Tone</span>
                        <span className={`text-xs font-bold px-2 py-1 rounded border ${toneColor.replace('text', 'border')}`}>
                            {posture.tone_label}
                        </span>
                    </div>
                    
                    <div className="space-y-3">
                        <div>
                            <div className="flex justify-between text-xs text-slate-500 mb-1">
                                <span>Shoulder Flexion</span>
                                <span>{Math.round(posture.shoulder_flexion_index * 100)}%</span>
                            </div>
                            <div className="w-full bg-slate-200 rounded-full h-1.5">
                                <div className="bg-sky-500 h-1.5 rounded-full" style={{ width: `${posture.shoulder_flexion_index * 100}%` }}></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-xs text-slate-500 mb-1">
                                <span>Hip Flexion</span>
                                <span>{Math.round(posture.hip_flexion_index * 100)}%</span>
                            </div>
                            <div className="w-full bg-slate-200 rounded-full h-1.5">
                                <div className="bg-sky-500 h-1.5 rounded-full" style={{ width: `${posture.hip_flexion_index * 100}%` }}></div>
                            </div>
                        </div>
                        
                        {/* SARNAT METRICS DISPLAY */}
                        <div className="grid grid-cols-2 gap-2 mt-2 pt-2 border-t border-slate-200">
                            <div className="p-2 bg-white rounded border border-slate-100">
                                <div className="text-[10px] text-slate-400 uppercase font-bold">Activity Index</div>
                                <div className="text-lg font-bold text-slate-700">{posture.spontaneous_activity ? posture.spontaneous_activity.toFixed(1) : 0}</div>
                            </div>
                            <div className="p-2 bg-white rounded border border-slate-100">
                                <div className="text-[10px] text-slate-400 uppercase font-bold">Frog Leg Score</div>
                                <div className="text-lg font-bold text-slate-700">{posture.frog_leg_score ? posture.frog_leg_score.toFixed(2) : 0}</div>
                            </div>
                        </div>

                         {/* CONSCIOUSNESS & AROUSAL SECTION */}
                        <div className="grid grid-cols-2 gap-2 mt-1">
                            <div className="p-2 bg-white rounded border border-slate-100">
                                <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center gap-1">
                                    <i className="fas fa-bullhorn text-xs"></i> Crying Index
                                </div>
                                <div className="text-lg font-bold text-slate-700">{posture.crying_index ? (posture.crying_index * 10).toFixed(1) : 0}</div>
                            </div>
                            <div className="p-2 bg-white rounded border border-slate-100">
                                <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center gap-1">
                                    <i className="fas fa-eye text-xs"></i> Eye Openness
                                </div>
                                <div className="text-lg font-bold text-slate-700">{posture.eye_openness_index ? (posture.eye_openness_index * 10).toFixed(1) : 0}</div>
                            </div>
                        </div>
                    </div>
                </div>
             )}

             {/* Seizure Risk Score Bar */}
             <div className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                <div className="flex justify-between items-end mb-2">
                    <span className="text-sm font-bold text-slate-700">Seizure Risk (Spectral + Synchrony)</span>
                    <span className={`text-lg font-mono font-bold ${report.rawData.seizure.rhythmicity_score > 0.6 ? 'text-red-500' : 'text-green-500'}`}>
                        {Math.max(report.rawData.seizure.rhythmicity_score, report.rawData.seizure.stiffness_score).toFixed(2)}
                    </span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-2.5">
                    <div 
                        className={`h-2.5 rounded-full ${report.rawData.seizure.rhythmicity_score > 0.6 ? 'bg-red-500' : 'bg-green-500'}`} 
                        style={{ width: `${Math.min(100, Math.max(report.rawData.seizure.rhythmicity_score, report.rawData.seizure.stiffness_score) * 100)}%` }}
                    ></div>
                </div>
                
                {/* ADVANCED SEIZURE METRICS GRID */}
                <div className="grid grid-cols-2 gap-4 mt-3">
                    <div className="p-2 bg-white rounded border border-slate-100 flex flex-col items-center">
                        <span className="text-[10px] text-slate-400 uppercase font-bold">Dominant Freq</span>
                        <span className="text-lg font-bold text-slate-700">{report.rawData.seizure.dominant_frequency.toFixed(1)} <span className="text-xs text-slate-400">Hz</span></span>
                    </div>
                    <div className="p-2 bg-white rounded border border-slate-100 flex flex-col items-center">
                        <span className="text-[10px] text-slate-400 uppercase font-bold">Limb Synchrony</span>
                        <span className={`text-lg font-bold ${report.rawData.seizure.limb_synchrony > 0.7 ? 'text-red-500' : 'text-slate-700'}`}>
                            {report.rawData.seizure.limb_synchrony.toFixed(2)}
                        </span>
                    </div>
                </div>

                <div className="flex justify-between text-[10px] text-slate-400 mt-2">
                     <span>Eye Deviation: {report.rawData.seizure.eye_deviation_score.toFixed(2)}</span>
                     <span>Stiffness: {report.rawData.seizure.stiffness_score.toFixed(2)}</span>
                </div>
             </div>
             
             {/* NEW: UNREAL ENGINE PHYSICS SUMMARY */}
             {report.rawData.avg_kinetic_energy > 0 && (
                <div className="bg-yellow-50 p-4 rounded-lg border border-yellow-100">
                    <div className="text-sm font-bold text-yellow-800 mb-2 flex items-center">
                        <i className="fas fa-cubes mr-2"></i> UE5 Physics Analysis
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                        <div>
                            <div className="text-[10px] uppercase text-yellow-600 font-bold">Kinetic Energy</div>
                            <div className="text-lg font-bold text-yellow-900">{report.rawData.avg_kinetic_energy.toFixed(1)} <span className="text-xs">J</span></div>
                        </div>
                        <div>
                            <div className="text-[10px] uppercase text-yellow-600 font-bold">Root Stress</div>
                            <div className="text-lg font-bold text-yellow-900">{report.rawData.avg_root_stress.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
             )}

             <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Metric 1: Entropy */}
                <div className="p-3 border border-slate-100 rounded-lg shadow-sm">
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="text-xs text-slate-500 font-medium">Sample Entropy</div>
                            <div className="text-xl font-bold text-slate-800">{report.rawData.entropy.toFixed(2)}</div>
                        </div>
                        <i className="fas fa-random text-rose-400 opacity-50"></i>
                    </div>
                    {report.timelineData && <MiniChart data={report.timelineData} dataKey="entropy" color="#f43f5e" />}
                </div>

                {/* Metric 2: Fluency */}
                <div className="p-3 border border-slate-100 rounded-lg shadow-sm">
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="text-xs text-slate-500 font-medium">Fluency (Jerk)</div>
                            <div className="text-xl font-bold text-slate-800">{report.rawData.fluency.toFixed(2)}</div>
                        </div>
                        <i className="fas fa-water text-sky-400 opacity-50"></i>
                    </div>
                    {report.timelineData && <MiniChart data={report.timelineData} dataKey="fluency_jerk" color="#0ea5e9" />}
                </div>

                {/* Metric 3: Complexity */}
                <div className="p-3 border border-slate-100 rounded-lg shadow-sm">
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="text-xs text-slate-500 font-medium">Complexity (FD)</div>
                            <div className="text-xl font-bold text-slate-800">{report.rawData.complexity.toFixed(2)}</div>
                        </div>
                        <i className="fas fa-fingerprint text-emerald-400 opacity-50"></i>
                    </div>
                    {report.timelineData && <MiniChart data={report.timelineData} dataKey="fractal_dim" color="#10b981" />}
                </div>
             </div>
           </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 flex flex-col">
          <div>
             <h4 className="font-semibold text-slate-700 text-xs uppercase tracking-wide mb-3">Recommendations</h4>
             <ul className="space-y-2">
                {report.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start text-sm text-slate-600">
                    <i className="fas fa-check-circle text-emerald-500 mt-0.5 mr-2"></i>
                    {rec}
                  </li>
                ))}
             </ul>
          </div>
        </div>
      </div>
      
      {/* EXPERT TRAINING SECTION */}
      {onSaveCorrection && (
        <div className="bg-slate-50 border border-slate-200 rounded-xl p-6 ring-1 ring-slate-100 no-print">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-slate-700 flex items-center gap-2">
                    <i className="fas fa-graduation-cap text-slate-500"></i>
                    Expert Review & Model Training
                </h3>
                {!isCorrecting ? (
                    <button 
                        onClick={() => setIsCorrecting(true)}
                        className="text-sm bg-white border border-slate-300 text-slate-600 px-4 py-2 rounded-lg hover:text-sky-600 hover:border-sky-300 transition-colors shadow-sm"
                    >
                        Disagree with AI? Correct Diagnosis
                    </button>
                ) : (
                    <button 
                         onClick={() => setIsCorrecting(false)}
                         className="text-sm text-slate-400 hover:text-slate-600"
                    >
                        Cancel
                    </button>
                )}
            </div>
            
            {isCorrecting && (
                <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-lg animate-fade-in relative overflow-hidden">
                    <div className="absolute top-0 left-0 w-1 h-full bg-purple-500"></div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">Correct Diagnosis (Ground Truth)</label>
                            <select 
                                className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none bg-slate-50"
                                value={selectedCategory}
                                onChange={(e) => setSelectedCategory(e.target.value as FMCategory)}
                            >
                                <optgroup label="Assessment Categories">
                                    <option value="Normal">Normal</option>
                                    <option value="Sarnat Stage I">Sarnat Stage I (Mild)</option>
                                    <option value="Sarnat Stage II">Sarnat Stage II (Moderate)</option>
                                    <option value="Sarnat Stage III">Sarnat Stage III (Severe)</option>
                                    <option value="Seizures">Seizures</option>
                                </optgroup>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-bold text-slate-700 mb-2">Clinical Reasoning (Required for Training)</label>
                            <textarea 
                                className="w-full p-2.5 border border-slate-300 rounded-lg focus:ring-2 focus:ring-purple-500 outline-none text-sm h-24 bg-slate-50"
                                placeholder="Explain why the AI was incorrect. E.g., 'Tremors indicate Sarnat I, not Seizures.'"
                                value={notes}
                                onChange={(e) => setNotes(e.target.value)}
                            />
                        </div>
                    </div>
                    <div className="mt-6 flex justify-end">
                        <button 
                            onClick={submitCorrection}
                            className="bg-purple-600 hover:bg-purple-700 text-white px-8 py-3 rounded-lg font-bold shadow-md shadow-purple-200 transition-all transform active:scale-95 flex items-center"
                        >
                            <i className="fas fa-brain mr-2"></i> 
                            Teach AI & Save to Memory
                        </button>
                    </div>
                </div>
            )}
            
             <p className="text-xs text-slate-400 mt-2 flex items-center">
                <i className="fas fa-info-circle mr-1"></i>
                Corrections are added to the "Expert Knowledge Base". If the AI sees this data pattern again, it will strictly apply your correction.
            </p>
        </div>
      )}
      
      <div className="flex justify-end pt-4 no-print">
        {onClose && (
            <button 
                onClick={onClose}
                className="py-3 px-6 bg-white border border-slate-300 text-slate-600 rounded-lg hover:bg-slate-50 font-medium transition-colors mr-4"
            >
                Back to Dashboard
            </button>
        )}
        
        <div className="flex gap-3">
            {report.timelineData && (
                <button 
                    onClick={handleExportCSV}
                    className="py-3 px-6 bg-white border border-slate-300 text-slate-600 rounded-lg hover:bg-slate-50 hover:text-sky-600 font-medium transition-colors shadow-sm"
                >
                    <i className="fas fa-file-csv mr-2 text-green-600"></i> Export Raw Data (CSV)
                </button>
            )}
        </div>
      </div>
    </div>
  );
};