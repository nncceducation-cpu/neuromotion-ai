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
  if (!data || data.length === 0) return <div className="h-16 flex items-center justify-center text-xs text-neutral-300">No Data</div>;

  return (
    <div className="h-20 w-full mt-2">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`color-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={color} stopOpacity={0.15}/>
              <stop offset="95%" stopColor={color} stopOpacity={0}/>
            </linearGradient>
          </defs>
          <XAxis dataKey="timestamp" hide />
          <YAxis hide domain={['auto', 'auto']} />
          <RechartsTooltip
            contentStyle={{ backgroundColor: '#171717', border: 'none', borderRadius: '4px', color: '#fff', fontSize: '10px' }}
            itemStyle={{ color: '#fff' }}
            formatter={(value: number) => [value.toFixed(2), dataKey]}
            labelStyle={{ display: 'none' }}
          />
          <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#color-${dataKey})`} strokeWidth={1.5} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ReportView: React.FC<ReportViewProps> = ({ report, onClose, onSaveCorrection, userRole = 'Specialist' }) => {
  const [isCorrecting, setIsCorrecting] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState<FMCategory>(report.classification);
  const [notes, setNotes] = useState('');

  useEffect(() => {
    setSelectedCategory(report.classification);
    setIsCorrecting(false);
    setNotes('');
  }, [report]);

  const internalClassification = report.expertCorrection ? report.expertCorrection.correctClassification : report.classification;

  let displayLabel: string = internalClassification;
  if (internalClassification === 'Seizures') {
      displayLabel = "Possible Rhythmic activity detected";
  } else if (internalClassification === 'Sarnat Stage II' || internalClassification === 'Sarnat Stage III') {
      displayLabel = "Possible encephalopathy detected";
  }

  const isSeizure = internalClassification === 'Seizures';
  const displayIsNormal = internalClassification === 'Normal';

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

  return (
    <div className="animate-fade-in space-y-6 report-container">
      <style>{`
        @media print {
            .no-print { display: none !important; }
            header, nav, .pipeline-visualizer { display: none !important; }
            body { background-color: white; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
            main { padding-top: 0 !important; margin: 0 !important; }
            .report-container { margin: 0; padding: 0; }
        }
      `}</style>

      {/* Classification Header */}
      <div className="bg-white p-6 rounded-md border border-neutral-200">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
          <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                  {report.expertCorrection && (
                      <span className="bg-neutral-900 text-white text-[10px] px-2 py-0.5 rounded font-medium">
                          Expert Override
                      </span>
                  )}
                  <span className={`text-[10px] px-2 py-0.5 rounded font-medium ${
                    displayIsNormal ? 'bg-neutral-100 text-neutral-600' :
                    isSeizure ? 'bg-neutral-900 text-white' :
                    'bg-neutral-200 text-neutral-700'
                  }`}>
                    {internalClassification}
                  </span>
              </div>
              <h2 className="text-2xl font-semibold text-neutral-900 tracking-tight">
                  {displayLabel}
              </h2>

              {report.seizureDetected && report.seizureType !== 'None' && (
                  <div className="mt-2 inline-flex items-center bg-neutral-100 text-neutral-700 px-3 py-1 rounded text-xs font-medium">
                      {report.seizureType} Type
                  </div>
              )}
          </div>
          <div className="w-32 h-32">
               <ConfidenceGauge value={report.expertCorrection ? 100 : report.confidence} />
          </div>
        </div>
      </div>

      {/* Differential Diagnosis Alert */}
      {report.differentialAlert && (
         <div className="bg-neutral-50 border border-neutral-200 p-4 rounded-md flex items-start gap-3">
             <span className="text-neutral-400 mt-0.5 text-sm">i</span>
             <div>
                 <h4 className="font-medium text-neutral-900 text-sm">Differential Diagnosis</h4>
                 <p className="text-neutral-600 text-sm mt-0.5">{report.differentialAlert}</p>
             </div>
         </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 break-inside-avoid">

        {/* Raw Data Analysis */}
        <div className="bg-white p-6 rounded-md border border-neutral-200">
           <div className="flex items-center justify-between mb-6">
             <h3 className="text-sm font-semibold text-neutral-900">Raw Data Analysis</h3>
             <span className="text-[10px] text-neutral-400 font-mono">Physics Engine v6.0</span>
           </div>

           <div className="space-y-5">
             {/* Posture & Tone */}
             {posture && (
                <div className="space-y-3">
                    <div className="flex justify-between items-center">
                        <span className="text-xs font-medium text-neutral-700">Posture & Tone</span>
                        <span className={`text-[10px] font-medium px-2 py-0.5 rounded ${
                          posture.tone_label === 'Normal' ? 'bg-neutral-100 text-neutral-600' : 'bg-neutral-200 text-neutral-700'
                        }`}>
                            {posture.tone_label}
                        </span>
                    </div>

                    <div className="space-y-2">
                        <div>
                            <div className="flex justify-between text-[11px] text-neutral-500 mb-1">
                                <span>Shoulder Flexion</span>
                                <span className="font-mono">{Math.round(posture.shoulder_flexion_index * 100)}%</span>
                            </div>
                            <div className="w-full bg-neutral-100 rounded-full h-1">
                                <div className="bg-neutral-900 h-1 rounded-full transition-all" style={{ width: `${posture.shoulder_flexion_index * 100}%` }}></div>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-[11px] text-neutral-500 mb-1">
                                <span>Hip Flexion</span>
                                <span className="font-mono">{Math.round(posture.hip_flexion_index * 100)}%</span>
                            </div>
                            <div className="w-full bg-neutral-100 rounded-full h-1">
                                <div className="bg-neutral-900 h-1 rounded-full transition-all" style={{ width: `${posture.hip_flexion_index * 100}%` }}></div>
                            </div>
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                        <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100">
                            <div className="text-[10px] text-neutral-400 uppercase font-medium">Activity</div>
                            <div className="text-base font-semibold text-neutral-900 font-mono">{posture.spontaneous_activity ? posture.spontaneous_activity.toFixed(1) : 0}</div>
                        </div>
                        <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100">
                            <div className="text-[10px] text-neutral-400 uppercase font-medium">Frog Leg</div>
                            <div className="text-base font-semibold text-neutral-900 font-mono">{posture.frog_leg_score ? posture.frog_leg_score.toFixed(2) : 0}</div>
                        </div>
                        <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100">
                            <div className="text-[10px] text-neutral-400 uppercase font-medium">Crying</div>
                            <div className="text-base font-semibold text-neutral-900 font-mono">{posture.crying_index ? (posture.crying_index * 10).toFixed(1) : 0}</div>
                        </div>
                        <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100">
                            <div className="text-[10px] text-neutral-400 uppercase font-medium">Eye Open</div>
                            <div className="text-base font-semibold text-neutral-900 font-mono">{posture.eye_openness_index ? (posture.eye_openness_index * 10).toFixed(1) : 0}</div>
                        </div>
                    </div>
                </div>
             )}

             {/* Seizure Risk */}
             <div className="pt-4 border-t border-neutral-100">
                <div className="flex justify-between items-end mb-2">
                    <span className="text-xs font-medium text-neutral-700">Seizure Risk</span>
                    <span className="text-sm font-mono font-semibold text-neutral-900">
                        {Math.max(report.rawData.seizure.rhythmicity_score, report.rawData.seizure.stiffness_score).toFixed(2)}
                    </span>
                </div>
                <div className="w-full bg-neutral-100 rounded-full h-1.5">
                    <div
                        className="h-1.5 rounded-full bg-neutral-900 transition-all"
                        style={{ width: `${Math.min(100, Math.max(report.rawData.seizure.rhythmicity_score, report.rawData.seizure.stiffness_score) * 100)}%` }}
                    ></div>
                </div>

                <div className="grid grid-cols-2 gap-2 mt-3">
                    <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100 text-center">
                        <span className="text-[10px] text-neutral-400 uppercase font-medium block">Dom. Freq</span>
                        <span className="text-sm font-semibold text-neutral-900 font-mono">{report.rawData.seizure.dominant_frequency.toFixed(1)} Hz</span>
                    </div>
                    <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100 text-center">
                        <span className="text-[10px] text-neutral-400 uppercase font-medium block">Synchrony</span>
                        <span className="text-sm font-semibold text-neutral-900 font-mono">{report.rawData.seizure.limb_synchrony.toFixed(2)}</span>
                    </div>
                </div>

                <div className="flex justify-between text-[10px] text-neutral-400 mt-2 font-mono">
                     <span>Eye Dev: {report.rawData.seizure.eye_deviation_score.toFixed(2)}</span>
                     <span>Stiff: {report.rawData.seizure.stiffness_score.toFixed(2)}</span>
                </div>
             </div>

             {/* Physics Summary */}
             {report.rawData.avg_kinetic_energy > 0 && (
                <div className="pt-4 border-t border-neutral-100">
                    <div className="text-xs font-medium text-neutral-700 mb-2">Physics Analysis</div>
                    <div className="grid grid-cols-2 gap-2">
                        <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100">
                            <div className="text-[10px] uppercase text-neutral-400 font-medium">Kinetic Energy</div>
                            <div className="text-base font-semibold text-neutral-900 font-mono">{report.rawData.avg_kinetic_energy.toFixed(1)} <span className="text-xs text-neutral-400">J</span></div>
                        </div>
                        <div className="p-2.5 bg-neutral-50 rounded border border-neutral-100">
                            <div className="text-[10px] uppercase text-neutral-400 font-medium">Root Stress</div>
                            <div className="text-base font-semibold text-neutral-900 font-mono">{report.rawData.avg_root_stress.toFixed(2)}</div>
                        </div>
                    </div>
                </div>
             )}

             {/* Biomarker Mini Charts */}
             <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-4 border-t border-neutral-100">
                <div className="p-3 border border-neutral-100 rounded-md">
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="text-[11px] text-neutral-500">Entropy</div>
                            <div className="text-lg font-semibold text-neutral-900 font-mono">{report.rawData.entropy.toFixed(2)}</div>
                        </div>
                    </div>
                    {report.timelineData && <MiniChart data={report.timelineData} dataKey="entropy" color="#171717" />}
                </div>

                <div className="p-3 border border-neutral-100 rounded-md">
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="text-[11px] text-neutral-500">Fluency</div>
                            <div className="text-lg font-semibold text-neutral-900 font-mono">{report.rawData.fluency.toFixed(2)}</div>
                        </div>
                    </div>
                    {report.timelineData && <MiniChart data={report.timelineData} dataKey="fluency_jerk" color="#525252" />}
                </div>

                <div className="p-3 border border-neutral-100 rounded-md">
                    <div className="flex justify-between items-start">
                        <div>
                            <div className="text-[11px] text-neutral-500">Complexity</div>
                            <div className="text-lg font-semibold text-neutral-900 font-mono">{report.rawData.complexity.toFixed(2)}</div>
                        </div>
                    </div>
                    {report.timelineData && <MiniChart data={report.timelineData} dataKey="fractal_dim" color="#737373" />}
                </div>
             </div>
           </div>
        </div>

        {/* Recommendations */}
        <div className="bg-white p-6 rounded-md border border-neutral-200 flex flex-col">
          <div>
             <h4 className="font-semibold text-sm text-neutral-900 mb-4">Recommendations</h4>
             <ul className="space-y-2.5">
                {report.recommendations.map((rec, i) => (
                  <li key={i} className="flex items-start text-sm text-neutral-600 leading-relaxed">
                    <span className="text-neutral-300 mr-2.5 mt-1 text-xs">—</span>
                    {rec}
                  </li>
                ))}
             </ul>
          </div>
        </div>
      </div>

      {/* Expert Training Section */}
      {onSaveCorrection && (
        <div className="bg-white border border-neutral-200 rounded-md p-6 no-print">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-semibold text-sm text-neutral-900">Expert Review</h3>
                {!isCorrecting ? (
                    <button
                        onClick={() => setIsCorrecting(true)}
                        className="text-xs bg-white border border-neutral-200 text-neutral-600 px-3 py-1.5 rounded-md hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
                    >
                        Correct Diagnosis
                    </button>
                ) : (
                    <button
                         onClick={() => setIsCorrecting(false)}
                         className="text-xs text-neutral-400 hover:text-neutral-600"
                    >
                        Cancel
                    </button>
                )}
            </div>

            {isCorrecting && (
                <div className="bg-neutral-50 p-5 rounded-md border border-neutral-200 animate-fade-in">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                        <div>
                            <label className="block text-xs font-medium text-neutral-700 mb-1.5">Correct Diagnosis</label>
                            <select
                                className="w-full p-2.5 border border-neutral-200 rounded-md focus:ring-2 focus:ring-neutral-900 outline-none bg-white text-sm"
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
                            <label className="block text-xs font-medium text-neutral-700 mb-1.5">Clinical Reasoning</label>
                            <textarea
                                className="w-full p-2.5 border border-neutral-200 rounded-md focus:ring-2 focus:ring-neutral-900 outline-none text-sm h-24 bg-white"
                                placeholder="Explain why the AI was incorrect..."
                                value={notes}
                                onChange={(e) => setNotes(e.target.value)}
                            />
                        </div>
                    </div>
                    <div className="mt-4 flex justify-end">
                        <button
                            onClick={submitCorrection}
                            className="bg-neutral-900 hover:bg-neutral-800 text-white px-6 py-2.5 rounded-md font-medium text-sm transition-colors"
                        >
                            Save Correction
                        </button>
                    </div>
                </div>
            )}

             <p className="text-[11px] text-neutral-400 mt-3">
                Corrections are stored in the knowledge base. Matching patterns will apply your correction automatically.
            </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-3 pt-2 no-print">
        {onClose && (
            <button
                onClick={onClose}
                className="py-2.5 px-5 bg-white border border-neutral-200 text-neutral-600 rounded-md hover:bg-neutral-50 font-medium transition-colors text-sm"
            >
                Back
            </button>
        )}
        {report.timelineData && (
            <button
                onClick={handleExportCSV}
                className="py-2.5 px-5 bg-neutral-900 text-white rounded-md hover:bg-neutral-800 font-medium transition-colors text-sm"
            >
                Export CSV
            </button>
        )}
      </div>
    </div>
  );
};
