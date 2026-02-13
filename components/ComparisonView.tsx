
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Legend, BarChart, Bar } from 'recharts';
import { MovementMetrics, ComparisonDataset, SavedReport } from '../types';
import { SERVER_URL } from '../constants';

interface ComparisonViewProps {
  onBack: () => void;
  initialReports?: SavedReport[];
}

const chartTooltipStyle = { backgroundColor: '#171717', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '11px' };

export const ComparisonView: React.FC<ComparisonViewProps> = ({ onBack, initialReports }) => {
  const [datasets, setDatasets] = useState<ComparisonDataset[]>([]);
  const [report, setReport] = useState<string | null>(null);

  const [aiReport, setAiReport] = useState<string>("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'ai', text: string }[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const parseCSV = async (file: File): Promise<MovementMetrics[]> => {
    const text = await file.text();
    let content = text;
    if (text.startsWith("data:text/csv")) {
        content = decodeURI(text.split(',')[1]);
    }

    const lines = content.split('\n').filter(l => l.trim() !== '');
    const headers = lines[0].split(',').map(h => h.trim());

    return lines.slice(1).map(line => {
        const values = line.split(',');
        const obj: any = {};
        headers.forEach((h, i) => {
            const val = parseFloat(values[i]);
            obj[h] = isNaN(val) ? 0 : val;
        });
        return obj as MovementMetrics;
    });
  };

  const calculateStats = (data: MovementMetrics[]) => {
      const metrics = ['entropy', 'fluency_velocity', 'fluency_jerk', 'fractal_dim', 'kinetic_energy', 'root_stress'];
      const stats: Record<string, any> = {};

      metrics.forEach(key => {
          const values = data.map(d => (d as any)[key] || 0);
          const n = values.length;
          if (n === 0) {
              stats[key] = { mean: 0, min: 0, max: 0, std: 0 };
              return;
          }
          const mean = values.reduce((a,b)=>a+b,0) / n;
          const variance = values.reduce((a,b)=>a+Math.pow(b-mean,2),0) / n;
          const std = Math.sqrt(variance);
          stats[key] = { mean, min: Math.min(...values), max: Math.max(...values), std };
      });
      return stats;
  };

  const generateAutomatedReport = async (currentDatasets: ComparisonDataset[]) => {
      if (currentDatasets.length === 0) return;

      try {
          const payload = currentDatasets.map(d => ({
              label: d.label,
              stats: d.stats
          }));
          const res = await fetch(`${SERVER_URL}/compare/automated_report`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ datasets: payload })
          });
          if (res.ok) {
              const data = await res.json();
              setReport(data.report);
          }
      } catch (e) {
          console.error("Failed to generate automated report", e);
      }
  };

  const getComparisonSummary = () => {
    return datasets.map(d =>
      `- ${d.label}: Entropy=${d.stats?.entropy.mean.toFixed(2)}, Energy=${d.stats?.kinetic_energy?.mean.toFixed(2) || 0}, Stress=${d.stats?.root_stress?.mean.toFixed(2) || 0}, Jerk=${d.stats?.fluency_jerk.mean.toFixed(2)}`
    ).join('\n');
  };

  const handleGenerateAIReport = async () => {
    if (datasets.length < 1) return;
    setIsGeneratingReport(true);

    try {
        const res = await fetch(`${SERVER_URL}/compare/ai_report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_summaries: getComparisonSummary() })
        });
        if (res.ok) {
            const data = await res.json();
            setAiReport(data.report || "No analysis generated.");
        } else {
            setAiReport("Failed to generate analysis. Please check API connection.");
        }
    } catch (e) {
        console.error("AI Error", e);
        setAiReport("Failed to generate analysis. Please check API connection.");
    } finally {
        setIsGeneratingReport(false);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || datasets.length === 0) return;

    const userMsg = chatInput;
    setChatInput("");
    setChatHistory(prev => [...prev, { role: 'user', text: userMsg }]);
    setIsChatLoading(true);

    try {
        const res = await fetch(`${SERVER_URL}/compare/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: userMsg,
                dataset_summaries: getComparisonSummary()
            })
        });
        if (res.ok) {
            const data = await res.json();
            setChatHistory(prev => [...prev, { role: 'ai', text: data.response || "I couldn't generate a response." }]);
        } else {
            setChatHistory(prev => [...prev, { role: 'ai', text: "Error connecting to AI service." }]);
        }
    } catch (err) {
        setChatHistory(prev => [...prev, { role: 'ai', text: "Error connecting to AI service." }]);
    } finally {
        setIsChatLoading(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  useEffect(() => {
    if (initialReports && initialReports.length > 0) {
        const loaded: ComparisonDataset[] = [];
        initialReports.forEach(r => {
            if (r.timelineData && r.timelineData.length > 0) {
                loaded.push({
                    label: r.videoName || 'Untitled',
                    data: r.timelineData,
                    stats: calculateStats(r.timelineData)
                });
            }
        });
        setDatasets(loaded);
        generateAutomatedReport(loaded);
    }
  }, [initialReports]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
          const newDatasets: ComparisonDataset[] = [];
          for (let i = 0; i < e.target.files.length; i++) {
              const file = e.target.files[i];
              try {
                  const data = await parseCSV(file);
                  const stats = calculateStats(data);
                  newDatasets.push({
                      label: file.name.replace('.csv', '').replace('neuromotion_raw_', ''),
                      data,
                      stats
                  });
              } catch (err) {
                  console.error(`Error parsing ${file.name}`, err);
              }
          }
          const updated = [...datasets, ...newDatasets];
          setDatasets(updated);
          generateAutomatedReport(updated);
      }
  };

  const colors = ['#171717', '#737373', '#a3a3a3', '#d4d4d4', '#404040'];

  return (
    <div className="animate-fade-in space-y-6">
      <style>{`
        @media print {
            .no-print { display: none !important; }
            body * { visibility: hidden; }
            #printable-report, #printable-report * { visibility: visible; }
            #printable-report { position: absolute; left: 0; top: 0; width: 100%; margin: 0; padding: 20px; border: none; box-shadow: none; }
            html, body { overflow: visible !important; height: auto !important; }
        }
      `}</style>

      {/* Header */}
      <div className="flex justify-between items-center no-print">
         <div className="flex items-center gap-4">
            <button onClick={onBack} className="text-neutral-400 hover:text-neutral-900 transition-colors text-sm">
                <i className="fas fa-arrow-left mr-2"></i>Back
            </button>
            <h2 className="text-lg font-semibold text-neutral-900">Comparative Analysis</h2>
         </div>
         <label className="bg-neutral-900 hover:bg-neutral-800 text-white px-4 py-2 rounded-md cursor-pointer transition-colors text-sm font-medium flex items-center gap-2">
            <i className="fas fa-plus text-xs"></i> Add CSV
            <input type="file" multiple accept=".csv" className="hidden" onChange={handleFileUpload} />
         </label>
      </div>

      {datasets.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 bg-white border border-dashed border-neutral-300 rounded-md text-neutral-400">
              <p className="text-sm mb-1">No datasets loaded</p>
              <p className="text-xs">Upload CSV files or select reports from the Dashboard to compare.</p>
          </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Time Series */}
            <div className="bg-white p-5 rounded-md border border-neutral-200 col-span-1 lg:col-span-2">
                <h3 className="text-sm font-medium text-neutral-900 mb-4">Entropy Time Series</h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                            <XAxis dataKey="timestamp" type="number" allowDuplicatedCategory={false} hide />
                            <YAxis domain={[0, 1.2]} tick={{ fontSize: 11 }} stroke="#a3a3a3" />
                            <Tooltip contentStyle={chartTooltipStyle} />
                            <Legend />
                            {datasets.map((ds, i) => (
                                <Line
                                    key={ds.label}
                                    data={ds.data}
                                    dataKey="entropy"
                                    name={ds.label}
                                    stroke={colors[i % colors.length]}
                                    dot={false}
                                    strokeWidth={1.5}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* AI Section */}
            <div className="col-span-1 lg:col-span-2 grid grid-cols-1 lg:grid-cols-3 gap-6">

                {/* AI Report */}
                <div className="lg:col-span-2 bg-white p-5 rounded-md border border-neutral-200 flex flex-col">
                     <div className="flex justify-between items-center mb-4">
                        <h3 className="text-sm font-medium text-neutral-900">AI Analysis</h3>
                        <button
                          onClick={handleGenerateAIReport}
                          disabled={isGeneratingReport}
                          className="px-3 py-1.5 bg-neutral-900 hover:bg-neutral-800 disabled:opacity-50 text-white rounded-md text-xs font-medium transition-colors"
                        >
                          {isGeneratingReport ? "Analyzing..." : "Generate"}
                        </button>
                    </div>

                    <div className="bg-neutral-50 rounded-md p-5 min-h-[200px] border border-neutral-100 flex-1">
                        {aiReport ? (
                          <div className="whitespace-pre-line leading-relaxed text-sm text-neutral-700">
                            {aiReport}
                          </div>
                        ) : (
                          <div className="h-full flex items-center justify-center text-neutral-400 text-sm">
                            Click "Generate" to analyze your data with AI.
                          </div>
                        )}
                    </div>
                </div>

                {/* Chat */}
                <div className="bg-white p-5 rounded-md border border-neutral-200 flex flex-col h-[400px]">
                    <h3 className="text-sm font-medium text-neutral-900 mb-4">Ask AI</h3>

                    <div className="flex-1 overflow-y-auto mb-3 space-y-2.5 pr-1">
                        {chatHistory.length === 0 && (
                          <div className="text-center text-neutral-400 text-xs mt-10">
                             Ask about stress levels, energy output, or stability differences...
                          </div>
                        )}
                        {chatHistory.map((msg, i) => (
                          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] rounded-md p-2.5 text-sm ${
                              msg.role === 'user'
                                ? 'bg-neutral-900 text-white'
                                : 'bg-neutral-100 text-neutral-700'
                            }`}>
                              {msg.text}
                            </div>
                          </div>
                        ))}
                        {isChatLoading && (
                           <div className="flex justify-start">
                             <div className="bg-neutral-100 rounded-md p-2.5">
                               <i className="fas fa-circle-notch fa-spin text-neutral-400 text-xs"></i>
                             </div>
                           </div>
                        )}
                        <div ref={chatEndRef} />
                    </div>

                    <form onSubmit={handleChatSubmit} className="relative">
                        <input
                          type="text"
                          value={chatInput}
                          onChange={(e) => setChatInput(e.target.value)}
                          placeholder="Ask about your data..."
                          className="w-full bg-neutral-50 border border-neutral-200 rounded-md py-2.5 pl-3 pr-10 text-sm focus:outline-none focus:ring-2 focus:ring-neutral-900/20 transition-colors text-neutral-800 placeholder-neutral-400"
                        />
                        <button
                          type="submit"
                          disabled={!chatInput.trim() || isChatLoading}
                          className="absolute right-2 top-1.5 p-1.5 bg-neutral-900 hover:bg-neutral-800 disabled:opacity-30 text-white rounded-md transition-colors w-7 h-7 flex items-center justify-center"
                        >
                          <i className="fas fa-arrow-up text-[10px]"></i>
                        </button>
                    </form>
                </div>
            </div>

            {/* Phase Space */}
            <div className="bg-white p-5 rounded-md border border-neutral-200">
                <h3 className="text-sm font-medium text-neutral-900 mb-4">Phase Space Overlay</h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
                            <XAxis type="number" dataKey="phase_x" name="Pos" tick={{ fontSize: 11 }} stroke="#a3a3a3" />
                            <YAxis type="number" dataKey="phase_v" name="Vel" tick={{ fontSize: 11 }} stroke="#a3a3a3" />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Legend />
                            {datasets.map((ds, i) => (
                                <Scatter
                                    key={ds.label}
                                    name={ds.label}
                                    data={ds.data}
                                    fill={colors[i % colors.length]}
                                    opacity={0.5}
                                />
                            ))}
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Bar Chart */}
            <div className="bg-white p-5 rounded-md border border-neutral-200">
                 <h3 className="text-sm font-medium text-neutral-900 mb-4">Mean Velocity</h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={datasets.map(d => ({ name: d.label, velocity: d.stats?.fluency_velocity.mean || 0 }))}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="#a3a3a3" />
                            <YAxis tick={{ fontSize: 11 }} stroke="#a3a3a3" />
                            <Tooltip cursor={{fill: 'transparent'}} contentStyle={chartTooltipStyle} />
                            <Bar dataKey="velocity" fill="#171717" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Report & Stats Table */}
            <div id="printable-report" className="bg-white p-5 rounded-md border border-neutral-200 col-span-1 lg:col-span-2">
                <div className="flex justify-between items-center mb-4 no-print">
                     <h3 className="text-sm font-medium text-neutral-900">Automated Report</h3>
                    <button
                        onClick={() => window.print()}
                        className="bg-neutral-900 hover:bg-neutral-800 text-white px-4 py-2 rounded-md transition-colors text-xs font-medium no-print"
                    >
                        Export PDF
                    </button>
                </div>

                {report && (
                    <div className="mb-6 p-5 bg-neutral-50 rounded-md border border-neutral-100 overflow-hidden">
                         <pre className="whitespace-pre-wrap font-mono text-xs text-neutral-600 leading-relaxed">{report}</pre>
                    </div>
                )}

                <div className="overflow-x-auto rounded-md border border-neutral-100">
                    <table className="w-full text-sm text-left text-neutral-600">
                        <thead className="text-[11px] text-neutral-500 uppercase bg-neutral-50">
                            <tr>
                                <th className="px-5 py-3 font-medium border-b border-neutral-200">Metric</th>
                                {datasets.map(ds => <th key={ds.label} className="px-5 py-3 border-b border-neutral-200 font-medium">{ds.label}</th>)}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-neutral-100">
                            <tr className="hover:bg-neutral-50">
                                <td className="px-5 py-3 font-medium text-neutral-900 text-xs">Avg Entropy</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className="px-5 py-3 font-mono text-xs">
                                        {ds.stats?.entropy.mean.toFixed(3)}
                                    </td>
                                ))}
                            </tr>
                            <tr className="hover:bg-neutral-50">
                                <td className="px-5 py-3 font-medium text-neutral-900 text-xs">Avg Velocity</td>
                                {datasets.map(ds => <td key={ds.label} className="px-5 py-3 font-mono text-xs">{ds.stats?.fluency_velocity.mean.toFixed(3)}</td>)}
                            </tr>
                            <tr className="hover:bg-neutral-50">
                                <td className="px-5 py-3 font-medium text-neutral-900 text-xs">Avg Jerk</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className="px-5 py-3 font-mono text-xs">
                                        {ds.stats?.fluency_jerk.mean.toFixed(3)}
                                    </td>
                                ))}
                            </tr>
                            <tr className="hover:bg-neutral-50">
                                <td className="px-5 py-3 font-medium text-neutral-900 text-xs">Fractal Dim</td>
                                {datasets.map(ds => <td key={ds.label} className="px-5 py-3 font-mono text-xs">{ds.stats?.fractal_dim.mean.toFixed(3)}</td>)}
                            </tr>
                            <tr className="hover:bg-neutral-50">
                                <td className="px-5 py-3 font-medium text-neutral-900 text-xs">Kinetic Energy (J)</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className="px-5 py-3 font-mono text-xs">
                                        {ds.stats?.kinetic_energy?.mean.toFixed(2) || "0.00"}
                                    </td>
                                ))}
                            </tr>
                            <tr className="hover:bg-neutral-50">
                                <td className="px-5 py-3 font-medium text-neutral-900 text-xs">Root Stress</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className="px-5 py-3 font-mono text-xs">
                                        {ds.stats?.root_stress?.mean.toFixed(2) || "0.00"}
                                    </td>
                                ))}
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
      )}
    </div>
  );
};
