
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, Legend, BarChart, Bar } from 'recharts';
import { GoogleGenAI } from "@google/genai";
import { MovementMetrics, ComparisonDataset, SavedReport } from '../types';

interface ComparisonViewProps {
  onBack: () => void;
  initialReports?: SavedReport[];
}

export const ComparisonView: React.FC<ComparisonViewProps> = ({ onBack, initialReports }) => {
  const [datasets, setDatasets] = useState<ComparisonDataset[]>([]);
  const [report, setReport] = useState<string | null>(null);

  // AI State
  const [aiReport, setAiReport] = useState<string>("");
  const [isGeneratingReport, setIsGeneratingReport] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<{ role: 'user' | 'ai', text: string }[]>([]);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // --- CSV PARSING & LOGIC ---
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
      // Added kinetic_energy and root_stress to stats calculation
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
          stats[key] = {
              mean,
              min: Math.min(...values),
              max: Math.max(...values),
              std
          };
      });
      return stats;
  };

  const generateAutomatedReport = (currentDatasets: ComparisonDataset[]) => {
      if (currentDatasets.length === 0) return;
      
      const lines: string[] = [];
      const date = new Date().toLocaleDateString();
      
      lines.push(`NEUROMOTION AI - COMPARATIVE CLINICAL REPORT`);
      lines.push(`Date: ${date}`);
      lines.push(`Subjects: ${currentDatasets.map(d => d.label).join(', ')}`);
      lines.push(``);
      lines.push(`----------------------------------------------------------------`);
      lines.push(`1. EXECUTIVE SUMMARY & DETAILED ANALYSIS`);
      lines.push(`----------------------------------------------------------------`);
      
      // Calculate leaders
      let maxEntropy = currentDatasets[0];
      let minEntropy = currentDatasets[0];
      let maxJerk = currentDatasets[0];
      let maxVel = currentDatasets[0];
      
      currentDatasets.forEach(d => {
          if ((d.stats?.entropy.mean || 0) > (maxEntropy.stats?.entropy.mean || 0)) maxEntropy = d;
          if ((d.stats?.entropy.mean || 0) < (minEntropy.stats?.entropy.mean || 0)) minEntropy = d;
          if ((d.stats?.fluency_jerk.mean || 0) > (maxJerk.stats?.fluency_jerk.mean || 0)) maxJerk = d;
          if ((d.stats?.fluency_velocity.mean || 0) > (maxVel.stats?.fluency_velocity.mean || 0)) maxVel = d;
      });

      lines.push(`• COMPLEXITY LEADER: ${maxEntropy.label}`);
      lines.push(`  - Highest variability (Mean Entropy: ${(maxEntropy.stats?.entropy.mean || 0).toFixed(3)}).`);
      lines.push(`  - Clinical significance: Indicates a richer motor repertoire and healthy corticospinal integrity.`);
      lines.push(``);
      
      if (minEntropy.label !== maxEntropy.label) {
          lines.push(`• CONCERN FOR POVERTY OF MOVEMENT: ${minEntropy.label}`);
          lines.push(`  - Lowest variability (Mean Entropy: ${(minEntropy.stats?.entropy.mean || 0).toFixed(3)}).`);
          lines.push(`  - Clinical significance: May indicate lethargy, hypotonia, or encephalopathy (Sarnat II/III).`);
          lines.push(``);
      }

      lines.push(`• HIGHEST ACTIVITY INTENSITY: ${maxJerk.label}`);
      lines.push(`  - Peak Jerk Index: ${(maxJerk.stats?.fluency_jerk.mean || 0).toFixed(2)}`);
      lines.push(`  - Clinical significance: If excessive (>8.0), consider jitteriness, hyperexcitability, or tremors.`);
      lines.push(``);

      lines.push(`----------------------------------------------------------------`);
      lines.push(`2. DETAILED BIOMARKER PROFILES`);
      lines.push(`----------------------------------------------------------------`);
      
      currentDatasets.forEach(d => {
          const e = d.stats?.entropy.mean || 0;
          const v = d.stats?.fluency_velocity.mean || 0;
          const j = d.stats?.fluency_jerk.mean || 0;
          const f = d.stats?.fractal_dim.mean || 0;
          const ke = d.stats?.kinetic_energy.mean || 0;
          
          lines.push(`SUBJECT: ${d.label}`);
          lines.push(`  • Entropy (Complexity):   ${e.toFixed(3)} [Norm: >0.6]`);
          lines.push(`  • Velocity (Activity):    ${v.toFixed(3)}`);
          lines.push(`  • Jerk (Smoothness):      ${j.toFixed(3)} [Norm: <7.0]`);
          lines.push(`  • Fractal Dim (Texture):  ${f.toFixed(3)}`);
          lines.push(`  • Kinetic Energy (PhysX): ${ke.toFixed(2)} J`);
          
          let impression = [];
          if (e < 0.4) impression.push("Markedly reduced complexity (Warning)");
          else if (e < 0.6) impression.push("Mildly reduced complexity");
          else impression.push("Normal complexity");

          if (j > 8.0) impression.push("High frequency tremors detected");
          
          lines.push(`  => INTERPRETATION: ${impression.join(", ")}`);
          lines.push(``);
      });

      if (currentDatasets.length === 2) {
          lines.push(`----------------------------------------------------------------`);
          lines.push(`3. DIRECT COMPARISON (${currentDatasets[0].label} vs ${currentDatasets[1].label})`);
          lines.push(`----------------------------------------------------------------`);
          const d1 = currentDatasets[0];
          const d2 = currentDatasets[1];
          
          const eDiff = ((d1.stats?.entropy.mean || 0) - (d2.stats?.entropy.mean || 0));
          const vDiff = ((d1.stats?.fluency_velocity.mean || 0) - (d2.stats?.fluency_velocity.mean || 0));
          const jDiff = ((d1.stats?.fluency_jerk.mean || 0) - (d2.stats?.fluency_jerk.mean || 0));
          
          lines.push(`• ENTROPY: ${d1.label} is ${Math.abs(eDiff / (d2.stats?.entropy.mean || 1) * 100).toFixed(1)}% ${eDiff > 0 ? 'more' : 'less'} complex.`);
          lines.push(`• VELOCITY: ${d1.label} is ${Math.abs(vDiff / (d2.stats?.fluency_velocity.mean || 1) * 100).toFixed(1)}% ${vDiff > 0 ? 'faster' : 'slower'}.`);
          lines.push(`• JERK: ${d1.label} is ${Math.abs(jDiff / (d2.stats?.fluency_jerk.mean || 1) * 100).toFixed(1)}% ${jDiff > 0 ? 'jitterier' : 'smoother'}.`);
      }
      
      lines.push(``);
      lines.push(`Report generated automatically by NeuroMotion AI.`);

      setReport(lines.join('\n'));
  };

  // --- AI HANDLERS ---
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

  const getComparisonSummary = () => {
    return datasets.map(d => 
      `- ${d.label}: Entropy=${d.stats?.entropy.mean.toFixed(2)}, Energy=${d.stats?.kinetic_energy?.mean.toFixed(2) || 0}, Stress=${d.stats?.root_stress?.mean.toFixed(2) || 0}, Jerk=${d.stats?.fluency_jerk.mean.toFixed(2)}`
    ).join('\n');
  };

  const handleGenerateAIReport = async () => {
    if (datasets.length < 1) return;
    setIsGeneratingReport(true);

    const summaryData = getComparisonSummary();
    const prompt = `
      You are an expert Biomechanics Data Scientist.
      Analyze the difference between the following motion sessions based on these computed metrics:
      
      ${summaryData}

      Provide a concise 3-paragraph summary:
      1. Performance Comparison (Intensity, Kinetic Energy & Output)
      2. Stability & Control Analysis (Root Stress & Entropy/Complexity)
      3. Kinematic Variability & Smoothness.
      
      STRICT REQUIREMENT: Focus ONLY on the physics, movement patterns, and data trends. 
      DO NOT provide any medical diagnoses, clinical interpretations (e.g. Sarnat stages, CP), or medical advice.
      
      Use professional technical language.
    `;

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
        });
        setAiReport(response.text || "No analysis generated.");
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

    const summaryData = getComparisonSummary();
    const prompt = `
      Context: The user is looking at a dashboard comparing motion capture sessions.
      Data Summary:
      ${summaryData}
      
      User Question: "${userMsg}"
      
      Answer the user specifically using the data provided. Keep it helpful, encouraging, and brief (under 50 words if possible).
    `;

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: prompt,
            config: { systemInstruction: "You are a helpful AI Sports Science and Biomechanics Assistant." }
        });
        setChatHistory(prev => [...prev, { role: 'ai', text: response.text || "I couldn't generate a response." }]);
    } catch (err) {
        setChatHistory(prev => [...prev, { role: 'ai', text: "Error connecting to AI service." }]);
    } finally {
        setIsChatLoading(false);
    }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // --- LOAD INITIAL REPORTS ---
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

  // --- RENDER HELPERS ---
  const colors = ['#0ea5e9', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6'];

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
        .scrollbar-hide::-webkit-scrollbar { display: none; }
      `}</style>
      <div className="flex justify-between items-center bg-white p-4 rounded-xl border border-slate-200 shadow-sm no-print">
         <div className="flex items-center gap-4">
            <button onClick={onBack} className="text-slate-500 hover:text-slate-800 transition-colors">
                <i className="fas fa-arrow-left mr-2"></i> Back
            </button>
            <h2 className="text-xl font-bold text-slate-800">
                <i className="fas fa-balance-scale-right mr-2 text-indigo-500"></i>
                Comparative Analysis
            </h2>
         </div>
         <label className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg cursor-pointer transition-colors shadow-lg shadow-indigo-200 flex items-center">
            <i className="fas fa-file-csv mr-2"></i> Add CSV Files
            <input type="file" multiple accept=".csv" className="hidden" onChange={handleFileUpload} />
         </label>
      </div>

      {datasets.length === 0 ? (
          <div className="flex flex-col items-center justify-center p-20 bg-slate-50 border-2 border-dashed border-slate-300 rounded-2xl text-slate-400">
              <i className="fas fa-table text-5xl mb-4 opacity-50"></i>
              <h3 className="text-lg font-bold mb-2">No Datasets Loaded</h3>
              <p className="text-sm max-w-md text-center">Select reports from the Dashboard or upload exported CSV files to compare biomarkers side-by-side.</p>
          </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            
            {/* 1. TIME SERIES COMPARISON */}
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm col-span-1 lg:col-span-2">
                <h3 className="text-slate-700 font-bold mb-4 flex items-center">
                    <i className="fas fa-chart-line mr-2 text-sky-500"></i> Multi-Subject Time Series (Entropy)
                </h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <LineChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis dataKey="timestamp" type="number" allowDuplicatedCategory={false} hide />
                            <YAxis domain={[0, 1.2]} />
                            <Tooltip contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff'}} />
                            <Legend />
                            {datasets.map((ds, i) => (
                                <Line 
                                    key={ds.label} 
                                    data={ds.data} 
                                    dataKey="entropy" 
                                    name={`${ds.label} (Entropy)`}
                                    stroke={colors[i % colors.length]} 
                                    dot={false} 
                                    strokeWidth={2}
                                />
                            ))}
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* AI INSIGHTS SECTION */}
            <div className="col-span-1 lg:col-span-2 grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* AI REPORT CARD */}
                <div className="lg:col-span-2 bg-gradient-to-br from-indigo-50 to-white p-6 rounded-xl border border-indigo-100 shadow-sm flex flex-col">
                     <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-bold flex items-center gap-2 text-indigo-900">
                          <i className="fas fa-magic text-indigo-500"></i> AI Biomechanics Report
                        </h3>
                        <button 
                          onClick={handleGenerateAIReport}
                          disabled={isGeneratingReport}
                          className="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-sm font-semibold transition-colors flex items-center gap-2 shadow-sm"
                        >
                          {isGeneratingReport ? <i className="fas fa-circle-notch fa-spin"></i> : <i className="fas fa-wand-magic-sparkles"></i>}
                          {isGeneratingReport ? "Analyzing..." : "Generate Analysis"}
                        </button>
                    </div>

                    <div className="bg-white rounded-lg p-6 min-h-[200px] border border-indigo-50 flex-1">
                        {aiReport ? (
                          <div className="prose prose-sm max-w-none text-slate-700">
                            <div className="whitespace-pre-line leading-relaxed">
                              {aiReport}
                            </div>
                          </div>
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center text-slate-400 text-sm">
                            <i className="fas fa-brain text-3xl mb-3 opacity-20"></i>
                            <p>Click "Generate Analysis" to have Gemini interpret your motion data.</p>
                          </div>
                        )}
                    </div>
                </div>

                {/* AI COACH CHAT CARD */}
                <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm flex flex-col h-[400px]">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-emerald-600">
                        <i className="fas fa-comments"></i> AI Coach
                    </h3>
                    
                    <div className="flex-1 overflow-y-auto mb-4 space-y-3 pr-2 scrollbar-thin">
                        {chatHistory.length === 0 && (
                          <div className="text-center text-slate-400 text-xs mt-10 italic">
                             <i className="fas fa-comment-dots text-xl mb-2 opacity-30"></i>
                             <p>Ask me about stress levels, energy output, or stability differences...</p>
                          </div>
                        )}
                        {chatHistory.map((msg, i) => (
                          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`max-w-[85%] rounded-lg p-3 text-sm ${
                              msg.role === 'user' 
                                ? 'bg-emerald-50 text-emerald-800 border border-emerald-100' 
                                : 'bg-slate-100 text-slate-700'
                            }`}>
                              {msg.text}
                            </div>
                          </div>
                        ))}
                        {isChatLoading && (
                           <div className="flex justify-start">
                             <div className="bg-slate-100 rounded-lg p-3">
                               <i className="fas fa-circle-notch fa-spin text-slate-400"></i>
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
                          className="w-full bg-slate-50 border border-slate-200 rounded-lg py-3 pl-4 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/50 transition-colors text-slate-800 placeholder-slate-400"
                        />
                        <button 
                          type="submit" 
                          disabled={!chatInput.trim() || isChatLoading}
                          className="absolute right-2 top-2 p-1.5 bg-emerald-500 hover:bg-emerald-400 disabled:opacity-50 text-white rounded-md transition-colors w-8 h-8 flex items-center justify-center"
                        >
                          <i className="fas fa-paper-plane text-xs"></i>
                        </button>
                    </form>
                </div>
            </div>

            {/* 2. PHASE SPACE COMPARISON */}
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                <h3 className="text-slate-700 font-bold mb-4 flex items-center">
                    <i className="fas fa-atom mr-2 text-purple-500"></i> Phase Space Overlay
                </h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart>
                            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                            <XAxis type="number" dataKey="phase_x" name="Pos" />
                            <YAxis type="number" dataKey="phase_v" name="Vel" />
                            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                            <Legend />
                            {datasets.map((ds, i) => (
                                <Scatter 
                                    key={ds.label} 
                                    name={ds.label} 
                                    data={ds.data} 
                                    fill={colors[i % colors.length]} 
                                    opacity={0.6}
                                />
                            ))}
                        </ScatterChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* 3. STATISTICAL DISTRIBUTION (Bar Chart) */}
            <div className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm">
                 <h3 className="text-slate-700 font-bold mb-4 flex items-center">
                    <i className="fas fa-chart-bar mr-2 text-emerald-500"></i> Mean Velocity Comparison
                </h3>
                <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={datasets.map(d => ({ name: d.label, velocity: d.stats?.fluency_velocity.mean || 0 }))}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                            <XAxis dataKey="name" />
                            <YAxis />
                            <Tooltip cursor={{fill: 'transparent'}} contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff'}} />
                            <Bar dataKey="velocity" fill="#10b981" radius={[4, 4, 0, 0]} />
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* 4. TEXT REPORT & STATS TABLE */}
            <div id="printable-report" className="bg-white p-6 rounded-xl border border-slate-200 shadow-sm col-span-1 lg:col-span-2">
                <div className="flex justify-between items-center mb-4 no-print">
                     <h3 className="text-slate-700 font-bold flex items-center">
                        <i className="fas fa-file-medical-alt mr-2 text-slate-500"></i> Automated Clinical Report
                    </h3>
                    <button 
                        onClick={() => window.print()}
                        className="bg-sky-600 hover:bg-sky-500 text-white px-6 py-2 rounded-lg transition-colors shadow-md hover:shadow-lg font-bold flex items-center no-print"
                    >
                        <i className="fas fa-file-pdf mr-2"></i> Export Comparison PDF
                    </button>
                </div>
                
                {report && (
                    <div className="mb-8 p-6 bg-slate-50 rounded-lg border border-slate-200 shadow-inner overflow-hidden relative">
                         <div className="absolute top-0 left-0 w-2 h-full bg-indigo-500"></div>
                         <pre className="whitespace-pre-wrap font-mono text-sm text-slate-700 leading-relaxed border-l pl-4">{report}</pre>
                    </div>
                )}

                <div className="overflow-x-auto rounded-lg border border-slate-100">
                    <table className="w-full text-sm text-left text-slate-600">
                        <thead className="text-xs text-slate-700 uppercase bg-slate-50">
                            <tr>
                                <th className="px-6 py-3 font-bold border-b border-slate-200">Metric</th>
                                {datasets.map(ds => <th key={ds.label} className="px-6 py-3 border-b border-slate-200 text-indigo-900">{ds.label}</th>)}
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                            <tr className="bg-white hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-800">Avg Entropy</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className={`px-6 py-4 ${(ds.stats?.entropy.mean || 0) < 0.6 ? 'text-amber-600 font-bold' : ''}`}>
                                        {ds.stats?.entropy.mean.toFixed(3)}
                                    </td>
                                ))}
                            </tr>
                            <tr className="bg-white hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-800">Avg Velocity</td>
                                {datasets.map(ds => <td key={ds.label} className="px-6 py-4">{ds.stats?.fluency_velocity.mean.toFixed(3)}</td>)}
                            </tr>
                            <tr className="bg-white hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-800">Avg Jerk</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className={`px-6 py-4 ${(ds.stats?.fluency_jerk.mean || 0) > 8.0 ? 'text-red-600 font-bold' : ''}`}>
                                        {ds.stats?.fluency_jerk.mean.toFixed(3)}
                                    </td>
                                ))}
                            </tr>
                            <tr className="bg-white hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-800">Fractal Dim</td>
                                {datasets.map(ds => <td key={ds.label} className="px-6 py-4">{ds.stats?.fractal_dim.mean.toFixed(3)}</td>)}
                            </tr>
                            <tr className="bg-white hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-800">Avg Kinetic Energy (J)</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className="px-6 py-4 text-slate-600">
                                        {ds.stats?.kinetic_energy?.mean.toFixed(2) || "0.00"}
                                    </td>
                                ))}
                            </tr>
                            <tr className="bg-white hover:bg-slate-50">
                                <td className="px-6 py-4 font-medium text-slate-800">Avg Root Stress</td>
                                {datasets.map(ds => (
                                    <td key={ds.label} className="px-6 py-4 text-slate-600">
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
