
import React, { useEffect, useState } from 'react';
import { User, SavedReport } from '../types';
import { storageService } from '../services/storage';

interface DashboardProps {
  user: User;
  onNewAnalysis: () => void;
  onLiveAnalysis: () => void;
  onTrainingMode: () => void; 
  onComparisonMode: () => void; 
  onViewReport: (report: SavedReport) => void;
  onCompareReports: (reports: SavedReport[]) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ user, onNewAnalysis, onLiveAnalysis, onTrainingMode, onComparisonMode, onViewReport, onCompareReports }) => {
  const [reports, setReports] = useState<SavedReport[]>([]);
  const [learnedStats, setLearnedStats] = useState<{ totalLearned: number, breakdown: Record<string, number> }>({ totalLearned: 0, breakdown: {} });
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    setReports(storageService.getReports(user.id));
    setLearnedStats(storageService.getLearnedStats(user.id));
  }, [user.id]);

  const toggleSelection = (id: string) => {
    const newSet = new Set(selectedIds);
    if (newSet.has(id)) newSet.delete(id);
    else newSet.add(id);
    setSelectedIds(newSet);
  };

  const handleCompareClick = () => {
    const selected = reports.filter(r => selectedIds.has(r.id));
    onCompareReports(selected);
  };

  const handleExportCSV = (e: React.MouseEvent, report: SavedReport) => {
    e.stopPropagation(); // Prevent opening the report
    if (!report.timelineData || report.timelineData.length === 0) {
        alert("No raw data available for this report.");
        return;
    }
    
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
    link.setAttribute("download", `neuromotion_${report.videoName.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_${new Date(report.date).getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="animate-fade-in space-y-8 pb-24">
      {/* Welcome Banner */}
      <div className="bg-gradient-to-r from-slate-800 to-slate-900 rounded-2xl p-8 text-white shadow-xl relative overflow-hidden">
        <div className="relative z-10">
          <h1 className="text-3xl font-bold mb-2">Hello, {user.name}</h1>
          <p className="text-slate-300 max-w-xl">
            Welcome to your NeuroMotion AI dashboard. The system has analyzed {reports.length} videos and learned from {learnedStats.totalLearned} expert corrections.
          </p>
          <div className="mt-8 flex flex-wrap gap-4">
            <button 
                onClick={onNewAnalysis}
                className="bg-sky-500 hover:bg-sky-400 text-white px-6 py-3 rounded-lg font-semibold shadow-lg shadow-sky-900/50 transition-all flex items-center"
            >
                <i className="fas fa-file-upload mr-2"></i> Clinical Analysis
            </button>
            <button 
                onClick={onLiveAnalysis}
                className="bg-rose-500 hover:bg-rose-400 text-white px-6 py-3 rounded-lg font-semibold shadow-lg shadow-rose-900/50 transition-all flex items-center animate-pulse"
            >
                <i className="fas fa-video mr-2"></i> Live Assessment
            </button>
            <button 
                onClick={onTrainingMode}
                className="bg-indigo-600 hover:bg-indigo-500 text-white px-6 py-3 rounded-lg font-semibold shadow-lg shadow-indigo-900/50 transition-all flex items-center border border-indigo-400/30"
            >
                <i className="fas fa-graduation-cap mr-2"></i> Training Mode
            </button>
             {/* COMPARISON MODE BUTTON (Manual Upload) */}
             <button 
                onClick={onComparisonMode}
                className="bg-slate-700 hover:bg-slate-600 text-white px-6 py-3 rounded-lg font-semibold shadow-lg transition-all flex items-center border border-slate-500"
            >
                <i className="fas fa-balance-scale mr-2"></i> CSV Compare
            </button>
          </div>
        </div>
        <div className="absolute right-0 bottom-0 opacity-10 text-9xl transform translate-x-12 translate-y-12">
            <i className="fas fa-brain"></i>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stats Column */}
          <div className="lg:col-span-2 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-slate-500 font-medium text-sm">Total Analyses</h3>
                    <div className="w-10 h-10 rounded-full bg-blue-50 text-blue-500 flex items-center justify-center">
                        <i className="fas fa-file-medical-alt"></i>
                    </div>
                  </div>
                  <p className="text-3xl font-bold text-slate-800">{reports.length}</p>
                </div>
                <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-slate-500 font-medium text-sm">Avg Confidence</h3>
                    <div className="w-10 h-10 rounded-full bg-emerald-50 text-emerald-500 flex items-center justify-center">
                        <i className="fas fa-chart-pie"></i>
                    </div>
                  </div>
                  <p className="text-3xl font-bold text-slate-800">
                    {reports.length > 0 
                        ? Math.round(reports.reduce((acc, r) => acc + r.confidence, 0) / reports.length) + '%' 
                        : '-'}
                  </p>
                </div>
                <div className="bg-white p-6 rounded-xl border border-slate-100 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-slate-500 font-medium text-sm">Last Activity</h3>
                        <div className="w-10 h-10 rounded-full bg-purple-50 text-purple-500 flex items-center justify-center">
                            <i className="fas fa-clock"></i>
                        </div>
                    </div>
                    <p className="text-lg font-semibold text-slate-800">
                        {reports.length > 0 ? new Date(reports[0].date).toLocaleDateString() : 'None'}
                    </p>
                </div>
              </div>

              {/* Recent Activity List */}
              <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-6 border-b border-slate-100 flex justify-between items-center">
                    <h2 className="font-bold text-slate-800 text-lg">Recent Assessments</h2>
                    {selectedIds.size > 0 && (
                        <span className="text-sm text-slate-500">{selectedIds.size} selected</span>
                    )}
                </div>
                
                {reports.length === 0 ? (
                    <div className="p-12 text-center text-slate-400">
                        <i className="fas fa-folder-open text-4xl mb-3 opacity-50"></i>
                        <p>No assessments found. Start your first analysis above.</p>
                    </div>
                ) : (
                    <div className="divide-y divide-slate-100">
                        {reports.map((report) => (
                            <div key={report.id} className="p-4 hover:bg-slate-50 transition-colors flex items-center justify-between group cursor-pointer" onClick={() => onViewReport(report)}>
                                <div className="flex items-center gap-4">
                                    {/* SELECTION CHECKBOX */}
                                    <div 
                                        onClick={(e) => { e.stopPropagation(); toggleSelection(report.id); }} 
                                        className="w-8 h-8 flex items-center justify-center text-slate-300 hover:text-sky-500 transition-colors rounded hover:bg-slate-100"
                                    >
                                        {selectedIds.has(report.id) 
                                            ? <i className="fas fa-check-square text-sky-500 text-xl"></i> 
                                            : <i className="far fa-square text-xl"></i>
                                        }
                                    </div>
                                    
                                    {/* Assessment Category Icon REMOVED per request */}
                                    
                                    <div>
                                        <h4 className="font-semibold text-slate-800">{report.videoName}</h4>
                                        <p className="text-xs text-slate-500">
                                            {new Date(report.date).toLocaleDateString()} • {new Date(report.date).toLocaleTimeString()}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4 md:gap-6">
                                    {/* EXPORT CSV BUTTON */}
                                    {report.timelineData && (
                                        <button 
                                            onClick={(e) => handleExportCSV(e, report)}
                                            className="w-8 h-8 flex items-center justify-center rounded-full bg-slate-100 text-slate-400 hover:bg-sky-100 hover:text-sky-600 transition-colors shadow-sm"
                                            title="Export Raw Data (CSV)"
                                        >
                                            <i className="fas fa-file-csv"></i>
                                        </button>
                                    )}

                                    <i className="fas fa-chevron-right text-slate-300 group-hover:text-sky-500 transition-colors"></i>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
              </div>
          </div>

          {/* Neural Memory Column */}
          <div className="lg:col-span-1">
             <div className="bg-gradient-to-br from-indigo-900 to-slate-900 rounded-xl p-6 text-white shadow-lg h-full border border-indigo-700">
                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 bg-indigo-500/20 rounded-lg flex items-center justify-center border border-indigo-400/30">
                        <i className="fas fa-network-wired text-indigo-300"></i>
                    </div>
                    <div>
                        <h3 className="font-bold text-lg">Neural Knowledge Base</h3>
                        <p className="text-indigo-300 text-xs">Expert Patterns Learned</p>
                    </div>
                </div>

                <div className="space-y-4">
                    <div className="text-center py-6 border-b border-white/10">
                         <span className="text-5xl font-bold text-white">{learnedStats.totalLearned}</span>
                         <p className="text-sm text-indigo-300 mt-2">Unique Video Patterns Memorized</p>
                    </div>
                    
                    <div>
                        <h4 className="text-xs font-bold uppercase tracking-wider text-indigo-400 mb-3">Learned Categories</h4>
                        {Object.keys(learnedStats.breakdown).length === 0 ? (
                             <p className="text-sm text-white/50 italic">No patterns learned yet. Correct AI diagnoses to teach the system.</p>
                        ) : (
                            <ul className="space-y-2">
                                {Object.entries(learnedStats.breakdown).map(([cat, count]) => (
                                    <li key={cat} className="flex justify-between items-center bg-white/5 p-2 rounded hover:bg-white/10 transition-colors">
                                        <span className="text-sm text-indigo-100">{cat}</span>
                                        <span className="bg-indigo-600 text-white text-xs px-2 py-0.5 rounded-full">{count}</span>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                    
                    <div className="mt-8 pt-4 border-t border-white/10 text-xs text-indigo-300 leading-relaxed">
                        <i className="fas fa-info-circle mr-1"></i>
                        The system uses "Video Fingerprinting" to identify previously corrected patterns. New videos with similar biomarkers will automatically apply your expert rules.
                    </div>
                </div>
             </div>
          </div>
      </div>

      {/* FLOATING COMPARE BUTTON */}
      {selectedIds.size > 0 && (
          <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
              <button 
                onClick={handleCompareClick} 
                className="bg-slate-800 hover:bg-slate-700 text-white pl-6 pr-8 py-4 rounded-full shadow-2xl font-bold flex items-center gap-4 hover:scale-105 transition-all animate-bounce-in border border-slate-600"
              >
                  <div className="bg-sky-500 text-white w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm shadow-sm">
                      {selectedIds.size}
                  </div>
                  <span className="text-lg">Compare Selected Cases</span>
                  <i className="fas fa-arrow-right"></i>
              </button>
          </div>
      )}
    </div>
  );
};
