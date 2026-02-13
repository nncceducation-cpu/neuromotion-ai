
import React, { useEffect, useState } from 'react';
import { User, SavedReport } from '../types';
import { storageService } from '../services/storage';

interface DashboardProps {
  user: User;
  onNewAnalysis: () => void;
  onLiveAnalysis: () => void;
  onTrainingMode: () => void;
  onComparisonMode: () => void;
  onValidationView: () => void;
  onGraphsView: () => void;
  onViewReport: (report: SavedReport) => void;
  onCompareReports: (reports: SavedReport[]) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ user, onNewAnalysis, onLiveAnalysis, onTrainingMode, onComparisonMode, onValidationView, onGraphsView, onViewReport, onCompareReports }) => {
  const [reports, setReports] = useState<SavedReport[]>([]);
  const [learnedStats, setLearnedStats] = useState<{ totalLearned: number, breakdown: Record<string, number> }>({ totalLearned: 0, breakdown: {} });
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    storageService.getReports(user.id).then(setReports);
    storageService.getLearnedStats(user.id).then(setLearnedStats);
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

  const handleDelete = async (e: React.MouseEvent, reportId: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this assessment? This cannot be undone.')) return;
    await storageService.deleteReport(user.id, reportId);
    setReports(prev => prev.filter(r => r.id !== reportId));
    setSelectedIds(prev => { const next = new Set(prev); next.delete(reportId); return next; });
  };

  const handleExportCSV = (e: React.MouseEvent, report: SavedReport) => {
    e.stopPropagation();
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
      {/* Welcome Section */}
      <div className="pt-4">
        <h1 className="text-3xl font-bold text-neutral-900 tracking-tight">Welcome back, {user.name}</h1>
        <p className="text-neutral-500 mt-1 text-sm">
          {reports.length} analyses completed · {learnedStats.totalLearned} expert corrections learned
        </p>
      </div>

      {/* Action Grid */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <button
            onClick={onNewAnalysis}
            className="flex flex-col items-center gap-2 px-4 py-5 bg-neutral-900 text-white rounded-lg hover:bg-neutral-800 transition-colors"
        >
            <i className="fas fa-plus text-sm"></i>
            <span className="text-xs font-medium">New Analysis</span>
        </button>
        <button
            onClick={onLiveAnalysis}
            className="flex flex-col items-center gap-2 px-4 py-5 bg-white border border-neutral-200 text-neutral-700 rounded-lg hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
        >
            <i className="fas fa-video text-sm"></i>
            <span className="text-xs font-medium">Live Capture</span>
        </button>
        <button
            onClick={onTrainingMode}
            className="flex flex-col items-center gap-2 px-4 py-5 bg-white border border-neutral-200 text-neutral-700 rounded-lg hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
        >
            <i className="fas fa-graduation-cap text-sm"></i>
            <span className="text-xs font-medium">Training</span>
        </button>
        <button
            onClick={onValidationView}
            className="flex flex-col items-center gap-2 px-4 py-5 bg-white border border-neutral-200 text-neutral-700 rounded-lg hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
        >
            <i className="fas fa-clipboard-check text-sm"></i>
            <span className="text-xs font-medium">Validation</span>
        </button>
        <button
            onClick={onComparisonMode}
            className="flex flex-col items-center gap-2 px-4 py-5 bg-white border border-neutral-200 text-neutral-700 rounded-lg hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
        >
            <i className="fas fa-columns text-sm"></i>
            <span className="text-xs font-medium">Compare</span>
        </button>
        <button
            onClick={onGraphsView}
            className="flex flex-col items-center gap-2 px-4 py-5 bg-white border border-neutral-200 text-neutral-700 rounded-lg hover:bg-neutral-50 hover:border-neutral-300 transition-colors"
        >
            <i className="fas fa-chart-line text-sm"></i>
            <span className="text-xs font-medium">Graphs</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stats Row */}
          <div className="lg:col-span-2 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white p-5 rounded-lg border border-neutral-200">
                  <p className="text-xs text-neutral-500 font-medium mb-1">Total Analyses</p>
                  <p className="text-2xl font-semibold text-neutral-900 tracking-tight">{reports.length}</p>
                </div>
                <div className="bg-white p-5 rounded-lg border border-neutral-200">
                  <p className="text-xs text-neutral-500 font-medium mb-1">Avg Confidence</p>
                  <p className="text-2xl font-semibold text-neutral-900 tracking-tight">
                    {reports.length > 0
                        ? Math.round(reports.reduce((acc, r) => acc + r.confidence, 0) / reports.length) + '%'
                        : '—'}
                  </p>
                </div>
                <div className="bg-white p-5 rounded-lg border border-neutral-200">
                    <p className="text-xs text-neutral-500 font-medium mb-1">Last Activity</p>
                    <p className="text-2xl font-semibold text-neutral-900 tracking-tight">
                        {reports.length > 0 ? new Date(reports[0].date).toLocaleDateString() : '—'}
                    </p>
                </div>
              </div>

              {/* Recent Assessments */}
              <div className="bg-white rounded-lg border border-neutral-200 overflow-hidden">
                <div className="px-5 py-4 border-b border-neutral-100 flex justify-between items-center">
                    <h2 className="font-semibold text-neutral-900 text-sm">Recent Assessments</h2>
                    {selectedIds.size > 0 && (
                        <span className="text-xs text-neutral-500">{selectedIds.size} selected</span>
                    )}
                </div>

                {reports.length === 0 ? (
                    <div className="px-5 py-16 text-center text-neutral-400">
                        <p className="text-sm">No assessments yet. Start your first analysis above.</p>
                    </div>
                ) : (
                    <div className="divide-y divide-neutral-100">
                        {reports.map((report) => (
                            <div key={report.id} className="px-5 py-3 hover:bg-neutral-50 transition-colors flex items-center justify-between group cursor-pointer" onClick={() => onViewReport(report)}>
                                <div className="flex items-center gap-3">
                                    <div
                                        onClick={(e) => { e.stopPropagation(); toggleSelection(report.id); }}
                                        className="w-5 h-5 flex items-center justify-center text-neutral-300 hover:text-neutral-600 transition-colors rounded"
                                    >
                                        {selectedIds.has(report.id)
                                            ? <i className="fas fa-check-square text-neutral-900"></i>
                                            : <i className="far fa-square"></i>
                                        }
                                    </div>

                                    <div>
                                        <h4 className="font-medium text-sm text-neutral-900">{report.videoName}</h4>
                                        <p className="text-xs text-neutral-400">
                                            {new Date(report.date).toLocaleDateString()} · {new Date(report.date).toLocaleTimeString()}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center gap-3">
                                    {report.timelineData && (
                                        <button
                                            onClick={(e) => handleExportCSV(e, report)}
                                            className="w-7 h-7 flex items-center justify-center rounded text-neutral-300 hover:text-neutral-700 hover:bg-neutral-100 transition-colors"
                                            title="Export CSV"
                                        >
                                            <i className="fas fa-download text-xs"></i>
                                        </button>
                                    )}
                                    <button
                                        onClick={(e) => handleDelete(e, report.id)}
                                        className="w-7 h-7 flex items-center justify-center rounded text-neutral-300 hover:text-red-500 hover:bg-red-50 transition-colors opacity-0 group-hover:opacity-100"
                                        title="Delete assessment"
                                    >
                                        <i className="fas fa-trash-alt text-xs"></i>
                                    </button>
                                    <i className="fas fa-chevron-right text-neutral-300 text-xs group-hover:text-neutral-500 transition-colors"></i>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
              </div>
          </div>

          {/* Knowledge Base Sidebar */}
          <div className="lg:col-span-1">
             <div className="bg-white rounded-lg p-5 border border-neutral-200 h-full">
                <h3 className="font-semibold text-sm text-neutral-900 mb-4">Knowledge Base</h3>

                <div className="text-center py-6 border-b border-neutral-100 mb-4">
                     <span className="text-4xl font-bold text-neutral-900 tracking-tight">{learnedStats.totalLearned}</span>
                     <p className="text-xs text-neutral-400 mt-1">Patterns Learned</p>
                </div>

                <div>
                    <h4 className="text-xs font-medium text-neutral-500 uppercase tracking-wider mb-3">Categories</h4>
                    {Object.keys(learnedStats.breakdown).length === 0 ? (
                         <p className="text-xs text-neutral-400">No patterns learned yet. Correct AI diagnoses to teach the system.</p>
                    ) : (
                        <ul className="space-y-1.5">
                            {Object.entries(learnedStats.breakdown).map(([cat, count]) => (
                                <li key={cat} className="flex justify-between items-center py-1.5 px-2 rounded hover:bg-neutral-50 transition-colors">
                                    <span className="text-sm text-neutral-700">{cat}</span>
                                    <span className="text-xs font-mono text-neutral-400 bg-neutral-100 px-2 py-0.5 rounded">{count}</span>
                                </li>
                            ))}
                        </ul>
                    )}
                </div>

                <div className="mt-6 pt-4 border-t border-neutral-100 text-xs text-neutral-400 leading-relaxed">
                    The system uses video fingerprinting to identify previously corrected patterns. Similar biomarkers will apply your expert rules automatically.
                </div>
             </div>
          </div>
      </div>

      {/* Floating Compare Button */}
      {selectedIds.size > 0 && (
          <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-50">
              <button
                onClick={handleCompareClick}
                className="bg-neutral-900 hover:bg-neutral-800 text-white pl-5 pr-6 py-3 rounded-full shadow-lg font-medium flex items-center gap-3 transition-all text-sm"
              >
                  <span className="bg-white text-neutral-900 w-6 h-6 rounded-full flex items-center justify-center font-semibold text-xs">
                      {selectedIds.size}
                  </span>
                  Compare Selected
                  <i className="fas fa-arrow-right text-xs"></i>
              </button>
          </div>
      )}
    </div>
  );
};
