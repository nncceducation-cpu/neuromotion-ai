
import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { User, SavedReport } from '../types';
import { storageService } from '../services/storage';

interface GraphsViewProps {
  user: User;
  onClose: () => void;
}

const chartTooltipStyle = { backgroundColor: '#171717', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '11px' };

const METRICS = [
  { key: 'entropy', label: 'Entropy', description: 'Movement Predictability', color: '#171717', accessor: (r: SavedReport) => r.rawData.entropy },
  { key: 'fluency', label: 'Fluency (Jerk)', description: 'Movement Smoothness', color: '#525252', accessor: (r: SavedReport) => r.rawData.fluency },
  { key: 'complexity', label: 'Fractal Dimension', description: 'Movement Complexity', color: '#737373', accessor: (r: SavedReport) => r.rawData.complexity },
  { key: 'kinetic_energy', label: 'Kinetic Energy', description: 'Movement Vigor', color: '#404040', accessor: (r: SavedReport) => r.rawData.avg_kinetic_energy },
  { key: 'confidence', label: 'Confidence', description: 'AI Certainty', color: '#171717', accessor: (r: SavedReport) => r.confidence },
] as const;

export const GraphsView: React.FC<GraphsViewProps> = ({ user, onClose }) => {
  const [reports, setReports] = useState<SavedReport[]>([]);

  useEffect(() => {
    storageService.getReports(user.id).then(all => {
      setReports([...all].reverse());
    });
  }, [user.id]);

  // Build chart data: one point per report
  const chartData = reports.map((r, i) => ({
    index: i + 1,
    date: new Date(r.date).toLocaleDateString(),
    label: r.videoName,
    classification: r.classification,
    entropy: r.rawData.entropy,
    fluency: r.rawData.fluency,
    complexity: r.rawData.complexity,
    kinetic_energy: r.rawData.avg_kinetic_energy,
    confidence: r.confidence,
  }));

  return (
    <div className="animate-fade-in space-y-6 pb-12">
      <div className="flex items-center justify-between">
        <button onClick={onClose} className="text-neutral-400 hover:text-neutral-800 flex items-center gap-2 transition-colors text-sm">
          <i className="fas fa-arrow-left"></i> Back to Dashboard
        </button>
      </div>

      <div>
        <h1 className="text-2xl font-bold text-neutral-900 tracking-tight">Trends</h1>
        <p className="text-neutral-500 text-sm mt-1">
          {reports.length} analyses over time
        </p>
      </div>

      {reports.length < 2 ? (
        <div className="bg-white rounded-lg border border-neutral-200 p-16 text-center">
          <i className="fas fa-chart-line text-4xl text-neutral-200 mb-4"></i>
          <p className="text-neutral-500 text-sm">Run at least 2 analyses to see trends.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {METRICS.map(metric => (
            <div key={metric.key} className="bg-white rounded-lg border border-neutral-200 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-neutral-900">{metric.label}</h3>
                <span className="text-[10px] text-neutral-400 font-normal">{metric.description}</span>
              </div>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10, fill: '#a3a3a3' }}
                      tickLine={false}
                      axisLine={{ stroke: '#e5e5e5' }}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: '#a3a3a3' }}
                      tickLine={false}
                      axisLine={false}
                      width={40}
                      domain={metric.key === 'confidence' ? [0, 100] : ['auto', 'auto']}
                    />
                    <Tooltip
                      contentStyle={chartTooltipStyle}
                      labelFormatter={(label) => `${label}`}
                      formatter={(value: number) => [value.toFixed(2), metric.label]}
                    />
                    <Line
                      type="monotone"
                      dataKey={metric.key}
                      stroke={metric.color}
                      strokeWidth={2}
                      dot={{ r: 3, fill: metric.color }}
                      activeDot={{ r: 5 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}

          {/* Classification timeline */}
          <div className="bg-white rounded-lg border border-neutral-200 p-5 lg:col-span-2">
            <h3 className="text-sm font-semibold text-neutral-900 mb-4">Classification History</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-neutral-100">
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">#</th>
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">Date</th>
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">Video</th>
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">Classification</th>
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">Confidence</th>
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">Entropy</th>
                    <th className="text-left py-2 px-3 text-xs font-medium text-neutral-500">Jerk</th>
                  </tr>
                </thead>
                <tbody>
                  {chartData.map((row, i) => (
                    <tr key={i} className="border-b border-neutral-50 hover:bg-neutral-50">
                      <td className="py-2 px-3 text-neutral-400 font-mono text-xs">{row.index}</td>
                      <td className="py-2 px-3 text-neutral-600">{row.date}</td>
                      <td className="py-2 px-3 text-neutral-600 truncate max-w-[200px]">{row.label}</td>
                      <td className="py-2 px-3">
                        <span className="text-xs font-medium px-2 py-0.5 rounded bg-neutral-100 text-neutral-700">
                          {row.classification}
                        </span>
                      </td>
                      <td className="py-2 px-3 font-mono text-neutral-700">{row.confidence}%</td>
                      <td className="py-2 px-3 font-mono text-neutral-500">{row.entropy.toFixed(3)}</td>
                      <td className="py-2 px-3 font-mono text-neutral-500">{row.fluency.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
