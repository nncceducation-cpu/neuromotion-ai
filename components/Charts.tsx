
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis } from 'recharts';
import { MovementMetrics } from '../types';

interface ChartProps {
  data: MovementMetrics[];
}

// --- 1. Variability: Phase Space Reconstruction ---
export const PhaseSpaceChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-xl shadow-sm border border-slate-100">
      <h3 className="text-slate-600 text-sm font-semibold mb-2 flex items-center justify-between">
        <span className="flex items-center"><i className="fas fa-atom mr-2 text-purple-500"></i> Phase Space (Variability)</span>
        <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">Repertoire</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis type="number" dataKey="phase_x" name="Position" hide domain={['dataMin', 'dataMax']} />
            <YAxis type="number" dataKey="phase_v" name="Velocity" hide domain={['dataMin', 'dataMax']} />
            <ZAxis type="number" range={[20]} />
            <Tooltip 
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-slate-800 text-white text-xs p-2 rounded">
                      <p>Pos: {Number(payload[0].value).toFixed(2)}</p>
                      <p>Vel: {Number(payload[1].value).toFixed(2)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter name="Limb State" data={data} fill="#8b5cf6" fillOpacity={0.6} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// --- 2. Entropy & Complexity ---
export const EntropyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-xl shadow-sm border border-slate-100">
      <h3 className="text-slate-600 text-sm font-semibold mb-2 flex items-center justify-between">
        <span className="flex items-center"><i className="fas fa-random mr-2 text-rose-500"></i> Sample Entropy</span>
        <span className="text-xs bg-rose-100 text-rose-700 px-2 py-0.5 rounded">Predictability</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorEntropy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#f43f5e" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={[0, 1.2]} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
            <Area type="monotone" dataKey="entropy" stroke="#f43f5e" fillOpacity={1} fill="url(#colorEntropy)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// --- 3. Fluency (Velocity Profile) ---
export const FluencyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-xl shadow-sm border border-slate-100">
      <h3 className="text-slate-600 text-sm font-semibold mb-2 flex items-center justify-between">
         <span className="flex items-center"><i className="fas fa-water mr-2 text-sky-500"></i> Fluency (SAL Proxy)</span>
         <span className="text-xs bg-sky-100 text-sky-700 px-2 py-0.5 rounded">Gracefulness</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
            <Line type="basis" dataKey="fluency_velocity" stroke="#0ea5e9" strokeWidth={3} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// --- 4. Complexity (Fractal Dimension) ---
export const FractalChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-xl shadow-sm border border-slate-100">
      <h3 className="text-slate-600 text-sm font-semibold mb-2 flex items-center justify-between">
        <span className="flex items-center"><i className="fas fa-fingerprint mr-2 text-emerald-500"></i> Fractal Dimension</span>
        <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded">Complexity</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={[1, 2]} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
            <Line type="stepAfter" dataKey="fractal_dim" stroke="#10b981" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// --- 5. UE5 Physics: Kinetic Energy ---
export const KineticEnergyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-xl shadow-sm border border-slate-100">
      <h3 className="text-slate-600 text-sm font-semibold mb-2 flex items-center justify-between">
        <span className="flex items-center"><i className="fas fa-bolt mr-2 text-yellow-500"></i> Kinetic Energy (Rigid Body)</span>
        <span className="text-xs bg-yellow-100 text-yellow-700 px-2 py-0.5 rounded">PhysX Simulation</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorEnergy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#eab308" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#eab308" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px', color: '#fff' }} />
            <Area type="monotone" dataKey="kinetic_energy" stroke="#eab308" fillOpacity={1} fill="url(#colorEnergy)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// --- Confidence Gauge (Existing) ---
export const ConfidenceGauge: React.FC<{ value: number }> = ({ value }) => {
  const data = [
    { name: 'Score', value: value },
    { name: 'Remaining', value: 100 - value }
  ];

  return (
    <div className="h-full w-full bg-white p-4 rounded-xl shadow-sm border border-slate-100 flex flex-col items-center justify-center relative min-h-[200px]">
       <h3 className="absolute top-4 left-4 text-slate-600 text-sm font-semibold flex items-center">
        <i className="fas fa-tachometer-alt mr-2 text-slate-400"></i> AI Confidence
      </h3>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="70%"
            startAngle={180}
            endAngle={0}
            innerRadius={60}
            outerRadius={80}
            paddingAngle={5}
            dataKey="value"
          >
            <Cell key="cell-0" fill={value > 70 ? '#10b981' : value > 40 ? '#f59e0b' : '#ef4444'} />
            <Cell key="cell-1" fill="#f1f5f9" />
          </Pie>
        </PieChart>
      </ResponsiveContainer>
      <div className="absolute bottom-4 text-center">
        <div className="text-3xl font-bold text-slate-800">{value}%</div>
        <div className="text-xs text-slate-400">Certainty</div>
      </div>
    </div>
  );
};
