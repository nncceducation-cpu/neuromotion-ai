
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis } from 'recharts';
import { MovementMetrics } from '../types';

interface ChartProps {
  data: MovementMetrics[];
}

const chartTooltipStyle = { backgroundColor: '#171717', border: 'none', borderRadius: '6px', color: '#fff', fontSize: '11px' };

export const PhaseSpaceChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-md border border-neutral-200">
      <h3 className="text-neutral-900 text-xs font-medium mb-3 flex items-center justify-between">
        <span>Phase Space</span>
        <span className="text-[10px] text-neutral-400 font-normal">Variability</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e5e5" />
            <XAxis type="number" dataKey="phase_x" name="Position" hide domain={['dataMin', 'dataMax']} />
            <YAxis type="number" dataKey="phase_v" name="Velocity" hide domain={['dataMin', 'dataMax']} />
            <ZAxis type="number" range={[20, 20]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  return (
                    <div className="bg-neutral-900 text-white text-[10px] px-2 py-1.5 rounded">
                      <p>Pos: {Number(payload[0].value).toFixed(2)}</p>
                      <p>Vel: {Number(payload[1].value).toFixed(2)}</p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Scatter name="Limb State" data={data} fill="#737373" fillOpacity={0.5} />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const EntropyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-md border border-neutral-200">
      <h3 className="text-neutral-900 text-xs font-medium mb-3 flex items-center justify-between">
        <span>Sample Entropy</span>
        <span className="text-[10px] text-neutral-400 font-normal">Predictability</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorEntropy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#171717" stopOpacity={0.15}/>
                <stop offset="95%" stopColor="#171717" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={[0, 1.2]} />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Area type="monotone" dataKey="entropy" stroke="#171717" fillOpacity={1} fill="url(#colorEntropy)" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const FluencyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-md border border-neutral-200">
      <h3 className="text-neutral-900 text-xs font-medium mb-3 flex items-center justify-between">
         <span>Fluency</span>
         <span className="text-[10px] text-neutral-400 font-normal">SAL Proxy</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Line type="basis" dataKey="fluency_velocity" stroke="#525252" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const FractalChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-md border border-neutral-200">
      <h3 className="text-neutral-900 text-xs font-medium mb-3 flex items-center justify-between">
        <span>Fractal Dimension</span>
        <span className="text-[10px] text-neutral-400 font-normal">Complexity</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={[1, 2]} />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Line type="stepAfter" dataKey="fractal_dim" stroke="#737373" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const KineticEnergyChart: React.FC<ChartProps> = ({ data }) => {
  return (
    <div className="h-64 w-full bg-white p-4 rounded-md border border-neutral-200">
      <h3 className="text-neutral-900 text-xs font-medium mb-3 flex items-center justify-between">
        <span>Kinetic Energy</span>
        <span className="text-[10px] text-neutral-400 font-normal">Rigid Body</span>
      </h3>
      <div className="h-[85%] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorEnergy" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#404040" stopOpacity={0.15}/>
                <stop offset="95%" stopColor="#404040" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e5e5" />
            <XAxis dataKey="timestamp" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip contentStyle={chartTooltipStyle} />
            <Area type="monotone" dataKey="kinetic_energy" stroke="#404040" fillOpacity={1} fill="url(#colorEnergy)" strokeWidth={1.5} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export const ConfidenceGauge: React.FC<{ value: number }> = ({ value }) => {
  const data = [
    { name: 'Score', value: value },
    { name: 'Remaining', value: 100 - value }
  ];

  return (
    <div className="h-full w-full bg-white p-3 rounded-md border border-neutral-200 flex flex-col items-center relative">
       <h3 className="text-neutral-500 text-[10px] font-medium mb-1">Confidence</h3>
      <div className="flex-1 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="65%"
              startAngle={180}
              endAngle={0}
              innerRadius="50%"
              outerRadius="80%"
              paddingAngle={3}
              dataKey="value"
            >
              <Cell key="cell-0" fill={value > 70 ? '#171717' : value > 40 ? '#737373' : '#a3a3a3'} />
              <Cell key="cell-1" fill="#f5f5f5" />
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="text-center -mt-3">
        <div className="text-lg font-semibold text-neutral-900">{value}%</div>
        <div className="text-[9px] text-neutral-400">Certainty</div>
      </div>
    </div>
  );
};
