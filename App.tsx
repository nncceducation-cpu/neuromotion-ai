
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { PipelineStage, AnalysisReport, User, SavedReport, MovementMetrics, SkeletonFrame, ExpertCorrection, Point3D, MotionConfig } from './types';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { EntropyChart, FluencyChart, FractalChart, PhaseSpaceChart, KineticEnergyChart } from './components/Charts';
import { ReportView } from './components/ReportView';
import { Dashboard } from './components/Dashboard';
import { ComparisonView } from './components/ComparisonView';
import { ValidationView } from './components/ValidationView';
import { storageService } from './services/storage';
import { GraphsView } from './components/GraphsView';
import { getProfileBySeed, SERVER_URL, ProfileInfo } from './constants';

const DEFAULT_CONFIG: MotionConfig = {
  sensitivity: 0.85,
  windowSize: 30,
  entropyThreshold: 0.4,
  jerkThreshold: 5.0,
  rhythmicityWeight: 0.7,
  stiffnessThreshold: 0.6
};

// Helper for deterministic hashing
const stringToHash = (str: string) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return hash;
};

// --- INTERPOLATION HELPERS FOR SMOOTH ANIMATION ---
const lerp = (start: number, end: number, t: number) => start * (1 - t) + end * t;

const interpolateSkeleton = (f1: SkeletonFrame, f2: SkeletonFrame, t: number): SkeletonFrame => {
    if (!f1 || !f2) return f1 || f2;
    const joints: any = {};
    const keys = Object.keys(f1.joints) as Array<keyof typeof f1.joints>;
    
    keys.forEach(key => {
        const p1 = f1.joints[key];
        const p2 = f2.joints[key];
        if (p1 && p2) {
            joints[key] = {
                x: lerp(p1.x, p2.x, t),
                y: lerp(p1.y, p2.y, t),
                z: lerp(p1.z, p2.z, t),
                visibility: lerp(p1.visibility ?? 1, p2.visibility ?? 1, t)
            };
        } else {
            joints[key] = p1 || p2;
        }
    });

    return {
        timestamp: lerp(f1.timestamp, f2.timestamp, t),
        joints: joints
    };
};

const calculateLocalFrequency = (frames: SkeletonFrame[], targetJoint: 'right_wrist' | 'left_wrist', fps: number = 30) => {
    if (frames.length < 5) return { freq: 0, amp: 0 };
    const values = frames.map(f => f.joints[targetJoint].y);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const amp = maxVal - minVal;
    let peaks = 0;
    const mean = values.reduce((a,b)=>a+b,0) / values.length;
    for(let i=1; i<values.length-1; i++) {
        if ((values[i] > values[i-1] && values[i] > values[i+1]) && values[i] > mean) {
            peaks++;
        }
    }
    const duration = (frames[frames.length-1].timestamp - frames[0].timestamp) || 1;
    const freq = peaks / duration;
    return { freq, amp };
};

const VideoOverlay: React.FC<{
  videoRef: React.RefObject<HTMLVideoElement>;
  rawFrames: SkeletonFrame[];
  realTimeSkeleton: SkeletonFrame | null;
  isCapturing: boolean;
  isLive: boolean;
  stage: PipelineStage;
  timeRemaining?: string;
}> = ({ videoRef, rawFrames, realTimeSkeleton, isCapturing, isLive, stage, timeRemaining }) => {
    const [displayFrame, setDisplayFrame] = useState<SkeletonFrame | null>(null);

    useEffect(() => {
        let animationFrameId: number;
        const animate = () => {
            if (isCapturing || isLive) {
            } else if (rawFrames.length > 0 && videoRef.current) {
                const video = videoRef.current;
                if (!video.paused && !video.ended && video.duration > 0) {
                    const currentTime = video.currentTime;
                    let nextIdx = rawFrames.findIndex(f => f.timestamp > currentTime);
                    let frame = null;
                    if (nextIdx === -1) {
                        frame = rawFrames[rawFrames.length - 1];
                    } else if (nextIdx === 0) {
                        frame = rawFrames[0];
                    } else {
                        const f1 = rawFrames[nextIdx - 1];
                        const f2 = rawFrames[nextIdx];
                        const dt = f2.timestamp - f1.timestamp;
                        const t = dt > 0.001 ? (currentTime - f1.timestamp) / dt : 0;
                        frame = interpolateSkeleton(f1, f2, Math.max(0, Math.min(1, t)));
                    }
                    setDisplayFrame(frame);
                }
            }
            animationFrameId = requestAnimationFrame(animate);
        };

        if (!isCapturing && (stage === PipelineStage.MOVEMENT_LAB || stage === PipelineStage.CLASSIFIER || stage === PipelineStage.COMPLETE)) {
            animationFrameId = requestAnimationFrame(animate);
        }
        return () => cancelAnimationFrame(animationFrameId);
    }, [rawFrames, stage, isCapturing, isLive, videoRef]);

    const frame = (isCapturing || isLive) ? realTimeSkeleton : displayFrame;
    
    let rhythmStats = { freq: 0, amp: 0 };
    let activeJoint: 'right_wrist' | 'left_wrist' = 'right_wrist';
    
    if (frame && !isCapturing && !isLive && rawFrames.length > 0 && videoRef.current) {
         const currentTime = videoRef.current.currentTime;
         let nextIdx = rawFrames.findIndex(f => f.timestamp > currentTime);
         const endIdx = nextIdx === -1 ? rawFrames.length : nextIdx;
         const startIdx = Math.max(0, endIdx - 30); 
         const windowFrames = rawFrames.slice(startIdx, endIdx);
         if (windowFrames.length > 5) {
             rhythmStats = calculateLocalFrequency(windowFrames, 'right_wrist');
             if (rhythmStats.freq < 1.5) {
                 const leftStats = calculateLocalFrequency(windowFrames, 'left_wrist');
                 if (leftStats.freq > rhythmStats.freq) {
                     rhythmStats = leftStats;
                     activeJoint = 'left_wrist';
                 }
             }
         }
    }

    if (!frame && !isCapturing && !isLive) return null;

    return (
        <div className="absolute inset-0 pointer-events-none">
            <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
                {(() => {
                    if (!frame) return null;
                    const j = frame.joints;
                    const connections = [
                        [j.left_shoulder, j.right_shoulder],
                        [j.left_shoulder, j.left_elbow], [j.left_elbow, j.left_wrist],
                        [j.right_shoulder, j.right_elbow], [j.right_elbow, j.right_wrist],
                        [j.left_shoulder, j.left_hip], [j.right_shoulder, j.right_hip],
                        [j.left_hip, j.right_hip],
                        [j.left_hip, j.left_knee], [j.left_knee, j.left_ankle],
                        [j.right_hip, j.right_knee], [j.right_knee, j.right_ankle],
                        [j.nose, j.left_eye], [j.nose, j.right_eye]
                    ];
                    
                    const isSeizureBand = rhythmStats.freq >= 1.5 && rhythmStats.freq <= 5.0 && rhythmStats.amp > 2.0;
                    const isTremorBand = rhythmStats.freq > 5.0 && rhythmStats.amp > 1.0;
                    const activeJointPt = activeJoint === 'right_wrist' ? j.right_wrist : j.left_wrist;

                    return (
                        <>
                            {connections.map(([a, b], i) => 
                                (a && b && (a.visibility || 1) > 0.5 && (b.visibility || 1) > 0.5) && (
                                <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="rgba(255,255,255,0.6)" strokeWidth="0.5" />
                            ))}
                            {Object.values(j).map((val, i) => {
                                const pt = val as Point3D | undefined;
                                return (pt && (pt.visibility || 1) > 0.5) && (
                                <circle key={i} cx={pt.x} cy={pt.y} r="0.8" fill="#ffffff" />
                            )})}
                            {activeJointPt && (isSeizureBand || isTremorBand) && (
                                <>
                                    <circle cx={activeJointPt.x} cy={activeJointPt.y} r="4" fill="none" stroke={isSeizureBand ? "#ef4444" : "#eab308"} strokeWidth="0.8" className="animate-ping" opacity="0.8"/>
                                    <circle cx={activeJointPt.x} cy={activeJointPt.y} r="4" fill="none" stroke={isSeizureBand ? "#ef4444" : "#eab308"} strokeWidth="0.5" />
                                </>
                            )}
                        </>
                    );
                })()}
            </svg>
            {stage !== PipelineStage.INGESTION && (
                <div className="absolute top-4 left-4 bg-black/60 backdrop-blur text-white px-3 py-1 rounded-md text-xs font-mono flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${isCapturing ? 'bg-white animate-pulse' : 'bg-neutral-400'}`}></div>
                    {isCapturing ? 'PROCESSING VISION...' : 'PLAYBACK'}
                </div>
            )}
            {!isCapturing && !isLive && rawFrames.length > 0 && rhythmStats.freq > 1.5 && rhythmStats.amp > 1.0 && (
                <div className="absolute top-4 right-4 flex flex-col items-end gap-2">
                     {(() => {
                         const isSeizure = rhythmStats.freq >= 1.5 && rhythmStats.freq <= 5.0 && rhythmStats.amp > 2.0;
                         return (
                             <div className={`px-4 py-2 rounded-md backdrop-blur border bg-neutral-900/90 border-neutral-500 text-white`}>
                                 <div className="text-[10px] font-bold uppercase tracking-wider mb-1 flex items-center gap-1.5">
                                     <div className={`w-1.5 h-1.5 rounded-full ${isSeizure ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
                                     {isSeizure ? 'RHYTHMIC BURST (SEIZURE BAND)' : 'TREMOR ACTIVITY'}
                                 </div>
                                 <div className="flex items-end gap-2">
                                     <span className="text-2xl font-mono font-bold">{rhythmStats.freq.toFixed(1)}</span>
                                     <span className="text-sm font-medium mb-1 opacity-80">Hz</span>
                                 </div>
                                 <div className="w-full bg-black/30 h-1 mt-2 rounded-full overflow-hidden">
                                     <div className="h-full bg-white/80" style={{width: `${Math.min(100, (rhythmStats.amp / 10) * 100)}%`}}></div>
                                 </div>
                                 <div className="text-[9px] mt-1 opacity-60">Amplitude Index</div>
                             </div>
                         );
                     })()}
                </div>
            )}
            {isLive && stage === PipelineStage.INGESTION && (
                <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur px-4 py-2 rounded-md text-white border border-white/10">
                    <div className="text-[10px] uppercase font-medium text-neutral-300 mb-1 flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-white rounded-full animate-pulse"></div>
                        Live Monitor Active
                    </div>
                    <div className="font-mono text-xl font-bold tracking-widest">
                        {timeRemaining || "05:00"}
                    </div>
                    <div className="text-[9px] text-neutral-400 mt-0.5">Until next auto-assessment</div>
                </div>
            )}
        </div>
    );
};

const App: React.FC = () => {
  const [user, setUser] = useState<User>({ id: 'default-clinician', name: 'Clinical Specialist', email: 'lab@neuromotion.ai' });
  const [view, setView] = useState('dashboard');
  const [appMode, setAppMode] = useState<'clinical' | 'training'>('clinical');
  const [stage, setStage] = useState<PipelineStage>(PipelineStage.IDLE);
  const [file, setFile] = useState<File | null>(null);
  const [isLive, setIsLive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const [activeProfile, setActiveProfile] = useState<ProfileInfo | null>(null);
  const [rawFrames, setRawFrames] = useState<SkeletonFrame[]>([]); 
  const [chartData, setChartData] = useState<MovementMetrics[]>([]);
  const [selectedReport, setSelectedReport] = useState<SavedReport | null>(null);
  const [reportsToCompare, setReportsToCompare] = useState<SavedReport[]>([]);
  const [isRetraining, setIsRetraining] = useState(false);
  const [simulationSeed, setSimulationSeed] = useState<number>(0);
  const [showTrainingModal, setShowTrainingModal] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<'idle' | 'analyzing' | 'complete'>('idle');
  const [expertAnnotation, setExpertAnnotation] = useState("");
  const [selectedExpertDiagnosis, setSelectedExpertDiagnosis] = useState<string | null>(null);
  const [motionConfig, setMotionConfig] = useState<MotionConfig>(DEFAULT_CONFIG);
  const [prevConfig, setPrevConfig] = useState<MotionConfig | null>(null);
  const [realTimeSkeleton, setRealTimeSkeleton] = useState<SkeletonFrame | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [nextAutoScan, setNextAutoScan] = useState<number>(0);
  const [timeRemaining, setTimeRemaining] = useState<string>("");

  useEffect(() => {
    const savedConfig = localStorage.getItem(`motionConfig_${user.id}`);
    if (savedConfig) {
        try { setMotionConfig(JSON.parse(savedConfig)); } catch (e) { console.error("Failed to load saved config"); }
    }
  }, []);

  useEffect(() => {
    if (videoRef.current && cameraStream) {
        videoRef.current.srcObject = cameraStream;
        videoRef.current.play().catch(e => console.log("Autoplay handled", e));
    }
  }, [cameraStream, stage, view, isLive]);

  const stopCamera = () => { if (cameraStream) { cameraStream.getTracks().forEach(track => track.stop()); setCameraStream(null); } };
  
  const resetPipeline = () => {
    stopCamera(); setIsLive(false); setStage(PipelineStage.IDLE); setFile(null); setVideoPreview(null);
    setReport(null); setError(null); setActiveProfile(null); setChartData([]); setRawFrames([]);
    setIsRetraining(false); setSelectedReport(null); setRealTimeSkeleton(null);
    setIsCapturing(false); setNextAutoScan(0);
    if (appMode !== 'training') { setExpertAnnotation(""); setSelectedExpertDiagnosis(null); }
  };

  const startNewAnalysis = () => { setAppMode('clinical'); resetPipeline(); setStage(PipelineStage.INGESTION); setView('pipeline'); };
  const startTrainingMode = () => { setAppMode('training'); resetPipeline(); setStage(PipelineStage.INGESTION); setView('pipeline'); };
  
  const startLiveAnalysis = async () => {
    setAppMode('clinical'); resetPipeline(); setIsLive(true); setView('pipeline');
    setNextAutoScan(Date.now() + 300000);
    const seed = Date.now(); setSimulationSeed(seed);
    const profile = await getProfileBySeed(seed); setActiveProfile(profile);
    setRawFrames([]); setChartData([]);
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1080 }, height: { ideal: 1920 } } });
        setCameraStream(stream); setStage(PipelineStage.INGESTION);
    } catch (e) { setError("Could not access camera. Please allow permissions."); setIsLive(false); setView('dashboard'); }
  };

  const startComparisonMode = () => { setReportsToCompare([]); resetPipeline(); setView('comparison'); };
  const handleCompareReports = (reports: SavedReport[]) => { setReportsToCompare(reports); resetPipeline(); setView('comparison'); };
  const handleViewReport = (savedReport: SavedReport) => { setSelectedReport(savedReport); setReport(savedReport); setView('view_report'); };

  // Helper: build AnalysisReport from backend response + per-frame metrics
  const buildReportFromBackend = (backendReport: any, metrics: any[]): AnalysisReport => {
    const avg = (key: string) => metrics.length > 0 ? metrics.reduce((s: number, r: any) => s + (Number(r[key]) || 0), 0) / metrics.length : 0;
    return {
      classification: backendReport.classification || "Normal",
      confidence: backendReport.confidence ?? 50,
      seizureDetected: backendReport.seizureDetected ?? (backendReport.classification === 'Seizures'),
      seizureType: backendReport.seizureType || "None",
      differentialAlert: backendReport.differentialAlert || undefined,
      rawData: {
        entropy: avg('entropy'),
        fluency: avg('fluency_jerk'),
        complexity: avg('fractal_dim'),
        variabilityIndex: 0,
        csRiskScore: 0,
        avg_kinetic_energy: avg('kinetic_energy'),
        avg_root_stress: avg('root_stress'),
        posture: { shoulder_flexion_index: 0, hip_flexion_index: 0, symmetry_score: 1, tone_label: 'Normal' as const, frog_leg_score: 0, spontaneous_activity: 0, sustained_posture_score: 0, crying_index: 0, eye_openness_index: 0, arousal_index: 0, state_transition_probability: 0 },
        seizure: { rhythmicity_score: 0, stiffness_score: 0, eye_deviation_score: 0, dominant_frequency: 0, limb_synchrony: 0, calculated_type: 'None' as const }
      },
      clinicalAnalysis: backendReport.clinicalAnalysis || backendReport.reasoning || "Analysis via backend",
      recommendations: Array.isArray(backendReport.recommendations) ? backendReport.recommendations : backendReport.recommendations ? [backendReport.recommendations] : [],
      timelineData: metrics
    };
  };

  const handleSaveCorrection = async (correction: ExpertCorrection) => {
    if (!selectedReport) return;
    setIsRetraining(true);
    await storageService.saveExpertCorrection(user.id, selectedReport.id, correction);
  };

  const handleTrainingSubmit = async () => {
    if (!selectedExpertDiagnosis) { alert("Please select a diagnosis."); return; }
    setTrainingStatus('analyzing');
    try {
        // Step 1: Ask backend to optimize physics config
        const refineResponse = await fetch(`${SERVER_URL}/refine_config`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_report: report ? { classification: report.classification, rawData: report.rawData } : null,
                expert_diagnosis: selectedExpertDiagnosis,
                annotation: expertAnnotation,
                current_config: motionConfig
            })
        });
        if (!refineResponse.ok) throw new Error("Config refinement failed");
        const newConfig = await refineResponse.json() as MotionConfig;
        setPrevConfig(motionConfig); setMotionConfig(newConfig);
        localStorage.setItem(`motionConfig_${user.id}`, JSON.stringify(newConfig));

        // Step 2: Re-analyze with new config if we have frames
        if (rawFrames.length > 0) {
            const analyzeResponse = await fetch(`${SERVER_URL}/analyze_frames`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ frames: rawFrames, config: newConfig })
            });
            if (!analyzeResponse.ok) throw new Error("Re-analysis failed");
            const backendData = await analyzeResponse.json();

            if (backendData.metrics) {
                setChartData(backendData.metrics);
                const completeReport = buildReportFromBackend(backendData.report, backendData.metrics);
                const updatedReport: SavedReport = {
                    id: selectedReport?.id || crypto.randomUUID(),
                    date: selectedReport?.date || new Date().toISOString(),
                    videoName: selectedReport?.videoName || (file ? file.name : "Training Session"),
                    ...completeReport
                };
                setReport(updatedReport);
                setSelectedReport(updatedReport);

                const correction: ExpertCorrection = { correctClassification: selectedExpertDiagnosis as any, notes: expertAnnotation, timestamp: new Date().toISOString(), clinicianName: user.name };
                await storageService.saveExpertCorrection(user.id, updatedReport.id, correction);
            }
        }
        setTrainingStatus('complete');
    } catch (e) { console.error(e); setTrainingStatus('idle'); alert("Failed to train model."); }
  };

  const closeTrainingModal = () => { setShowTrainingModal(false); setTrainingStatus('idle'); setExpertAnnotation(""); setSelectedExpertDiagnosis(null); setPrevConfig(null); };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const uploadedFile = e.target.files[0];
      setFile(uploadedFile); setVideoPreview(URL.createObjectURL(uploadedFile)); setIsLive(false);
      const hash = stringToHash(uploadedFile.name); setSimulationSeed(hash);
      const profile = await getProfileBySeed(hash); setActiveProfile(profile);
      setRawFrames([]); setChartData([]);
      setStage(PipelineStage.INGESTION); setReport(null); setError(null);
    }
  };

  const runAnalysis = useCallback(async () => {
    if (!file || !user) {
      if (!file) setError("Please select a video file.");
      return;
    }

    try {
      setStage(PipelineStage.LIFTING_3D);
      setIsCapturing(true);

      const formData = new FormData();
      formData.append('file', file);

      setStage(PipelineStage.MOVEMENT_LAB);
      const response = await fetch(`${SERVER_URL}/upload_video`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({ detail: 'Server error' }));
        throw new Error(errBody.detail || 'Backend Error');
      }

      const backendData = await response.json();

      if (backendData.metrics) {
        setChartData(backendData.metrics);
        setStage(PipelineStage.CLASSIFIER);

        const completeReport = buildReportFromBackend(backendData.report, backendData.metrics);
        const savedReport = await storageService.saveReport(user.id, completeReport, file.name);
        await new Promise(resolve => setTimeout(resolve, 500));
        setStage(PipelineStage.COMPLETE); setReport(completeReport); setSelectedReport(savedReport);
        if (appMode === 'training') { setShowTrainingModal(true); }
      }
    } catch (err) {
      console.error('Analysis failed:', err);
      setError(`Server error: ${err instanceof Error ? err.message : 'Unknown'}. Ensure backend is running.`);
      setStage(PipelineStage.IDLE);
    } finally {
      setIsCapturing(false);
    }
  }, [file, user, appMode]);

  useEffect(() => {
     if (isLive && stage === PipelineStage.INGESTION && !isCapturing && nextAutoScan > 0) {
         const interval = setInterval(() => {
             const now = Date.now(); const remaining = nextAutoScan - now;
             if (remaining <= 0) { clearInterval(interval); runAnalysis(); } 
             else { const mins = Math.floor((remaining / 1000) / 60); const secs = Math.floor((remaining / 1000) % 60); setTimeRemaining(`${mins}:${secs < 10 ? '0' : ''}${secs}`); }
         }, 1000);
         return () => clearInterval(interval);
     }
  }, [isLive, stage, isCapturing, nextAutoScan, runAnalysis]);

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-800 font-sans">
      <header className="fixed top-0 w-full z-50 bg-white/90 backdrop-blur-md border-b border-neutral-200 no-print">
         <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
            <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView('dashboard')}>
                <div className="bg-neutral-900 w-7 h-7 rounded-md flex items-center justify-center text-white font-semibold text-xs">N</div>
                <span className="font-semibold text-sm tracking-tight text-neutral-900">NeuroMotion AI</span>
            </div>
            <div className="flex items-center gap-3">
               {appMode === 'training' && ( <div className="bg-neutral-900 text-white text-xs px-3 py-1 rounded-md font-medium flex items-center"><i className="fas fa-graduation-cap mr-2"></i> Training</div> )}
               <div className="flex items-center gap-2">
                   <div className="w-7 h-7 rounded-full bg-neutral-200 flex items-center justify-center text-neutral-500 text-xs font-semibold">{user.name.charAt(0)}</div>
                   <span className="text-sm font-medium hidden md:block text-neutral-700">{user.name}</span>
               </div>
               <button onClick={() => { storageService.logout(); window.location.reload(); }} className="text-neutral-400 hover:text-neutral-600 transition-colors"><i className="fas fa-sign-out-alt text-sm"></i></button>
            </div>
         </div>
      </header>
      <main className="pt-20 pb-12 px-4 max-w-7xl mx-auto">
         {view === 'dashboard' && ( <Dashboard user={user} onNewAnalysis={startNewAnalysis} onLiveAnalysis={startLiveAnalysis} onTrainingMode={startTrainingMode} onComparisonMode={startComparisonMode} onValidationView={() => setView('validation')} onGraphsView={() => setView('graphs')} onViewReport={handleViewReport} onCompareReports={handleCompareReports} /> )}
         {view === 'graphs' && ( <GraphsView user={user} onClose={() => setView('dashboard')} /> )}
         {view === 'validation' && ( <ValidationView onClose={() => setView('dashboard')} /> )}
         {view === 'comparison' && ( <ComparisonView onBack={() => setView('dashboard')} initialReports={reportsToCompare} /> )}
         {view === 'view_report' && selectedReport && ( <ReportView report={selectedReport} onClose={() => setView('dashboard')} onSaveCorrection={handleSaveCorrection} /> )}
         {view === 'pipeline' && (
             <div className="animate-fade-in">
                <div className="mb-6 flex items-center justify-between no-print">
                    <button onClick={() => setView('dashboard')} className="text-neutral-400 hover:text-neutral-800 flex items-center gap-2 transition-colors text-sm"><i className="fas fa-arrow-left"></i> Back to Dashboard</button>
                    {appMode === 'training' && ( <div className="text-sm text-neutral-500 font-medium">Step 1: Upload & Diagnose (Expert) &rarr; Step 2: AI Analysis & Optimization</div> )}
                </div>
                {appMode === 'training' && stage === PipelineStage.INGESTION && (
                    <div className="bg-neutral-50 border border-neutral-200 rounded-md p-6 mb-8">
                        <h3 className="text-neutral-900 font-semibold mb-4 flex items-center text-sm"><i className="fas fa-clipboard-check mr-2 text-neutral-400"></i> Expert Ground Truth Input</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-medium uppercase text-neutral-700 mb-1">Diagnosis</label>
                                <select className="w-full p-2 rounded-md border border-neutral-200 text-sm focus:ring-2 focus:ring-neutral-900 outline-none" onChange={(e) => setSelectedExpertDiagnosis(e.target.value)}>
                                    <option value="">Select Diagnosis...</option>
                                    <option value="Normal">Normal</option>
                                    <option value="Sarnat Stage I">Sarnat Stage I</option>
                                    <option value="Sarnat Stage II">Sarnat Stage II</option>
                                    <option value="Sarnat Stage III">Sarnat Stage III</option>
                                    <option value="Seizures">Seizures</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-xs font-medium uppercase text-neutral-700 mb-1">Notes</label>
                                <input className="w-full p-2 rounded-md border border-neutral-200 text-sm focus:ring-2 focus:ring-neutral-900 outline-none" placeholder="E.g. High frequency tremor noted..." onChange={(e) => setExpertAnnotation(e.target.value)} />
                            </div>
                        </div>
                    </div>
                )}
                <div className="no-print"><PipelineVisualizer currentStage={stage} /></div>
                {stage === PipelineStage.COMPLETE && report ? ( <ReportView report={report} onClose={() => setView('dashboard')} onSaveCorrection={handleSaveCorrection} /> ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
                        <div className="lg:col-span-2 space-y-4 flex flex-col items-center">
                            <div className="relative bg-black rounded-md overflow-hidden group w-auto inline-block border border-neutral-200">
                                <video ref={videoRef} className="block max-h-[75vh] w-auto max-w-full" src={videoPreview || undefined} controls={stage === PipelineStage.COMPLETE || stage === PipelineStage.INGESTION} playsInline muted />
                                {stage !== PipelineStage.INGESTION && stage !== PipelineStage.IDLE && ( <VideoOverlay videoRef={videoRef} rawFrames={rawFrames} realTimeSkeleton={realTimeSkeleton} isCapturing={isCapturing} isLive={isLive} stage={stage} timeRemaining={timeRemaining} /> )}
                                {stage === PipelineStage.INGESTION && !videoPreview && !isLive && (
                                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-neutral-900/60">
                                        <i className="fas fa-cloud-upload-alt text-4xl mb-4 opacity-70"></i>
                                        <label className="bg-white text-neutral-900 hover:bg-neutral-100 px-6 py-2 rounded-md cursor-pointer transition-colors font-medium text-sm">
                                            Select Video File
                                            <input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} />
                                        </label>
                                        <p className="mt-4 text-sm opacity-50">Supported formats: MP4, MOV, WEBM</p>
                                    </div>
                                )}
                            </div>
                            {stage === PipelineStage.INGESTION && (videoPreview || isLive) && (
                                <button onClick={runAnalysis} className="w-full py-3 bg-neutral-900 hover:bg-neutral-800 text-white rounded-md font-medium text-sm transition-colors flex items-center justify-center gap-3">
                                    {isCapturing ? ( <><i className="fas fa-circle-notch fa-spin"></i> Analyzing...</> ) : ( <><i className="fas fa-play"></i> Run Clinical Analysis</> )}
                                </button>
                            )}
                        </div>
                        <div className="space-y-4">
                            <EntropyChart data={chartData} />
                            <PhaseSpaceChart data={chartData} />
                            <div className="grid grid-cols-2 gap-4"><FluencyChart data={chartData} /><FractalChart data={chartData} /></div>
                            <KineticEnergyChart data={chartData} />
                        </div>
                    </div>
                )}
             </div>
         )}
      </main>
      {showTrainingModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
            <div className="bg-white rounded-md shadow-lg max-w-2xl w-full overflow-hidden border border-neutral-200">
                <div className="p-6 border-b border-neutral-100 flex justify-between items-center bg-neutral-50">
                    <h3 className="text-lg font-semibold text-neutral-900"><i className="fas fa-brain text-neutral-400 mr-2"></i> {trainingStatus === 'complete' ? 'Optimization Complete' : 'Expert Training'}</h3>
                    <button onClick={closeTrainingModal} className="text-neutral-400 hover:text-neutral-600"><i className="fas fa-times"></i></button>
                </div>
                <div className="p-6">
                    {trainingStatus === 'analyzing' ? (
                        <div className="text-center py-12"><i className="fas fa-microchip text-4xl text-neutral-300 animate-pulse mb-4"></i><h4 className="text-lg font-semibold text-neutral-900">Refining Physics Engine...</h4><p className="text-neutral-500 text-sm">The AI is adjusting signal thresholds based on your feedback.</p></div>
                    ) : trainingStatus === 'complete' && prevConfig && motionConfig ? (
                        <div className="space-y-6">
                            <div className="bg-neutral-50 text-neutral-800 p-4 rounded-md flex items-start gap-3 border border-neutral-200"><i className="fas fa-check-circle mt-1 text-neutral-400"></i><div><h4 className="font-semibold text-sm">Algorithm Updated</h4><p className="text-sm text-neutral-500">New parameters have been saved to memory.</p></div></div>
                            <div className="bg-neutral-900 rounded-md p-4 text-xs font-mono text-neutral-300">
                                <div className="grid grid-cols-3 border-b border-neutral-700 pb-2 mb-2 font-medium text-neutral-500"><div>PARAMETER</div><div>WAS</div><div>NOW</div></div>
                                {Object.entries(motionConfig).map(([key, val]) => { const old = (prevConfig as any)[key]; if (typeof val !== 'number') return null; const diff = val - old; const changed = Math.abs(diff) > 0.001; return ( <div key={key} className={`grid grid-cols-3 py-1 ${changed ? 'text-white bg-white/5' : 'opacity-50'}`}><div>{key}</div><div>{Number(old).toFixed(2)}</div><div className={changed ? 'text-white' : ''}>{Number(val).toFixed(2)} {changed && ` (${diff > 0 ? '+' : ''}${diff.toFixed(2)})`}</div></div> ); })}
                            </div>
                            <button onClick={closeTrainingModal} className="w-full bg-neutral-900 hover:bg-neutral-800 text-white py-3 rounded-md font-medium text-sm">Close & Save Results</button>
                        </div>
                    ) : ( <div><p>Initializing...</p></div> )}
                </div>
            </div>
        </div>
      )}
      {error && ( <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-neutral-900 text-white px-5 py-3 rounded-md shadow-lg flex items-center gap-3 z-[200] text-sm"><div className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0"></div>{error}<button onClick={() => setError(null)} className="ml-2 opacity-70 hover:opacity-100"><i className="fas fa-times text-xs"></i></button></div> )}
    </div>
  );
};

export default App;