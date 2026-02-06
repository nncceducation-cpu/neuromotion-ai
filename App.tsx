
import React, { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { PipelineStage, AnalysisReport, User, SavedReport, MovementMetrics, ClinicalProfile, SkeletonFrame, ExpertCorrection, Point3D, MotionConfig, AggregatedStats } from './types';
import { PipelineVisualizer } from './components/PipelineVisualizer';
import { EntropyChart, FluencyChart, FractalChart, PhaseSpaceChart, KineticEnergyChart } from './components/Charts';
import { ReportView } from './components/ReportView';
import { Dashboard } from './components/Dashboard';
import { ComparisonView } from './components/ComparisonView';
import { generateGMAReport, refineModelParameters } from './services/geminiService';
import { storageService } from './services/storage';
import { physicsEngine } from './services/physics'; 
import { getRandomProfile, getProfileBySeed, setSeed, SERVER_URL } from './constants';

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
                                <circle key={i} cx={pt.x} cy={pt.y} r="0.8" fill={i < 11 ? '#ef4444' : i % 2 === 0 ? '#f97316' : '#06b6d4'} />
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
                <div className="absolute top-4 left-4 bg-black/60 backdrop-blur text-white px-3 py-1 rounded-full text-xs font-mono flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${isCapturing ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                    {isCapturing ? 'PROCESSING VISION...' : 'PLAYBACK'}
                </div>
            )}
            {!isCapturing && !isLive && rawFrames.length > 0 && rhythmStats.freq > 1.5 && rhythmStats.amp > 1.0 && (
                <div className="absolute top-4 right-4 flex flex-col items-end gap-2">
                     {(() => {
                         const isSeizure = rhythmStats.freq >= 1.5 && rhythmStats.freq <= 5.0 && rhythmStats.amp > 2.0;
                         return (
                             <div className={`px-4 py-2 rounded-lg backdrop-blur border ${isSeizure ? 'bg-red-900/80 border-red-500 text-white' : 'bg-yellow-900/80 border-yellow-500 text-white'}`}>
                                 <div className="text-[10px] font-bold uppercase tracking-wider mb-1">
                                     {isSeizure ? '⚠️ RHYTHMIC BURST (SEIZURE BAND)' : '⚡ TREMOR ACTIVITY'}
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
                <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur px-4 py-2 rounded-lg text-white border border-white/20 shadow-lg">
                    <div className="text-[10px] uppercase font-bold text-slate-300 mb-1 flex items-center gap-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        Live Monitor Active
                    </div>
                    <div className="font-mono text-xl font-bold tracking-widest">
                        {timeRemaining || "05:00"}
                    </div>
                    <div className="text-[9px] text-slate-400 mt-0.5">Until next auto-assessment</div>
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
  const [activeProfile, setActiveProfile] = useState<ClinicalProfile | null>(null);
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
  const [holistic, setHolistic] = useState<any>(null);
  const [realTimeSkeleton, setRealTimeSkeleton] = useState<SkeletonFrame | null>(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const captureFramesRef = useRef<SkeletonFrame[]>([]);
  const isCollectingRef = useRef(false);
  const currentFrameTimeRef = useRef(0);
  const [nextAutoScan, setNextAutoScan] = useState<number>(0);
  const [timeRemaining, setTimeRemaining] = useState<string>("");

  // --- NEW: SERVER MODE STATE ---
  const [useBackend, setUseBackend] = useState(false);

  useEffect(() => {
    const savedConfig = localStorage.getItem(`motionConfig_${user.id}`);
    if (savedConfig) {
        try { setMotionConfig(JSON.parse(savedConfig)); } catch (e) { console.error("Failed to load saved config"); }
    }
    if ((window as any).Holistic) {
      const h = new (window as any).Holistic({ locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic@0.5.1675471629/${file}` });
      h.setOptions({ modelComplexity: 1, smoothLandmarks: true, enableSegmentation: false, refineFaceLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
      h.onResults(onHolisticResults);
      setHolistic(h);
    }
  }, []);

  const onHolisticResults = useCallback((results: any) => {
    if (!results.poseLandmarks) return;
    const mp = (lm: any) => ({ x: lm.x * 100, y: lm.y * 100, z: lm.z * 100, visibility: lm.visibility });
    const frame: SkeletonFrame = {
        timestamp: currentFrameTimeRef.current,
        joints: {
            nose: results.poseLandmarks[0] ? mp(results.poseLandmarks[0]) : undefined,
            left_eye: results.poseLandmarks[2] ? mp(results.poseLandmarks[2]) : undefined,
            right_eye: results.poseLandmarks[5] ? mp(results.poseLandmarks[5]) : undefined,
            left_mouth: results.poseLandmarks[9] ? mp(results.poseLandmarks[9]) : undefined,
            right_mouth: results.poseLandmarks[10] ? mp(results.poseLandmarks[10]) : undefined,
            left_shoulder: mp(results.poseLandmarks[11]),
            right_shoulder: mp(results.poseLandmarks[12]),
            left_elbow: mp(results.poseLandmarks[13]),
            right_elbow: mp(results.poseLandmarks[14]),
            left_wrist: mp(results.poseLandmarks[15]),
            right_wrist: mp(results.poseLandmarks[16]),
            left_hip: mp(results.poseLandmarks[23]),
            right_hip: mp(results.poseLandmarks[24]),
            left_knee: mp(results.poseLandmarks[25]),
            right_knee: mp(results.poseLandmarks[26]),
            left_ankle: mp(results.poseLandmarks[27]),
            right_ankle: mp(results.poseLandmarks[28])
        }
    };
    setRealTimeSkeleton(frame);
    if (captureFramesRef.current.length < 3000 && isCollectingRef.current) {
        captureFramesRef.current.push(frame);
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
    captureFramesRef.current = []; setIsCapturing(false); isCollectingRef.current = false; currentFrameTimeRef.current = 0;
    setNextAutoScan(0);
    if (appMode !== 'training') { setExpertAnnotation(""); setSelectedExpertDiagnosis(null); }
  };

  const startNewAnalysis = () => { setAppMode('clinical'); resetPipeline(); setStage(PipelineStage.INGESTION); setView('pipeline'); };
  const startTrainingMode = () => { setAppMode('training'); resetPipeline(); setStage(PipelineStage.INGESTION); setView('pipeline'); };
  
  const startLiveAnalysis = async () => {
    setAppMode('clinical'); resetPipeline(); setIsLive(true); setView('pipeline');
    setNextAutoScan(Date.now() + 300000);
    const seed = Date.now(); setSimulationSeed(seed); setSeed(seed); 
    const profile = getProfileBySeed(seed); setActiveProfile(profile);
    setRawFrames([]); setChartData([]);
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1080 }, height: { ideal: 1920 } } });
        setCameraStream(stream); setStage(PipelineStage.INGESTION);
    } catch (e) { setError("Could not access camera. Please allow permissions."); setIsLive(false); setView('dashboard'); }
  };

  const startComparisonMode = () => { setReportsToCompare([]); resetPipeline(); setView('comparison'); };
  const handleCompareReports = (reports: SavedReport[]) => { setReportsToCompare(reports); resetPipeline(); setView('comparison'); };
  const handleViewReport = (savedReport: SavedReport) => { setSelectedReport(savedReport); setReport(savedReport); setView('view_report'); };

  const calculateBiomarkers = (data: MovementMetrics[], frames: SkeletonFrame[], config?: MotionConfig) => {
    if (data.length === 0) return null;
    const meanEntropy = data.reduce((sum, d) => sum + d.entropy, 0) / data.length;
    const peakEntropy = Math.max(...data.map(d => d.entropy));
    const meanFractal = data.reduce((sum, d) => sum + d.fractal_dim, 0) / data.length;
    const peakFractal = Math.max(...data.map(d => d.fractal_dim));
    const meanJerk = data.reduce((sum, d) => sum + d.fluency_jerk, 0) / data.length;
    const variancePos = data.reduce((sum, d) => sum + Math.pow(d.phase_x - (data.reduce((a,b)=>a+b.phase_x,0)/data.length), 2), 0) / data.length;
    const stdPos = Math.sqrt(variancePos);
    const varianceVel = data.reduce((sum, d) => sum + Math.pow(d.fluency_velocity - (data.reduce((a,b)=>a+b.fluency_velocity,0)/data.length), 2), 0) / data.length;
    const stdVel = Math.sqrt(varianceVel);
    const meanKineticEnergy = data.reduce((sum, d) => sum + d.kinetic_energy, 0) / data.length;
    const meanRootStress = data.reduce((sum, d) => sum + d.root_stress, 0) / data.length;

    let csAnalysis = { riskScore: 0 };
    if (frames.length > 0) csAnalysis = physicsEngine.detectCrampedSynchronized(frames, 10);

    let postureMetrics = { shoulder_flexion_index: 0, hip_flexion_index: 0, symmetry_score: 1, tone_label: 'Normal', frog_leg_score: 0, spontaneous_activity: 0, crying_index: 0, eye_openness_index: 0, arousal_index: 0, state_transition_probability: 0 };
    if (frames.length > 0) postureMetrics = physicsEngine.calculatePostureMetrics(frames, data) as any;

    let seizureMetrics = { rhythmicity_score: 0, stiffness_score: 0, eye_deviation_score: 0, dominant_frequency: 0, limb_synchrony: 0 };
    if (frames.length > 0) seizureMetrics = physicsEngine.detectSeizureSignatures(frames, data, config);
    
    return {
      average_sample_entropy: meanEntropy, peak_sample_entropy: peakEntropy, 
      average_fractal_dimension: meanFractal, peak_fractal_dimension: peakFractal, 
      average_jerk: meanJerk, amplitude_index_std: stdPos, velocity_variability_std: stdVel,
      periodicity_index: 0.5, autocorrelation_risk_score: csAnalysis.riskScore,
      posture: postureMetrics, seizure: seizureMetrics, total_frames_analyzed: data.length,
      avg_kinetic_energy: meanKineticEnergy, avg_root_stress: meanRootStress
    };
  };

  const handleSaveCorrection = async (correction: ExpertCorrection) => {
    if (!selectedReport) return;
    setIsRetraining(true);
    storageService.saveExpertCorrection(user.id, selectedReport.id, correction);
  };

  const handleTrainingSubmit = async () => {
    if (!selectedExpertDiagnosis) { alert("Please select a diagnosis."); return; }
    setTrainingStatus('analyzing');
    try {
        const newConfig = await refineModelParameters(report, selectedExpertDiagnosis, expertAnnotation, motionConfig);
        setPrevConfig(motionConfig); setMotionConfig(newConfig);
        localStorage.setItem(`motionConfig_${user.id}`, JSON.stringify(newConfig));
        if (rawFrames.length > 0) {
             let currentChartData = chartData;
             if (Math.abs(newConfig.windowSize - motionConfig.windowSize) > 1 || Math.abs(newConfig.entropyThreshold - motionConfig.entropyThreshold) > 0.05 || Math.abs(newConfig.jerkThreshold - motionConfig.jerkThreshold) > 0.1) {
                 currentChartData = physicsEngine.processSignal(rawFrames, newConfig);
                 setChartData(currentChartData);
             }
             const newBiomarkers = calculateBiomarkers(currentChartData, rawFrames, newConfig);
             const syntheticExample = {
                inputs: { ...newBiomarkers, posture: newBiomarkers?.posture, seizure: newBiomarkers?.seizure },
                groundTruth: { correctClassification: selectedExpertDiagnosis as any, notes: expertAnnotation, timestamp: new Date().toISOString(), clinicianName: user.name }
             };
             const existingExamples = storageService.getTrainingExamples();
             const result = await generateGMAReport(newBiomarkers, [...existingExamples, syntheticExample]);
             
             // Construct correct SavedReport object
             const completeReportBase = { ...result, timelineData: currentChartData };
             const updatedReport: SavedReport = {
                 id: selectedReport?.id || crypto.randomUUID(),
                 date: selectedReport?.date || new Date().toISOString(),
                 videoName: selectedReport?.videoName || (file ? file.name : "Training Session"),
                 ...completeReportBase
             };

             setReport(updatedReport); 
             setSelectedReport(updatedReport);
             
             const correction: ExpertCorrection = { correctClassification: selectedExpertDiagnosis as any, notes: expertAnnotation, timestamp: new Date().toISOString(), clinicianName: user.name };
             storageService.saveExpertCorrection(user.id, updatedReport.id, correction);
        }
        setTrainingStatus('complete');
    } catch (e) { console.error(e); setTrainingStatus('idle'); alert("Failed to train model."); }
  };

  const closeTrainingModal = () => { setShowTrainingModal(false); setTrainingStatus('idle'); setExpertAnnotation(""); setSelectedExpertDiagnosis(null); setPrevConfig(null); };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const uploadedFile = e.target.files[0];
      setFile(uploadedFile); setVideoPreview(URL.createObjectURL(uploadedFile)); setIsLive(false);
      const hash = stringToHash(uploadedFile.name); setSimulationSeed(hash); setSeed(hash); 
      const profile = getProfileBySeed(hash); setActiveProfile(profile);
      setRawFrames([]); setChartData([]); captureFramesRef.current = [];
      setStage(PipelineStage.INGESTION); setReport(null); setError(null); currentFrameTimeRef.current = 0;
    }
  };

  const runAnalysis = useCallback(async () => {
    if ((!file && !isLive) || !user || !holistic) { if (!holistic) setError("Vision model loading..."); return; }

    try {
      setStage(PipelineStage.LIFTING_3D);
      setIsCapturing(true); captureFramesRef.current = []; isCollectingRef.current = true; currentFrameTimeRef.current = 0;

      if (videoRef.current) {
          const video = videoRef.current;
          await video.play().catch(e => console.warn("Video play interrupted", e));
          const safetyTimeout = setTimeout(() => { if (isCollectingRef.current) { finishCapture(); } }, 30000);

          const processFrame = async () => {
             if (!videoRef.current || !isCollectingRef.current) return;
             if (videoRef.current.ended) { finishCapture(); return; }
             if (videoRef.current.readyState < 2) { requestAnimationFrame(processFrame); return; }
             currentFrameTimeRef.current = videoRef.current.currentTime;
             if (captureFramesRef.current.length < 3000) {
                 try { await holistic.send({ image: videoRef.current }); } catch (mpError) { console.warn("Frame Error", mpError); }
                 if (isCollectingRef.current && captureFramesRef.current.length < 3000) { requestAnimationFrame(processFrame); } else { finishCapture(); }
             } else { finishCapture(); }
          };
          
          const finishCapture = async () => {
             clearTimeout(safetyTimeout); isCollectingRef.current = false; setIsCapturing(false); setNextAutoScan(0);
             if (videoRef.current) { videoRef.current.pause(); videoRef.current.currentTime = 0; }
             const captured = captureFramesRef.current;
             const frames = captured.length > 50 ? captured : activeProfile?.trajectoryGenerator(300) || [];
             setRawFrames(frames);
             
             // --- BRANCHING LOGIC FOR BACKEND ---
             if (useBackend) {
                try {
                    setStage(PipelineStage.MOVEMENT_LAB);
                    // Send to Python Backend
                    const response = await fetch(`${SERVER_URL}/analyze_frames`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ frames: frames, config: motionConfig })
                    });
                    
                    if (!response.ok) throw new Error("Backend Error");
                    
                    const backendData = await response.json();
                    
                    if (backendData.metrics) {
                        setChartData(backendData.metrics);
                        setStage(PipelineStage.CLASSIFIER);
                        // Backend also returns 'report' from Gemini
                        const completeReport: AnalysisReport = {
                            classification: backendData.report.classification || "Normal",
                            confidence: 100,
                            seizureDetected: backendData.report.classification === 'Seizures',
                            seizureType: "None",
                            rawData: { ...backendData.biomarkers, posture: {}, seizure: {} }, // simplified for demo
                            clinicalAnalysis: backendData.report.clinicalAnalysis || "Analysis via Python Backend (ViTPose)",
                            recommendations: ["Review backend logs"],
                            timelineData: backendData.metrics
                        };
                        
                        const savedReport = storageService.saveReport(user.id, completeReport, isLive ? `Live (ViTPose)` : file?.name || "Unknown");
                        await new Promise(resolve => setTimeout(resolve, 500));
                        setStage(PipelineStage.COMPLETE); setReport(completeReport); setSelectedReport(savedReport);
                        return;
                    }
                } catch (err) {
                    console.warn("Backend failed, falling back to local JS", err);
                    setError("Server connection failed. Switching to Local Mode.");
                    setUseBackend(false);
                    // Fallthrough to local logic...
                }
             }

             // --- LOCAL JS LOGIC (FALLBACK) ---
             setStage(PipelineStage.MOVEMENT_LAB);
             await new Promise(resolve => setTimeout(resolve, 500));
             const physicsMetrics = physicsEngine.processSignal(frames, motionConfig);
             setChartData(physicsMetrics);
             await new Promise(resolve => setTimeout(resolve, 500));
             setStage(PipelineStage.CLASSIFIER);
             const computedBiomarkers = calculateBiomarkers(physicsMetrics, frames, motionConfig);
             const trainingExamples = storageService.getTrainingExamples();
             const result = await generateGMAReport(computedBiomarkers, trainingExamples);
             const completeReport: AnalysisReport = { ...result, timelineData: physicsMetrics };
             const savedReport = storageService.saveReport(user.id, completeReport, isLive ? `Live Assessment ${new Date().toLocaleTimeString()}` : file?.name || "Unknown Video");
             await new Promise(resolve => setTimeout(resolve, 500));
             setStage(PipelineStage.COMPLETE); setReport(completeReport); setSelectedReport(savedReport); setIsCapturing(false);
             if (appMode === 'training') { setShowTrainingModal(true); }
         };
         requestAnimationFrame(processFrame);
      }
    } catch (e) { console.error("Analysis Error:", e); setError("Analysis failed. Please try again."); setStage(PipelineStage.IDLE); setIsCapturing(false); }
  }, [file, isLive, user, holistic, activeProfile, motionConfig, appMode, useBackend]);

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
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans">
      <header className="fixed top-0 w-full z-50 bg-white/90 backdrop-blur-md border-b border-slate-200 shadow-sm no-print">
         <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
            <div className="flex items-center gap-3 cursor-pointer" onClick={() => setView('dashboard')}>
                <div className="bg-sky-500 w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold shadow-sky-200 shadow-lg">N</div>
                <span className="font-bold text-lg tracking-tight text-slate-800">NeuroMotion AI</span>
            </div>
            <div className="flex items-center gap-4">
               <button 
                  onClick={() => setUseBackend(!useBackend)}
                  className={`text-xs px-3 py-1 rounded-full font-bold border flex items-center transition-all ${useBackend ? 'bg-emerald-100 text-emerald-700 border-emerald-300 shadow-emerald-200 shadow-md' : 'bg-slate-100 text-slate-500 border-slate-200'}`}
               >
                   <i className={`fas ${useBackend ? 'fa-server' : 'fa-laptop'} mr-2`}></i>
                   {useBackend ? 'Server Mode (ViTPose)' : 'Local Mode (Browser)'}
               </button>
               {appMode === 'training' && ( <div className="bg-indigo-600 text-white text-xs px-3 py-1 rounded-full font-bold flex items-center shadow-lg shadow-indigo-200"><i className="fas fa-graduation-cap mr-2"></i> Training Mode Active</div> )}
               <div className="flex items-center gap-2">
                   <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center text-slate-500 text-sm font-bold">{user.name.charAt(0)}</div>
                   <span className="text-sm font-medium hidden md:block">{user.name}</span>
               </div>
               <button onClick={() => { storageService.logout(); window.location.reload(); }} className="text-slate-400 hover:text-slate-600"><i className="fas fa-sign-out-alt"></i></button>
            </div>
         </div>
      </header>
      <main className="pt-24 pb-12 px-4 max-w-7xl mx-auto">
         {view === 'dashboard' && ( <Dashboard user={user} onNewAnalysis={startNewAnalysis} onLiveAnalysis={startLiveAnalysis} onTrainingMode={startTrainingMode} onComparisonMode={startComparisonMode} onViewReport={handleViewReport} onCompareReports={handleCompareReports} /> )}
         {view === 'comparison' && ( <ComparisonView onBack={() => setView('dashboard')} initialReports={reportsToCompare} /> )}
         {view === 'view_report' && selectedReport && ( <ReportView report={selectedReport} onClose={() => setView('dashboard')} onSaveCorrection={handleSaveCorrection} /> )}
         {view === 'pipeline' && (
             <div className="animate-fade-in">
                <div className="mb-6 flex items-center justify-between no-print">
                    <button onClick={() => setView('dashboard')} className="text-slate-500 hover:text-slate-800 flex items-center gap-2 transition-colors"><i className="fas fa-arrow-left"></i> Back to Dashboard</button>
                    {appMode === 'training' && ( <div className="text-sm text-indigo-600 font-bold">Step 1: Upload & Diagnose (Expert) -> Step 2: AI Analysis & Optimization</div> )}
                </div>
                {appMode === 'training' && stage === PipelineStage.INGESTION && (
                    <div className="bg-indigo-50 border border-indigo-100 rounded-xl p-6 mb-8">
                        <h3 className="text-indigo-900 font-bold mb-4 flex items-center"><i className="fas fa-clipboard-check mr-2"></i> Expert Ground Truth Input</h3>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <div>
                                <label className="block text-xs font-bold uppercase text-indigo-800 mb-1">Diagnosis</label>
                                <select className="w-full p-2 rounded border border-indigo-200" onChange={(e) => setSelectedExpertDiagnosis(e.target.value)}>
                                    <option value="">Select Diagnosis...</option>
                                    <option value="Normal">Normal</option>
                                    <option value="Sarnat Stage I">Sarnat Stage I</option>
                                    <option value="Sarnat Stage II">Sarnat Stage II</option>
                                    <option value="Sarnat Stage III">Sarnat Stage III</option>
                                    <option value="Seizures">Seizures</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-xs font-bold uppercase text-indigo-800 mb-1">Notes</label>
                                <input className="w-full p-2 rounded border border-indigo-200" placeholder="E.g. High frequency tremor noted..." onChange={(e) => setExpertAnnotation(e.target.value)} />
                            </div>
                        </div>
                    </div>
                )}
                <div className="no-print"><PipelineVisualizer currentStage={stage} /></div>
                {stage === PipelineStage.COMPLETE && report ? ( <ReportView report={report} onClose={() => setView('dashboard')} onSaveCorrection={handleSaveCorrection} /> ) : (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mt-8">
                        <div className="lg:col-span-2 space-y-4 flex flex-col items-center">
                            <div className="relative bg-black rounded-2xl overflow-hidden shadow-xl group w-auto inline-block">
                                <video ref={videoRef} className="block max-h-[75vh] w-auto max-w-full" src={videoPreview || undefined} controls={stage === PipelineStage.COMPLETE || stage === PipelineStage.INGESTION} playsInline muted />
                                {stage !== PipelineStage.INGESTION && stage !== PipelineStage.IDLE && ( <VideoOverlay videoRef={videoRef} rawFrames={rawFrames} realTimeSkeleton={realTimeSkeleton} isCapturing={isCapturing} isLive={isLive} stage={stage} timeRemaining={timeRemaining} /> )}
                                {stage === PipelineStage.INGESTION && !videoPreview && !isLive && (
                                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white bg-slate-900/50">
                                        <i className="fas fa-cloud-upload-alt text-5xl mb-4 opacity-80"></i>
                                        <label className="bg-sky-500 hover:bg-sky-400 px-6 py-2 rounded-lg cursor-pointer transition-colors shadow-lg">
                                            Select Video File
                                            <input type="file" className="hidden" accept="video/*" onChange={handleFileUpload} />
                                        </label>
                                        <p className="mt-4 text-sm opacity-60">Supported formats: MP4, MOV, WEBM</p>
                                        {useBackend && <div className="mt-2 text-xs text-emerald-300 font-mono">Server Mode Active (ViTPose)</div>}
                                    </div>
                                )}
                            </div>
                            {stage === PipelineStage.INGESTION && (videoPreview || isLive) && (
                                <button onClick={runAnalysis} className="w-full py-4 bg-sky-600 hover:bg-sky-500 text-white rounded-xl font-bold text-lg shadow-xl shadow-sky-900/20 transition-all transform active:scale-95 flex items-center justify-center gap-3">
                                    {isCapturing ? ( <><i className="fas fa-circle-notch fa-spin"></i> Analyzing ({useBackend ? 'Server' : 'Local'})...</> ) : ( <><i className="fas fa-play"></i> Run Clinical Analysis</> )}
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
            <div className="bg-white rounded-2xl shadow-2xl max-w-2xl w-full overflow-hidden">
                <div className="p-6 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                    <h3 className="text-xl font-bold text-slate-800"><i className="fas fa-brain text-purple-600 mr-2"></i> {trainingStatus === 'complete' ? 'Optimization Complete' : 'Expert Training'}</h3>
                    <button onClick={closeTrainingModal} className="text-slate-400 hover:text-slate-600"><i className="fas fa-times"></i></button>
                </div>
                <div className="p-6">
                    {trainingStatus === 'analyzing' ? (
                        <div className="text-center py-12"><i className="fas fa-microchip text-4xl text-sky-500 animate-bounce mb-4"></i><h4 className="text-lg font-bold text-slate-700">Refining Physics Engine...</h4><p className="text-slate-500">The AI is adjusting signal thresholds based on your feedback.</p></div>
                    ) : trainingStatus === 'complete' && prevConfig && motionConfig ? (
                        <div className="space-y-6">
                            <div className="bg-green-50 text-green-800 p-4 rounded-lg flex items-start gap-3"><i className="fas fa-check-circle mt-1 text-green-600"></i><div><h4 className="font-bold">Algorithm Updated</h4><p className="text-sm">New parameters have been saved to memory.</p></div></div>
                            <div className="bg-slate-900 rounded-xl p-4 text-xs font-mono text-slate-300">
                                <div className="grid grid-cols-3 border-b border-slate-700 pb-2 mb-2 font-bold text-slate-500"><div>PARAMETER</div><div>WAS</div><div>NOW</div></div>
                                {Object.entries(motionConfig).map(([key, val]) => { const old = (prevConfig as any)[key]; if (typeof val !== 'number') return null; const diff = val - old; const changed = Math.abs(diff) > 0.001; return ( <div key={key} className={`grid grid-cols-3 py-1 ${changed ? 'text-white bg-white/5' : 'opacity-50'}`}><div>{key}</div><div>{Number(old).toFixed(2)}</div><div className={changed ? (diff > 0 ? 'text-green-400' : 'text-red-400') : ''}>{Number(val).toFixed(2)} {changed && ` (${diff > 0 ? '+' : ''}${diff.toFixed(2)})`}</div></div> ); })}
                            </div>
                            <button onClick={closeTrainingModal} className="w-full bg-slate-800 hover:bg-slate-700 text-white py-3 rounded-lg font-bold">Close & Save Results</button>
                        </div>
                    ) : ( <div><p>Initializing...</p></div> )}
                </div>
            </div>
        </div>
      )}
      {error && ( <div className="fixed bottom-6 left-1/2 transform -translate-x-1/2 bg-red-600 text-white px-6 py-3 rounded-full shadow-2xl flex items-center gap-3 z-[200] animate-bounce"><i className="fas fa-exclamation-circle"></i>{error}<button onClick={() => setError(null)} className="ml-2 opacity-80 hover:opacity-100"><i className="fas fa-times"></i></button></div> )}
    </div>
  );
};

export default App;