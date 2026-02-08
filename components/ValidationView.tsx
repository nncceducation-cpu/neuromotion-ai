import React, { useState, useEffect, useRef } from 'react';
import { SERVER_URL } from '../constants';

/**
 * COCO 17 keypoint names in standard order.
 * These correspond to the joints extracted by YOLO26 Pose.
 *
 * **What this is:** Every human pose detection model outputs a fixed set of "keypoints"
 * representing major body joints. COCO (Common Objects in Context) defines 17 standard
 * keypoints that most models use.
 */
const COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
];

/**
 * Skeleton connections for drawing limbs.
 * Each pair represents a connection between two keypoints (e.g., [shoulder, elbow])
 *
 * **Why we need this:** To draw a "stick figure" skeleton, we need to know which
 * joints connect to which. This array defines the human skeletal structure.
 */
const SKELETON_CONNECTIONS = [
    ["left_shoulder", "right_shoulder"],  // Shoulder line
    ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],  // Left arm
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],  // Right arm
    ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],  // Torso
    ["left_hip", "right_hip"],  // Hip line
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"],  // Left leg
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"],  // Right leg
    ["nose", "left_eye"], ["nose", "right_eye"],  // Face
    ["left_eye", "left_ear"], ["right_eye", "right_ear"]  // Ears
];

interface ValidationCase {
    timestamp: string;
    biomarkers: {
        average_sample_entropy: number;
        peak_sample_entropy: number;
        average_jerk: number;
        backend_source?: string;
        frames_processed?: number;
    };
    gemini_prediction: {
        classification: string;
        confidence: number;
        reasoning: string;
        recommendations?: string;
    };
    first_frame_skeleton?: {
        joints: Record<string, { x: number; y: number; z: number; visibility?: number }>;
        keypoint_labels?: string[];
        note?: string;
    };
    ground_truth?: string;
    doctor_notes?: string;
    metadata?: {
        source?: string;
        frame_count?: number;
        filename?: string;
    };
}

/**
 * Draws a skeleton on a canvas element with labeled keypoints.
 *
 * **How it works:**
 * 1. Draws lines connecting joints (the "limbs")
 * 2. Draws circles at each joint
 * 3. Adds text labels next to each joint (e.g., "left_wrist")
 *
 * **Why canvas instead of SVG:** Canvas is faster for drawing operations and better
 * for pixel-perfect rendering of complex visualizations.
 */
const drawSkeleton = (
    canvas: HTMLCanvasElement,
    skeleton: ValidationCase['first_frame_skeleton']
) => {
    if (!skeleton || !skeleton.joints) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const joints = skeleton.joints;
    const w = canvas.width;
    const h = canvas.height;

    /**
     * Helper function to convert normalized coordinates (0-100) to canvas pixels.
     * YOLO outputs coordinates normalized to 0-100 range, we need to scale to canvas size.
     */
    const toCanvas = (x: number, y: number) => ({
        x: (x / 100) * w,
        y: (y / 100) * h
    });

    // Draw connections (limbs) first, so they appear behind the joints
    ctx.strokeStyle = 'rgba(96, 165, 250, 0.8)';  // Blue
    ctx.lineWidth = 3;
    SKELETON_CONNECTIONS.forEach(([joint1Name, joint2Name]) => {
        const j1 = joints[joint1Name];
        const j2 = joints[joint2Name];
        if (j1 && j2 && (j1.visibility || 1) > 0.5 && (j2.visibility || 1) > 0.5) {
            const p1 = toCanvas(j1.x, j1.y);
            const p2 = toCanvas(j2.x, j2.y);
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
        }
    });

    // Draw joints (circles) and labels
    Object.entries(joints).forEach(([jointName, joint]) => {
        if ((joint.visibility || 1) < 0.5) return;  // Skip low-confidence keypoints

        const pos = toCanvas(joint.x, joint.y);

        // Draw circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 6, 0, 2 * Math.PI);

        // Color code: Red for upper body, orange/cyan for limbs
        if (jointName.includes('eye') || jointName.includes('ear') || jointName === 'nose') {
            ctx.fillStyle = '#ef4444';  // Red for face
        } else if (jointName.includes('shoulder') || jointName.includes('hip')) {
            ctx.fillStyle = '#f97316';  // Orange for torso
        } else {
            ctx.fillStyle = '#06b6d4';  // Cyan for limbs
        }
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw label next to joint
        ctx.font = '10px monospace';
        ctx.fillStyle = 'white';
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 3;
        ctx.strokeText(jointName, pos.x + 10, pos.y + 4);
        ctx.fillText(jointName, pos.x + 10, pos.y + 4);
    });
};

export const ValidationView: React.FC<{ onClose: () => void }> = ({ onClose }) => {
    const [cases, setCases] = useState<ValidationCase[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCase, setSelectedCase] = useState<ValidationCase | null>(null);
    const [groundTruth, setGroundTruth] = useState<string>('');
    const [doctorNotes, setDoctorNotes] = useState<string>('');
    const [submitting, setSubmitting] = useState(false);
    const [showValidated, setShowValidated] = useState(false);

    const canvasRef = useRef<HTMLCanvasElement>(null);

    /**
     * Fetch pending validations from the backend when component mounts.
     *
     * **Why useEffect with empty dependency array []?**
     * This runs once when the component first renders (like componentDidMount in class components)
     * We fetch data once and store it in state.
     */
    useEffect(() => {
        fetchCases();
    }, [showValidated]);

    const fetchCases = async () => {
        setLoading(true);
        try {
            const response = await fetch(
                `${SERVER_URL}/pending_validations?limit=50&include_validated=${showValidated}`
            );
            const data = await response.json();
            setCases(data.cases || []);
        } catch (err) {
            console.error('Failed to fetch validations:', err);
        } finally {
            setLoading(false);
        }
    };

    /**
     * Redraw skeleton whenever selected case changes.
     *
     * **Why a separate useEffect?** Drawing on canvas is a "side effect" - it happens
     * after the component renders. We re-draw whenever the selected case changes.
     */
    useEffect(() => {
        if (selectedCase && canvasRef.current) {
            drawSkeleton(canvasRef.current, selectedCase.first_frame_skeleton);
        }
    }, [selectedCase]);

    /**
     * Submit validation to backend.
     *
     * **What happens:**
     * 1. POST to /validate endpoint with timestamp + ground truth + notes
     * 2. Backend finds the matching log entry and updates it
     * 3. We refresh the list to remove the validated case from pending
     */
    const handleSubmit = async () => {
        if (!selectedCase || !groundTruth) {
            alert('Please select a classification');
            return;
        }

        setSubmitting(true);
        try {
            const response = await fetch(`${SERVER_URL}/validate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    timestamp: selectedCase.timestamp,
                    ground_truth_classification: groundTruth,
                    doctor_notes: doctorNotes || undefined
                })
            });

            if (!response.ok) throw new Error('Validation failed');

            // Success! Clear form and refresh list
            setGroundTruth('');
            setDoctorNotes('');
            setSelectedCase(null);
            await fetchCases();
            alert('✓ Validation saved successfully!');
        } catch (err) {
            console.error('Submit error:', err);
            alert('Failed to submit validation. Please try again.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="min-h-screen bg-slate-50 p-6">
            {/* Header */}
            <div className="mb-6 flex items-center justify-between">
                <div>
                    <button
                        onClick={onClose}
                        className="text-slate-500 hover:text-slate-800 flex items-center gap-2 transition-colors mb-2"
                    >
                        <i className="fas fa-arrow-left"></i> Back to Dashboard
                    </button>
                    <h1 className="text-3xl font-bold text-slate-800 flex items-center gap-3">
                        <i className="fas fa-clipboard-check text-slate-600"></i>
                        Doctor Validation Review
                    </h1>
                    <p className="text-slate-600 mt-1">
                        Review AI predictions and provide ground truth labels for model training
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => setShowValidated(!showValidated)}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                            showValidated
                                ? 'bg-slate-200 text-slate-700'
                                : 'bg-white text-slate-600 border border-slate-200'
                        }`}
                    >
                        {showValidated ? 'Show Unvalidated Only' : 'Show All Cases'}
                    </button>
                    <div className="bg-slate-800 text-white px-4 py-2 rounded-lg font-mono">
                        {cases.length} {showValidated ? 'total' : 'pending'}
                    </div>
                </div>
            </div>

            {loading ? (
                <div className="text-center py-20">
                    <i className="fas fa-spinner fa-spin text-4xl text-slate-400 mb-4"></i>
                    <p className="text-slate-600">Loading cases...</p>
                </div>
            ) : cases.length === 0 ? (
                <div className="text-center py-20 bg-white rounded-xl border border-slate-200">
                    <i className="fas fa-check-circle text-6xl text-green-500 mb-4"></i>
                    <h3 className="text-xl font-bold text-slate-700 mb-2">All caught up!</h3>
                    <p className="text-slate-600">No pending validations at the moment.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Left: Case List */}
                    <div className="space-y-3 max-h-[80vh] overflow-y-auto">
                        {cases.map((c, idx) => (
                            <div
                                key={idx}
                                onClick={() => setSelectedCase(c)}
                                className={`bg-white rounded-xl p-4 cursor-pointer transition-all border-2 ${
                                    selectedCase === c
                                        ? 'border-slate-800 shadow-lg'
                                        : 'border-slate-200 hover:border-slate-300'
                                }`}
                            >
                                <div className="flex items-start justify-between mb-2">
                                    <div className="flex items-center gap-2">
                                        <i className="fas fa-user-injured text-slate-400"></i>
                                        <span className="text-xs font-mono text-slate-500">
                                            {new Date(c.timestamp).toLocaleString()}
                                        </span>
                                    </div>
                                    {c.ground_truth && (
                                        <span className="bg-green-100 text-green-700 text-xs px-2 py-1 rounded-full font-bold">
                                            ✓ Validated
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-3 mb-3">
                                    <div className={`px-3 py-1 rounded-lg text-sm font-bold ${
                                        c.gemini_prediction.classification === 'Normal'
                                            ? 'bg-green-100 text-green-700'
                                            : c.gemini_prediction.classification.includes('Sarnat')
                                            ? 'bg-orange-100 text-orange-700'
                                            : 'bg-red-100 text-red-700'
                                    }`}>
                                        {c.gemini_prediction.classification}
                                    </div>
                                    <div className="text-xs text-slate-500">
                                        Confidence: {(c.gemini_prediction.confidence * 100).toFixed(0)}%
                                    </div>
                                </div>
                                <div className="grid grid-cols-3 gap-2 text-xs">
                                    <div className="bg-slate-50 rounded px-2 py-1">
                                        <div className="text-slate-500 text-[10px]">Entropy</div>
                                        <div className="font-mono font-bold">{c.biomarkers.average_sample_entropy.toFixed(2)}</div>
                                    </div>
                                    <div className="bg-slate-50 rounded px-2 py-1">
                                        <div className="text-slate-500 text-[10px]">Jerk</div>
                                        <div className="font-mono font-bold">{c.biomarkers.average_jerk.toFixed(2)}</div>
                                    </div>
                                    <div className="bg-slate-50 rounded px-2 py-1">
                                        <div className="text-slate-500 text-[10px]">Frames</div>
                                        <div className="font-mono font-bold">{c.biomarkers.frames_processed || c.metadata?.frame_count || 'N/A'}</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Right: Detail View + Validation Form */}
                    {selectedCase ? (
                        <div className="bg-white rounded-xl shadow-lg p-6 sticky top-6 max-h-[85vh] overflow-y-auto">
                            <h2 className="text-xl font-bold text-slate-800 mb-4 flex items-center gap-2">
                                <i className="fas fa-file-medical"></i>
                                Case Details
                            </h2>

                            {/* Skeleton Visualization */}
                            {selectedCase.first_frame_skeleton ? (
                                <div className="mb-6">
                                    <h3 className="text-sm font-bold text-slate-700 mb-2 flex items-center gap-2">
                                        <i className="fas fa-user"></i>
                                        First Frame Skeleton (YOLO26 Pose)
                                    </h3>
                                    <div className="bg-slate-900 rounded-lg p-4">
                                        <canvas
                                            ref={canvasRef}
                                            width={400}
                                            height={400}
                                            className="w-full rounded border border-slate-700"
                                        />
                                        <p className="text-xs text-slate-400 mt-2 text-center">
                                            {selectedCase.first_frame_skeleton.note}
                                        </p>
                                    </div>
                                </div>
                            ) : (
                                <div className="mb-6 bg-slate-100 rounded-lg p-4 text-center text-slate-500 text-sm">
                                    <i className="fas fa-image text-2xl mb-2"></i>
                                    <p>No skeleton data available for this case</p>
                                </div>
                            )}

                            {/* AI Prediction */}
                            <div className="mb-6">
                                <h3 className="text-sm font-bold text-slate-700 mb-2">AI Prediction (Gemini)</h3>
                                <div className="bg-slate-50 rounded-lg p-4 space-y-2">
                                    <div className="flex items-center justify-between">
                                        <span className="text-slate-600 text-sm">Classification:</span>
                                        <span className={`px-3 py-1 rounded text-sm font-bold ${
                                            selectedCase.gemini_prediction.classification === 'Normal'
                                                ? 'bg-green-100 text-green-700'
                                                : 'bg-orange-100 text-orange-700'
                                        }`}>
                                            {selectedCase.gemini_prediction.classification}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-slate-600 text-sm">Confidence:</span>
                                        <span className="font-mono font-bold">{(selectedCase.gemini_prediction.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="pt-2 border-t border-slate-200">
                                        <p className="text-xs text-slate-600 italic">"{selectedCase.gemini_prediction.reasoning}"</p>
                                    </div>
                                    {selectedCase.gemini_prediction.recommendations && (
                                        <div className="pt-2 border-t border-slate-200">
                                            <span className="text-slate-600 text-xs font-bold">Recommendations:</span>
                                            <p className="text-xs text-slate-600 mt-1">{selectedCase.gemini_prediction.recommendations}</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Biomarkers */}
                            <div className="mb-6">
                                <h3 className="text-sm font-bold text-slate-700 mb-2">Biomarkers</h3>
                                <div className="grid grid-cols-2 gap-3">
                                    <div className="bg-blue-50 rounded-lg p-3">
                                        <div className="text-xs text-blue-600 font-bold mb-1">Avg Entropy</div>
                                        <div className="text-xl font-mono font-bold text-blue-900">
                                            {selectedCase.biomarkers.average_sample_entropy.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="bg-purple-50 rounded-lg p-3">
                                        <div className="text-xs text-purple-600 font-bold mb-1">Peak Entropy</div>
                                        <div className="text-xl font-mono font-bold text-purple-900">
                                            {selectedCase.biomarkers.peak_sample_entropy.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="bg-orange-50 rounded-lg p-3">
                                        <div className="text-xs text-orange-600 font-bold mb-1">Avg Jerk</div>
                                        <div className="text-xl font-mono font-bold text-orange-900">
                                            {selectedCase.biomarkers.average_jerk.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="bg-green-50 rounded-lg p-3">
                                        <div className="text-xs text-green-600 font-bold mb-1">Frames</div>
                                        <div className="text-xl font-mono font-bold text-green-900">
                                            {selectedCase.biomarkers.frames_processed || selectedCase.metadata?.frame_count || 'N/A'}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Validation Form */}
                            {!selectedCase.ground_truth ? (
                                <div className="border-t border-slate-200 pt-6">
                                    <h3 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <i className="fas fa-user-md"></i>
                                        Doctor Validation
                                    </h3>
                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-xs font-bold text-slate-700 mb-2">
                                                Ground Truth Classification *
                                            </label>
                                            <select
                                                value={groundTruth}
                                                onChange={(e) => setGroundTruth(e.target.value)}
                                                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-slate-800 focus:border-transparent"
                                            >
                                                <option value="">-- Select Diagnosis --</option>
                                                <option value="Normal">Normal</option>
                                                <option value="Sarnat Stage I">Sarnat Stage I</option>
                                                <option value="Sarnat Stage II">Sarnat Stage II</option>
                                                <option value="Sarnat Stage III">Sarnat Stage III</option>
                                                <option value="Seizures">Seizures</option>
                                                <option value="Uncertain">Uncertain (Need More Data)</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-xs font-bold text-slate-700 mb-2">
                                                Clinical Notes (Optional)
                                            </label>
                                            <textarea
                                                value={doctorNotes}
                                                onChange={(e) => setDoctorNotes(e.target.value)}
                                                placeholder="E.g., EEG confirmed moderate HIE, started therapeutic hypothermia..."
                                                rows={4}
                                                className="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-slate-800 focus:border-transparent text-sm"
                                            />
                                        </div>
                                        <button
                                            onClick={handleSubmit}
                                            disabled={submitting || !groundTruth}
                                            className="w-full bg-slate-800 hover:bg-slate-700 disabled:bg-slate-300 disabled:cursor-not-allowed text-white py-3 rounded-lg font-bold transition-colors flex items-center justify-center gap-2"
                                        >
                                            {submitting ? (
                                                <>
                                                    <i className="fas fa-spinner fa-spin"></i>
                                                    Submitting...
                                                </>
                                            ) : (
                                                <>
                                                    <i className="fas fa-check"></i>
                                                    Submit Validation
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="border-t border-slate-200 pt-6">
                                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                                        <div className="flex items-center gap-2 mb-2">
                                            <i className="fas fa-check-circle text-green-600"></i>
                                            <span className="text-green-800 font-bold">Already Validated</span>
                                        </div>
                                        <div className="space-y-2 text-sm">
                                            <div>
                                                <span className="text-green-700 font-bold">Ground Truth: </span>
                                                <span className="text-green-900">{selectedCase.ground_truth}</span>
                                            </div>
                                            {selectedCase.doctor_notes && (
                                                <div>
                                                    <span className="text-green-700 font-bold">Notes: </span>
                                                    <p className="text-green-900 italic mt-1">"{selectedCase.doctor_notes}"</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="bg-slate-100 rounded-xl p-12 text-center text-slate-500">
                            <i className="fas fa-hand-pointer text-5xl mb-4"></i>
                            <p className="text-lg font-medium">Select a case from the list to review</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
