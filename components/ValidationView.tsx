import React, { useState, useEffect, useRef } from 'react';
import { SERVER_URL } from '../constants';

const SKELETON_CONNECTIONS = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"], ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"], ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"], ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"], ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"], ["right_knee", "right_ankle"],
    ["nose", "left_eye"], ["nose", "right_eye"],
    ["left_eye", "left_ear"], ["right_eye", "right_ear"]
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

const drawSkeleton = (
    canvas: HTMLCanvasElement,
    skeleton: ValidationCase['first_frame_skeleton']
) => {
    if (!skeleton || !skeleton.joints) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const joints = skeleton.joints;
    const w = canvas.width;
    const h = canvas.height;

    const toCanvas = (x: number, y: number) => ({
        x: (x / 100) * w,
        y: (y / 100) * h
    });

    // Draw connections
    ctx.strokeStyle = 'rgba(163, 163, 163, 0.6)';
    ctx.lineWidth = 2;
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

    // Draw joints
    Object.entries(joints).forEach(([jointName, joint]) => {
        if ((joint.visibility || 1) < 0.5) return;

        const pos = toCanvas(joint.x, joint.y);

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#171717';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.font = '9px Inter, sans-serif';
        ctx.fillStyle = '#a3a3a3';
        ctx.fillText(jointName, pos.x + 8, pos.y + 3);
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

    useEffect(() => {
        if (selectedCase && canvasRef.current) {
            drawSkeleton(canvasRef.current, selectedCase.first_frame_skeleton);
        }
    }, [selectedCase]);

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

            setGroundTruth('');
            setDoctorNotes('');
            setSelectedCase(null);
            await fetchCases();
            alert('Validation saved successfully.');
        } catch (err) {
            console.error('Submit error:', err);
            alert('Failed to submit validation. Please try again.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div className="animate-fade-in">
            {/* Header */}
            <div className="mb-6 flex items-center justify-between">
                <div>
                    <button
                        onClick={onClose}
                        className="text-neutral-400 hover:text-neutral-900 flex items-center gap-2 transition-colors mb-2 text-sm"
                    >
                        <i className="fas fa-arrow-left"></i> Back
                    </button>
                    <h1 className="text-xl font-semibold text-neutral-900">Validation Review</h1>
                    <p className="text-neutral-500 text-sm mt-0.5">
                        Review AI predictions and provide ground truth labels
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <button
                        onClick={() => setShowValidated(!showValidated)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                            showValidated
                                ? 'bg-neutral-200 text-neutral-700'
                                : 'bg-white text-neutral-600 border border-neutral-200'
                        }`}
                    >
                        {showValidated ? 'Unvalidated Only' : 'Show All'}
                    </button>
                    <div className="bg-neutral-900 text-white px-3 py-1.5 rounded-lg font-mono text-xs">
                        {cases.length} {showValidated ? 'total' : 'pending'}
                    </div>
                </div>
            </div>

            {loading ? (
                <div className="text-center py-20">
                    <i className="fas fa-circle-notch fa-spin text-xl text-neutral-300 mb-3"></i>
                    <p className="text-neutral-500 text-sm">Loading cases...</p>
                </div>
            ) : cases.length === 0 ? (
                <div className="text-center py-20 bg-white rounded-lg border border-neutral-200">
                    <p className="text-neutral-900 font-medium mb-1">All caught up</p>
                    <p className="text-neutral-500 text-sm">No pending validations at the moment.</p>
                </div>
            ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Case List */}
                    <div className="space-y-2 max-h-[80vh] overflow-y-auto pr-1">
                        {cases.map((c, idx) => (
                            <div
                                key={idx}
                                onClick={() => setSelectedCase(c)}
                                className={`bg-white rounded-lg p-4 cursor-pointer transition-all border ${
                                    selectedCase === c
                                        ? 'border-neutral-900 shadow-sm'
                                        : 'border-neutral-200 hover:border-neutral-300'
                                }`}
                            >
                                <div className="flex items-start justify-between mb-2">
                                    <span className="text-[11px] font-mono text-neutral-400">
                                        {new Date(c.timestamp).toLocaleString()}
                                    </span>
                                    {c.ground_truth && (
                                        <span className="bg-neutral-100 text-neutral-600 text-[10px] px-2 py-0.5 rounded font-medium">
                                            Validated
                                        </span>
                                    )}
                                </div>
                                <div className="flex items-center gap-2 mb-3">
                                    <span className="px-2.5 py-0.5 rounded text-xs font-medium bg-neutral-100 text-neutral-700">
                                        {c.gemini_prediction.classification}
                                    </span>
                                    <span className="text-[11px] text-neutral-400 font-mono">
                                        {(c.gemini_prediction.confidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <div className="grid grid-cols-3 gap-2 text-xs">
                                    <div className="bg-neutral-50 rounded px-2 py-1.5">
                                        <div className="text-neutral-400 text-[10px]">Entropy</div>
                                        <div className="font-mono font-medium text-neutral-700">{c.biomarkers.average_sample_entropy.toFixed(2)}</div>
                                    </div>
                                    <div className="bg-neutral-50 rounded px-2 py-1.5">
                                        <div className="text-neutral-400 text-[10px]">Jerk</div>
                                        <div className="font-mono font-medium text-neutral-700">{c.biomarkers.average_jerk.toFixed(2)}</div>
                                    </div>
                                    <div className="bg-neutral-50 rounded px-2 py-1.5">
                                        <div className="text-neutral-400 text-[10px]">Frames</div>
                                        <div className="font-mono font-medium text-neutral-700">{c.biomarkers.frames_processed || c.metadata?.frame_count || 'N/A'}</div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Detail View */}
                    {selectedCase ? (
                        <div className="bg-white rounded-lg border border-neutral-200 p-5 sticky top-6 max-h-[85vh] overflow-y-auto">
                            <h2 className="text-sm font-semibold text-neutral-900 mb-4">Case Details</h2>

                            {/* Skeleton */}
                            {selectedCase.first_frame_skeleton ? (
                                <div className="mb-5">
                                    <p className="text-xs text-neutral-500 mb-2">Skeleton (YOLO26 Pose)</p>
                                    <div className="bg-neutral-950 rounded-lg p-3">
                                        <canvas
                                            ref={canvasRef}
                                            width={400}
                                            height={400}
                                            className="w-full rounded"
                                        />
                                        {selectedCase.first_frame_skeleton.note && (
                                            <p className="text-[10px] text-neutral-500 mt-2 text-center">
                                                {selectedCase.first_frame_skeleton.note}
                                            </p>
                                        )}
                                    </div>
                                </div>
                            ) : (
                                <div className="mb-5 bg-neutral-50 rounded-lg p-6 text-center text-neutral-400 text-sm border border-neutral-100">
                                    No skeleton data available
                                </div>
                            )}

                            {/* AI Prediction */}
                            <div className="mb-5">
                                <p className="text-xs text-neutral-500 mb-2">AI Prediction</p>
                                <div className="bg-neutral-50 rounded-lg p-4 space-y-2 border border-neutral-100">
                                    <div className="flex items-center justify-between">
                                        <span className="text-neutral-500 text-xs">Classification</span>
                                        <span className="px-2.5 py-0.5 rounded text-xs font-medium bg-neutral-100 text-neutral-700">
                                            {selectedCase.gemini_prediction.classification}
                                        </span>
                                    </div>
                                    <div className="flex items-center justify-between">
                                        <span className="text-neutral-500 text-xs">Confidence</span>
                                        <span className="font-mono text-xs font-medium text-neutral-900">{(selectedCase.gemini_prediction.confidence * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="pt-2 border-t border-neutral-200">
                                        <p className="text-xs text-neutral-600 leading-relaxed">{selectedCase.gemini_prediction.reasoning}</p>
                                    </div>
                                    {selectedCase.gemini_prediction.recommendations && (
                                        <div className="pt-2 border-t border-neutral-200">
                                            <p className="text-[10px] text-neutral-400 uppercase font-medium mb-1">Recommendations</p>
                                            <p className="text-xs text-neutral-600">{selectedCase.gemini_prediction.recommendations}</p>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* Biomarkers */}
                            <div className="mb-5">
                                <p className="text-xs text-neutral-500 mb-2">Biomarkers</p>
                                <div className="grid grid-cols-2 gap-2">
                                    <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                                        <div className="text-[10px] text-neutral-400 font-medium mb-0.5">Avg Entropy</div>
                                        <div className="text-base font-mono font-semibold text-neutral-900">
                                            {selectedCase.biomarkers.average_sample_entropy.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                                        <div className="text-[10px] text-neutral-400 font-medium mb-0.5">Peak Entropy</div>
                                        <div className="text-base font-mono font-semibold text-neutral-900">
                                            {selectedCase.biomarkers.peak_sample_entropy.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                                        <div className="text-[10px] text-neutral-400 font-medium mb-0.5">Avg Jerk</div>
                                        <div className="text-base font-mono font-semibold text-neutral-900">
                                            {selectedCase.biomarkers.average_jerk.toFixed(3)}
                                        </div>
                                    </div>
                                    <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                                        <div className="text-[10px] text-neutral-400 font-medium mb-0.5">Frames</div>
                                        <div className="text-base font-mono font-semibold text-neutral-900">
                                            {selectedCase.biomarkers.frames_processed || selectedCase.metadata?.frame_count || 'N/A'}
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Validation Form */}
                            {!selectedCase.ground_truth ? (
                                <div className="border-t border-neutral-200 pt-5">
                                    <p className="text-xs text-neutral-500 mb-3">Doctor Validation</p>
                                    <div className="space-y-3">
                                        <div>
                                            <label className="block text-[11px] font-medium text-neutral-700 mb-1">
                                                Ground Truth Classification
                                            </label>
                                            <select
                                                value={groundTruth}
                                                onChange={(e) => setGroundTruth(e.target.value)}
                                                className="w-full p-2.5 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-neutral-900 focus:border-transparent text-sm"
                                            >
                                                <option value="">Select...</option>
                                                <option value="Normal">Normal</option>
                                                <option value="Sarnat Stage I">Sarnat Stage I</option>
                                                <option value="Sarnat Stage II">Sarnat Stage II</option>
                                                <option value="Sarnat Stage III">Sarnat Stage III</option>
                                                <option value="Seizures">Seizures</option>
                                                <option value="Uncertain">Uncertain</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-[11px] font-medium text-neutral-700 mb-1">
                                                Notes (Optional)
                                            </label>
                                            <textarea
                                                value={doctorNotes}
                                                onChange={(e) => setDoctorNotes(e.target.value)}
                                                placeholder="Clinical observations..."
                                                rows={3}
                                                className="w-full p-2.5 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-neutral-900 focus:border-transparent text-sm"
                                            />
                                        </div>
                                        <button
                                            onClick={handleSubmit}
                                            disabled={submitting || !groundTruth}
                                            className="w-full bg-neutral-900 hover:bg-neutral-800 disabled:bg-neutral-200 disabled:text-neutral-400 disabled:cursor-not-allowed text-white py-2.5 rounded-lg font-medium transition-colors text-sm"
                                        >
                                            {submitting ? 'Submitting...' : 'Submit Validation'}
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <div className="border-t border-neutral-200 pt-5">
                                    <div className="bg-neutral-50 border border-neutral-200 rounded-lg p-4">
                                        <p className="text-xs font-medium text-neutral-900 mb-2">Already Validated</p>
                                        <div className="space-y-1.5 text-sm">
                                            <div>
                                                <span className="text-neutral-500 text-xs">Ground Truth: </span>
                                                <span className="text-neutral-900 font-medium text-xs">{selectedCase.ground_truth}</span>
                                            </div>
                                            {selectedCase.doctor_notes && (
                                                <div>
                                                    <span className="text-neutral-500 text-xs">Notes: </span>
                                                    <p className="text-neutral-700 text-xs mt-0.5">{selectedCase.doctor_notes}</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="bg-neutral-50 rounded-lg p-12 text-center text-neutral-400 border border-neutral-200">
                            <p className="text-sm">Select a case to review</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
