
import { PipelineStage, StageConfig, SkeletonFrame } from './types';

export const STAGES: StageConfig[] = [
  { id: PipelineStage.INGESTION, label: "Ingestion", icon: "fa-video", description: "Video upload and preprocessing" },
  { id: PipelineStage.LIFTING_3D, label: "YOLO26 Pose", icon: "fa-cube", description: "YOLO26x Pose Estimation (Single-Stage)" },
  { id: PipelineStage.MOVEMENT_LAB, label: "Movement Lab", icon: "fa-flask", description: "Entropy, Fluency & Complexity analysis" },
  { id: PipelineStage.CLASSIFIER, label: "Diagnose", icon: "fa-user-md", description: "Hybrid Transformer-GCN Classification" },
  { id: PipelineStage.COMPLETE, label: "Results", icon: "fa-clipboard-check", description: "Clinical assessment report" }
];

export const SERVER_URL = "";

// --- API helpers for trajectory generation (logic moved to Python backend) ---

export interface ProfileInfo {
  id: string;
  label: string;
  features: {
    entropyScore: string;
    fluencySAL: string;
    fractalDimension: string;
    convexHullVolume: string;
    phaseSpaceTopology: string;
    clinicalNote: string;
  };
}

export const getProfileBySeed = async (seed: number): Promise<ProfileInfo> => {
  const res = await fetch(`${SERVER_URL}/profile_by_seed/${seed}`);
  if (!res.ok) throw new Error('Failed to get profile');
  return await res.json();
};

export const getRandomProfile = async (): Promise<ProfileInfo> => {
  const seed = Math.floor(Math.random() * 233280);
  return getProfileBySeed(seed);
};

export const generateTrajectory = async (profileId: string, frames: number = 300, seed?: number): Promise<SkeletonFrame[]> => {
  const res = await fetch(`${SERVER_URL}/generate_trajectory`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ profile_id: profileId, frames, seed })
  });
  if (!res.ok) throw new Error('Failed to generate trajectory');
  const data = await res.json();
  return data.frames;
};

export const getClinicalProfiles = async (): Promise<ProfileInfo[]> => {
  const res = await fetch(`${SERVER_URL}/clinical_profiles`);
  if (!res.ok) throw new Error('Failed to get clinical profiles');
  return await res.json();
};
