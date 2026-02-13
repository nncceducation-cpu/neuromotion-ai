
import { User, SavedReport, AnalysisReport, ExpertCorrection } from '../types';
import { SERVER_URL } from '../constants';

// Session token stored in memory (persisted to localStorage for page refreshes)
let _token: string | null = localStorage.getItem('neuromotion_token');

const setToken = (token: string | null) => {
  _token = token;
  if (token) localStorage.setItem('neuromotion_token', token);
  else localStorage.removeItem('neuromotion_token');
};

const authHeaders = () => ({
  'Content-Type': 'application/json',
  ..._token ? { 'Authorization': `Bearer ${_token}` } : {}
});

export const storageService = {
  // --- Auth ---
  register: async (name: string, email: string, password: string): Promise<User> => {
    const res = await fetch(`${SERVER_URL}/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Registration failed' }));
      throw new Error(err.detail || 'Registration failed');
    }
    const data = await res.json();
    setToken(data.token);
    return data.user;
  },

  login: async (email: string, password: string): Promise<User> => {
    const res = await fetch(`${SERVER_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Invalid credentials' }));
      throw new Error(err.detail || 'Invalid credentials');
    }
    const data = await res.json();
    setToken(data.token);
    localStorage.setItem('neuromotion_session', JSON.stringify(data.user));
    return data.user;
  },

  logout: () => {
    if (_token) {
      fetch(`${SERVER_URL}/auth/logout`, {
        method: 'POST',
        headers: authHeaders()
      }).catch(() => {});
    }
    setToken(null);
    localStorage.removeItem('neuromotion_session');
  },

  getCurrentUser: (): User | null => {
    const stored = localStorage.getItem('neuromotion_session');
    return stored ? JSON.parse(stored) : null;
  },

  // --- Data ---
  saveReport: async (userId: string, report: AnalysisReport, videoName: string): Promise<SavedReport> => {
    const res = await fetch(`${SERVER_URL}/reports/${userId}`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ report, videoName })
    });
    if (!res.ok) throw new Error('Failed to save report');
    return await res.json();
  },

  getReports: async (userId: string): Promise<SavedReport[]> => {
    const res = await fetch(`${SERVER_URL}/reports/${userId}`, {
      headers: authHeaders()
    });
    if (!res.ok) return [];
    return await res.json();
  },

  deleteReport: async (userId: string, reportId: string): Promise<void> => {
    await fetch(`${SERVER_URL}/reports/${userId}/${reportId}`, {
      method: 'DELETE',
      headers: authHeaders()
    });
  },

  // --- Expert Feedback Loop ---
  saveExpertCorrection: async (userId: string, reportId: string, correction: ExpertCorrection) => {
    const res = await fetch(`${SERVER_URL}/reports/${userId}/${reportId}/correction`, {
      method: 'POST',
      headers: authHeaders(),
      body: JSON.stringify({ correction })
    });
    if (!res.ok) return null;
    return await res.json();
  },

  getTrainingExamples: async (): Promise<{ inputs: any, groundTruth: ExpertCorrection }[]> => {
    const res = await fetch(`${SERVER_URL}/training_examples`, {
      headers: authHeaders()
    });
    if (!res.ok) return [];
    return await res.json();
  },

  // --- Analytics for Dashboard ---
  getLearnedStats: async (userId: string) => {
    const res = await fetch(`${SERVER_URL}/learned_stats/${userId}`, {
      headers: authHeaders()
    });
    if (!res.ok) return { totalLearned: 0, breakdown: {} };
    return await res.json();
  }
};
