
import { User, SavedReport, AnalysisReport, ExpertCorrection } from '../types';

const STORAGE_KEYS = {
  USERS: 'neuromotion_users',
  CURRENT_USER: 'neuromotion_session',
  REPORTS: 'neuromotion_reports'
};

// Mock Auth & Database Service
export const storageService = {
  // --- Auth ---
  register: (name: string, email: string, password: string): User => {
    const users = JSON.parse(localStorage.getItem(STORAGE_KEYS.USERS) || '[]');
    const existing = users.find((u: any) => u.email === email);
    if (existing) throw new Error('Email already registered');

    const newUser = {
      id: crypto.randomUUID(),
      name,
      email,
      password // In a real app, hash this!
    };
    
    users.push(newUser);
    localStorage.setItem(STORAGE_KEYS.USERS, JSON.stringify(users));
    return { id: newUser.id, name: newUser.name, email: newUser.email };
  },

  login: (email: string, password: string): User => {
    let users = JSON.parse(localStorage.getItem(STORAGE_KEYS.USERS) || '[]');
    
    // Auto-seed for demo purposes if empty (since registration is hidden)
    if (users.length === 0) {
      const defaultUser = {
        id: 'user-demo-123',
        name: 'Dr. Demo User',
        email: 'demo@neuromotion.ai',
        password: 'demo'
      };
      users = [defaultUser];
      localStorage.setItem(STORAGE_KEYS.USERS, JSON.stringify(users));
    }

    const user = users.find((u: any) => u.email === email && u.password === password);
    
    if (!user) throw new Error('Invalid credentials');
    
    const userObj = { id: user.id, name: user.name, email: user.email };
    localStorage.setItem(STORAGE_KEYS.CURRENT_USER, JSON.stringify(userObj));
    return userObj;
  },

  logout: () => {
    localStorage.removeItem(STORAGE_KEYS.CURRENT_USER);
  },

  getCurrentUser: (): User | null => {
    const stored = localStorage.getItem(STORAGE_KEYS.CURRENT_USER);
    return stored ? JSON.parse(stored) : null;
  },

  // --- Data ---
  saveReport: (userId: string, report: AnalysisReport, videoName: string): SavedReport => {
    const reportsMap = JSON.parse(localStorage.getItem(STORAGE_KEYS.REPORTS) || '{}');
    const userReports = reportsMap[userId] || [];
    
    const newReport: SavedReport = {
      ...report,
      id: crypto.randomUUID(),
      date: new Date().toISOString(),
      videoName
    };

    userReports.unshift(newReport); // Add to top
    reportsMap[userId] = userReports;
    localStorage.setItem(STORAGE_KEYS.REPORTS, JSON.stringify(reportsMap));
    return newReport;
  },

  getReports: (userId: string): SavedReport[] => {
    const reportsMap = JSON.parse(localStorage.getItem(STORAGE_KEYS.REPORTS) || '{}');
    return reportsMap[userId] || [];
  },

  // --- Expert Feedback Loop ---
  saveExpertCorrection: (userId: string, reportId: string, correction: ExpertCorrection) => {
    const reportsMap = JSON.parse(localStorage.getItem(STORAGE_KEYS.REPORTS) || '{}');
    const userReports: SavedReport[] = reportsMap[userId] || [];
    
    const updatedReports = userReports.map(r => {
        if (r.id === reportId) {
            return { ...r, expertCorrection: correction };
        }
        return r;
    });

    reportsMap[userId] = updatedReports;
    localStorage.setItem(STORAGE_KEYS.REPORTS, JSON.stringify(reportsMap));
    return updatedReports.find(r => r.id === reportId);
  },

  getTrainingExamples: (): { inputs: any, groundTruth: ExpertCorrection }[] => {
    const reportsMap = JSON.parse(localStorage.getItem(STORAGE_KEYS.REPORTS) || '{}');
    const allExamples: { inputs: any, groundTruth: ExpertCorrection }[] = [];
    
    // Gather all corrected reports across all users to build the knowledge base
    Object.values(reportsMap).forEach((userReports: any) => {
        userReports.forEach((r: SavedReport) => {
            if (r.expertCorrection) {
                allExamples.push({
                    inputs: r.rawData,
                    groundTruth: r.expertCorrection
                });
            }
        });
    });

    // Return last 10 examples to ensure robust matching
    return allExamples.slice(0, 10); 
  },

  // --- Analytics for Dashboard ---
  getLearnedStats: (userId: string) => {
    const reportsMap = JSON.parse(localStorage.getItem(STORAGE_KEYS.REPORTS) || '{}');
    const userReports = reportsMap[userId] || [];
    
    const corrections = userReports.filter((r: SavedReport) => r.expertCorrection);
    const byCategory: Record<string, number> = {};
    
    corrections.forEach((r: SavedReport) => {
        const cat = r.expertCorrection!.correctClassification;
        byCategory[cat] = (byCategory[cat] || 0) + 1;
    });

    return {
        totalLearned: corrections.length,
        breakdown: byCategory
    };
  }
};
