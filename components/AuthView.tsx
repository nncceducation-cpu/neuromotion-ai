
import React, { useState } from 'react';
import { storageService } from '../services/storage';
import { User } from '../types';

interface AuthViewProps {
  onLogin: (user: User) => void;
}

export const AuthView: React.FC<AuthViewProps> = ({ onLogin }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      // Simulate network delay
      await new Promise(r => setTimeout(r, 800));
      
      const user = storageService.login(email, password);
      onLogin(user);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center px-4">
      <div className="bg-white p-8 rounded-2xl shadow-xl border border-slate-100 w-full max-w-md">
        <div className="text-center mb-8">
          <div className="w-12 h-12 bg-sky-500 rounded-xl flex items-center justify-center text-white font-bold text-xl mx-auto mb-4 shadow-lg shadow-sky-200">
            N
          </div>
          <h2 className="text-2xl font-bold text-slate-800">
            Sign In
          </h2>
          <p className="text-slate-500 mt-2 text-sm">
            Sign in to access your analysis history
          </p>
        </div>

        {error && (
          <div className="mb-6 p-3 bg-red-50 border border-red-100 text-red-600 text-sm rounded-lg flex items-center">
            <i className="fas fa-exclamation-circle mr-2"></i>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Email Address</label>
            <div className="relative">
              <i className="fas fa-envelope absolute left-3 top-3 text-slate-400"></i>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-transparent outline-none transition-all"
                placeholder="doctor@clinic.com"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1">Password</label>
            <div className="relative">
              <i className="fas fa-lock absolute left-3 top-3 text-slate-400"></i>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:ring-2 focus:ring-sky-500 focus:border-transparent outline-none transition-all"
                placeholder="••••••••"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-sky-600 text-white py-2.5 rounded-lg font-semibold hover:bg-sky-700 transition-colors shadow-lg shadow-sky-100 disabled:opacity-70 disabled:cursor-not-allowed mt-2"
          >
            {isLoading ? <i className="fas fa-circle-notch fa-spin"></i> : 'Sign In'}
          </button>
        </form>

        <div className="mt-6 text-center p-4 bg-slate-50 rounded-lg border border-slate-100">
          <p className="text-xs text-slate-500 font-medium">Demo Access</p>
          <div className="flex justify-between items-center text-xs text-slate-400 mt-2">
             <span>Email: demo@neuromotion.ai</span>
             <span>Pass: demo</span>
          </div>
        </div>
      </div>
    </div>
  );
};