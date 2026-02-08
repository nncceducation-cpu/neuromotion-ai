
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
      <div className="bg-white p-8 rounded-lg border border-neutral-200 w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="w-10 h-10 bg-neutral-900 rounded-lg flex items-center justify-center text-white font-semibold text-sm mx-auto mb-4">
            N
          </div>
          <h2 className="text-xl font-semibold text-neutral-900">
            Sign in to NeuroMotion
          </h2>
          <p className="text-neutral-500 mt-1 text-sm">
            Enter your credentials to continue
          </p>
        </div>

        {error && (
          <div className="mb-6 p-3 bg-neutral-50 border border-neutral-200 text-neutral-700 text-sm rounded-lg">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-neutral-700 mb-1.5">Email</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-neutral-900 focus:border-transparent outline-none transition-all text-sm"
              placeholder="doctor@clinic.com"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-neutral-700 mb-1.5">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-neutral-200 rounded-lg focus:ring-2 focus:ring-neutral-900 focus:border-transparent outline-none transition-all text-sm"
              placeholder="••••••••"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full bg-neutral-900 text-white py-2.5 rounded-lg font-medium hover:bg-neutral-800 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm"
          >
            {isLoading ? <i className="fas fa-circle-notch fa-spin"></i> : 'Continue'}
          </button>
        </form>

        <div className="mt-6 text-center p-3 bg-neutral-50 rounded-lg border border-neutral-200">
          <p className="text-[10px] text-neutral-400 uppercase tracking-wider font-medium mb-1">Demo Access</p>
          <p className="text-xs text-neutral-500 font-mono">demo@neuromotion.ai / demo</p>
        </div>
      </div>
    </div>
  );
};
