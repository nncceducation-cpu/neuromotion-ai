import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        // FIX: Add proxy to redirect API calls to FastAPI
        proxy: {
          '/analyze_frames': {
            target: 'http://127.0.0.1:8000',
            changeOrigin: true,
          },
          '/upload_video': {
            target: 'http://127.0.0.1:8000',
            changeOrigin: true,
          },
          '/health': {
             target: 'http://127.0.0.1:8000',
             changeOrigin: true,
          }
        }
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});