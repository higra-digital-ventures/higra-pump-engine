import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Backend target — overridable via HPE_BACKEND_PORT so the dev proxy follows
// the backend when it falls back to another port (see scripts/start-all.js).
const BACKEND_PORT = process.env.HPE_BACKEND_PORT || '8000'
const BACKEND = `http://localhost:${BACKEND_PORT}`
const FRONTEND_PORT = Number(process.env.HPE_FRONTEND_PORT || 3000)

// All API path prefixes that should be proxied to the backend in dev mode.
// Keep in sync with frontend/nginx.conf location rules.
const API_PREFIXES = [
  '/api',
  '/health',
  '/docs',
  '/openapi.json',
  '/redoc',
  '/sizing',
  '/volute',
  '/pipeline',
  '/assistant',
  '/surrogate',
  '/geometry',
  '/optimize',
  '/analyze',
  '/curves',
  '/auth',
  '/batch',
  '/noise',
  '/report',
  '/version',
  '/inverse',
  '/blade',
  '/domain',
  '/blockage',
  '/ansys',
  '/lean-sweep',
  '/mri',
  '/turbo',
  '/lete',
  '/rrs',
  '/template',
  '/db',
  '/convergence',
  '/udp',
  '/cfd',
]

export default defineConfig({
  plugins: [react()],
  server: {
    port: FRONTEND_PORT,
    proxy: {
      // WebSocket — must come first
      '/ws': {
        target: BACKEND,
        ws: true,
        changeOrigin: true,
      },
      // All REST API routes
      ...Object.fromEntries(
        API_PREFIXES.map(p => [p, { target: BACKEND, changeOrigin: true }]),
      ),
    },
  },
})
