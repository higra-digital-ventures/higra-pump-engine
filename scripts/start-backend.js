/**
 * Start only the HPE backend (FastAPI/Uvicorn).
 *
 * If the preferred port is busy, the next free port is used automatically.
 * Override the preferred port with HPE_BACKEND_PORT.
 *
 * Usage:
 *   node scripts/start-backend.js
 *   npm run start:backend
 */

const { spawn } = require("child_process");
const path = require("path");
const { findFreePort } = require("./find-port");

const BACKEND_DIR = path.join(__dirname, "..", "backend");
const srcDir = path.join(BACKEND_DIR, "src");
const currentPythonPath = process.env.PYTHONPATH || "";
const pythonPath = currentPythonPath ? `${srcDir};${currentPythonPath}` : srcDir;

const PREFERRED_PORT = Number(process.env.HPE_BACKEND_PORT || 8000);

(async () => {
  const port = await findFreePort(PREFERRED_PORT);
  if (port !== PREFERRED_PORT) {
    console.log(`[backend] Port ${PREFERRED_PORT} busy — using ${port} instead`);
  }
  console.log(`[backend] http://localhost:${port}`);

  const child = spawn(
    "python",
    ["-m", "uvicorn", "hpe.api.app:app", "--reload", "--port", String(port)],
    {
      cwd: BACKEND_DIR,
      stdio: "inherit",
      shell: true,
      env: { ...process.env, PYTHONPATH: pythonPath },
    }
  );

  process.on("SIGINT", () => {
    child.kill("SIGTERM");
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    child.kill("SIGTERM");
    process.exit(0);
  });
})();
