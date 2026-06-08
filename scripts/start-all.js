/**
 * Unified startup orchestrator for HPE backend and frontend.
 *
 * Spawns both the Python (FastAPI/Uvicorn) backend and the
 * Node (Vite) frontend dev servers concurrently, forwarding
 * their output to the terminal with colored prefixes.
 *
 * Usage:
 *   node scripts/start-all.js
 *   npm run start:all
 */

const { spawn, execSync } = require("child_process");
const path = require("path");
const fs = require("fs");
const { findFreePort } = require("./find-port");

const PREFERRED_BACKEND_PORT = Number(process.env.HPE_BACKEND_PORT || 8000);
const PREFERRED_FRONTEND_PORT = Number(process.env.HPE_FRONTEND_PORT || 3000);

const ROOT = path.resolve(__dirname, "..");
const BACKEND_DIR = path.join(ROOT, "backend");
const FRONTEND_DIR = path.join(ROOT, "frontend");

// ANSI colors for log prefixes
const CYAN = "\x1b[36m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const RESET = "\x1b[0m";

const children = [];

function spawnProcess(name, command, args, cwd, color, env) {
  console.log(`${color}[${name}]${RESET} Starting: ${command} ${args.join(" ")}`);

  const child = spawn(command, args, {
    cwd,
    stdio: ["inherit", "pipe", "pipe"],
    shell: true,
    env: { ...process.env, ...env },
  });

  child.stdout.on("data", (data) => {
    const lines = data.toString().trimEnd().split("\n");
    lines.forEach((line) => {
      console.log(`${color}[${name}]${RESET} ${line}`);
    });
  });

  child.stderr.on("data", (data) => {
    const lines = data.toString().trimEnd().split("\n");
    lines.forEach((line) => {
      console.log(`${color}[${name}]${RESET} ${line}`);
    });
  });

  child.on("close", (code) => {
    console.log(`${color}[${name}]${RESET} Process exited with code ${code}`);
  });

  children.push(child);
  return child;
}

// Graceful shutdown
function cleanup() {
  console.log("\nShutting down...");
  children.forEach((child) => {
    try {
      child.kill("SIGTERM");
    } catch (_) {
      // Process may already be dead
    }
  });
  process.exit(0);
}

process.on("SIGINT", cleanup);
process.on("SIGTERM", cleanup);

// --- Ensure frontend dependencies are installed ---
const nodeModules = path.join(FRONTEND_DIR, "node_modules");
if (!fs.existsSync(nodeModules)) {
  console.log(`${YELLOW}[setup]${RESET} Installing frontend dependencies...`);
  execSync("npm install", { cwd: FRONTEND_DIR, stdio: "inherit" });
}

// Add backend/src to PYTHONPATH so uvicorn can find the hpe package
const srcDir = path.join(BACKEND_DIR, "src");
const currentPythonPath = process.env.PYTHONPATH || "";
const pythonPath = currentPythonPath ? `${srcDir};${currentPythonPath}` : srcDir;

(async () => {
  // Pick free ports, falling back to the next available one if busy.
  const backendPort = await findFreePort(PREFERRED_BACKEND_PORT);
  const frontendPort = await findFreePort(PREFERRED_FRONTEND_PORT);

  if (backendPort !== PREFERRED_BACKEND_PORT) {
    console.log(`${YELLOW}[setup]${RESET} Backend port ${PREFERRED_BACKEND_PORT} busy — using ${backendPort}`);
  }
  if (frontendPort !== PREFERRED_FRONTEND_PORT) {
    console.log(`${YELLOW}[setup]${RESET} Frontend port ${PREFERRED_FRONTEND_PORT} busy — using ${frontendPort}`);
  }

  // --- Start Backend ---
  spawnProcess(
    "backend",
    "python",
    ["-m", "uvicorn", "hpe.api.app:app", "--reload", "--port", String(backendPort)],
    BACKEND_DIR,
    CYAN,
    { PYTHONPATH: pythonPath }
  );

  // --- Start Frontend ---
  // Use npx to ensure vite is found from local node_modules.
  // HPE_BACKEND_PORT tells vite where to proxy; HPE_FRONTEND_PORT sets its port.
  spawnProcess(
    "frontend",
    "npx",
    ["vite", "--port", String(frontendPort), "--strictPort"],
    FRONTEND_DIR,
    GREEN,
    { HPE_BACKEND_PORT: String(backendPort), HPE_FRONTEND_PORT: String(frontendPort) }
  );

  console.log(`
${CYAN}[backend]${RESET}  http://localhost:${backendPort}
${GREEN}[frontend]${RESET} http://localhost:${frontendPort}
`);
})();
