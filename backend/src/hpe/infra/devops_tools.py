"""DevOps + multi-tenant + dataset versioning — melhorias #91-100."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


# ===========================================================================
# #91 Docker GPU support
# ===========================================================================

def docker_gpu_check() -> dict:
    """Detectar GPU disponível para CUDA-enabled solvers."""
    import shutil
    import subprocess
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"gpu_available": False, "reason": "nvidia-smi not found"}
    try:
        out = subprocess.run([nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        gpus = [
            {"name": line.split(",")[0].strip(), "memory": line.split(",")[1].strip()}
            for line in out.stdout.strip().splitlines() if line
        ]
        return {"gpu_available": len(gpus) > 0, "gpus": gpus}
    except Exception as exc:
        return {"gpu_available": False, "reason": str(exc)}


# ===========================================================================
# #92 K8s deployment manifest generator
# ===========================================================================

def generate_k8s_manifest(
    image: str = "hpe:latest",
    replicas: int = 2,
    cpu: str = "2", memory: str = "4Gi",
) -> str:
    return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: hpe-backend
  labels:
    app: hpe
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: hpe
  template:
    metadata:
      labels:
        app: hpe
    spec:
      containers:
      - name: hpe
        image: {image}
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "{cpu}"
            memory: "{memory}"
          limits:
            cpu: "{cpu}"
            memory: "{memory}"
        env:
        - name: HPE_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: hpe-secrets
              key: database_url
---
apiVersion: v1
kind: Service
metadata:
  name: hpe-backend
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: hpe
"""


# ===========================================================================
# #93 Multi-tenant
# ===========================================================================

@dataclass
class Tenant:
    tenant_id: str
    name: str
    quota_simulations: int = 100
    used_simulations: int = 0
    storage_gb: int = 10
    created_at: float = field(default_factory=time.time)


class TenantManager:
    """Gerenciamento simples de tenants em memória."""

    def __init__(self):
        self._tenants: dict[str, Tenant] = {}

    def create(self, name: str, quota: int = 100) -> Tenant:
        tid = uuid.uuid4().hex[:12]
        t = Tenant(tenant_id=tid, name=name, quota_simulations=quota)
        self._tenants[tid] = t
        return t

    def get(self, tid: str) -> Optional[Tenant]:
        return self._tenants.get(tid)

    def can_simulate(self, tid: str) -> bool:
        t = self.get(tid)
        return bool(t and t.used_simulations < t.quota_simulations)

    def increment(self, tid: str) -> None:
        t = self.get(tid)
        if t:
            t.used_simulations += 1


# ===========================================================================
# #94 RBAC roles
# ===========================================================================

@dataclass
class Role:
    name: str
    permissions: list[str]


_ROLES = {
    "admin":     Role("admin", ["*"]),
    "engineer":  Role("engineer", ["sizing.run", "geometry.run", "cfd.run", "optimize.run", "report.create"]),
    "viewer":    Role("viewer", ["sizing.read", "report.read"]),
    "developer": Role("developer", ["sizing.run", "cfd.run", "api.debug"]),
}


def has_permission(role_name: str, action: str) -> bool:
    role = _ROLES.get(role_name)
    if not role:
        return False
    return "*" in role.permissions or action in role.permissions


# ===========================================================================
# #95 Audit log
# ===========================================================================

@dataclass
class AuditEntry:
    timestamp: float
    user: str
    action: str
    resource: str
    details: dict
    ip: str = ""


class AuditLog:
    def __init__(self):
        self._entries: list[AuditEntry] = []

    def log(self, user: str, action: str, resource: str, **details) -> None:
        self._entries.append(AuditEntry(
            timestamp=time.time(),
            user=user, action=action, resource=resource,
            details=details,
        ))

    def query(
        self, user: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[float] = None, limit: int = 100,
    ) -> list[AuditEntry]:
        out = self._entries[-limit:]
        if user:
            out = [e for e in out if e.user == user]
        if action:
            out = [e for e in out if e.action == action]
        if since:
            out = [e for e in out if e.timestamp >= since]
        return out

    def export_json(self) -> str:
        return json.dumps([{
            "ts": e.timestamp, "user": e.user, "action": e.action,
            "resource": e.resource, "details": e.details,
        } for e in self._entries], indent=2)


# ===========================================================================
# #96 Cost tracking per simulation
# ===========================================================================

@dataclass
class SimulationCost:
    simulation_id: str
    cpu_hours: float
    memory_gb_hours: float
    storage_gb: float
    estimated_cost_usd: float


def estimate_cost(
    n_procs: int, duration_seconds: float,
    memory_gb: float = 4.0,
    cpu_cost_per_hour: float = 0.05,
    memory_cost_per_gb_hour: float = 0.01,
) -> SimulationCost:
    cpu_hours = n_procs * duration_seconds / 3600
    mem_gb_hours = memory_gb * duration_seconds / 3600
    cost = cpu_hours * cpu_cost_per_hour + mem_gb_hours * memory_cost_per_gb_hour
    return SimulationCost(
        simulation_id=uuid.uuid4().hex[:12],
        cpu_hours=cpu_hours,
        memory_gb_hours=mem_gb_hours,
        storage_gb=0.5,
        estimated_cost_usd=round(cost, 4),
    )


# ===========================================================================
# #97 Queue priorities
# ===========================================================================

@dataclass
class JobPriority:
    job_id: str
    priority: int      # 0=low, 5=normal, 10=high
    submitted_at: float = field(default_factory=time.time)


class PriorityQueue:
    def __init__(self):
        self._jobs: list[JobPriority] = []

    def submit(self, job_id: str, priority: int = 5) -> None:
        self._jobs.append(JobPriority(job_id=job_id, priority=priority))
        self._jobs.sort(key=lambda j: (-j.priority, j.submitted_at))

    def next_job(self) -> Optional[JobPriority]:
        return self._jobs.pop(0) if self._jobs else None

    def __len__(self) -> int:
        return len(self._jobs)


# ===========================================================================
# #98 GPU scheduling
# ===========================================================================

class GPUScheduler:
    def __init__(self, n_gpus: int = 1):
        self._gpus = [None] * n_gpus

    def acquire(self, job_id: str) -> Optional[int]:
        for i, occupant in enumerate(self._gpus):
            if occupant is None:
                self._gpus[i] = job_id
                return i
        return None

    def release(self, gpu_id: int) -> None:
        if 0 <= gpu_id < len(self._gpus):
            self._gpus[gpu_id] = None


# ===========================================================================
# #99 Dataset versioning
# ===========================================================================

@dataclass
class DatasetVersion:
    name: str
    version: str
    hash: str
    n_records: int
    created_at: float
    parent: Optional[str] = None


def hash_dataset(records: list) -> str:
    payload = json.dumps(records, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def version_dataset(name: str, records: list, parent: Optional[str] = None) -> DatasetVersion:
    h = hash_dataset(records)
    return DatasetVersion(
        name=name,
        version=f"v{int(time.time())}",
        hash=h,
        n_records=len(records),
        created_at=time.time(),
        parent=parent,
    )


# ===========================================================================
# #100 Model registry
# ===========================================================================

@dataclass
class ModelEntry:
    model_id: str
    name: str
    version: str
    metric: float
    artifact_path: str
    promoted: bool = False
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, list[ModelEntry]] = {}

    def register(self, name: str, version: str, metric: float, artifact_path: str) -> ModelEntry:
        entry = ModelEntry(
            model_id=uuid.uuid4().hex[:12], name=name, version=version,
            metric=metric, artifact_path=artifact_path,
        )
        self._models.setdefault(name, []).append(entry)
        return entry

    def best(self, name: str) -> Optional[ModelEntry]:
        if name not in self._models:
            return None
        return max(self._models[name], key=lambda m: m.metric)

    def promote(self, name: str, version: str) -> bool:
        if name not in self._models:
            return False
        for e in self._models[name]:
            e.promoted = (e.version == version)
        return True

    def list_models(self) -> list[ModelEntry]:
        return [e for entries in self._models.values() for e in entries]
