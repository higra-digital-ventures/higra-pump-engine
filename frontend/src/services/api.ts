const API_BASE = '/api/v1'
const API_V2_BASE = ''

export async function runSizing(flowRate: number, head: number, rpm: number) {
  const res = await fetch(`${API_BASE}/sizing`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ flow_rate: flowRate, head, rpm }),
  })
  if (!res.ok) throw new Error(`Sizing failed: ${res.status}`)
  return res.json()
}

// --- Geometry modes (classic vs free) ---

export interface GeometryBackendInfo {
  name: string
  mode: string | null
  capabilities: {
    supports_step: boolean
    supports_stl: boolean
    supports_voxel: boolean
    supports_internal_channels: boolean
    mesh_strategy: string
  }
}

export async function listGeometryBackends(): Promise<GeometryBackendInfo[]> {
  const res = await fetch(`${API_BASE}/geometry/backends`)
  if (!res.ok) throw new Error(`Backends failed: ${res.status}`)
  const data = await res.json()
  return data.backends ?? []
}

export async function generateGeometryMode(
  flowRate: number, head: number, rpm: number, mode: string, exportFiles = false,
) {
  const res = await fetch(`${API_BASE}/geometry/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ Q: flowRate, H: head, n: rpm, mode, export: exportFiles }),
  })
  if (res.status === 501) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || 'Modo não implementado')
  }
  if (!res.ok) throw new Error(`Geometry generate failed: ${res.status}`)
  return res.json()
}

export async function getCurves(flowRate: number, head: number, rpm: number, nPoints = 25) {
  const res = await fetch(`${API_BASE}/curves`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ flow_rate: flowRate, head, rpm, n_points: nPoints }),
  })
  if (!res.ok) throw new Error(`Curves failed: ${res.status}`)
  return res.json()
}

export async function runOptimize(
  flowRate: number, head: number, rpm: number,
  method = 'nsga2', popSize = 20, nGen = 20,
) {
  const res = await fetch(`${API_BASE}/optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      flow_rate: flowRate, head, rpm, method,
      pop_size: popSize, n_gen: nGen,
    }),
  })
  if (!res.ok) throw new Error(`Optimize failed: ${res.status}`)
  return res.json()
}

export async function runStressAnalysis(flowRate: number, head: number, rpm: number) {
  const res = await fetch(`${API_BASE}/stress`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ flow_rate: flowRate, head, rpm }),
  })
  if (!res.ok) throw new Error(`Stress failed: ${res.status}`)
  return res.json()
}

export async function runInverseDesign(
  flowRate: number, head: number, rpm: number,
  loadingType = 'mid_loaded',
) {
  const res = await fetch(`${API_BASE}/inverse`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ flow_rate: flowRate, head, rpm, loading_type: loadingType }),
  })
  if (!res.ok) throw new Error(`Inverse failed: ${res.status}`)
  return res.json()
}

export async function getLossBreakdown(flowRate: number, head: number, rpm: number) {
  const res = await fetch(`${API_BASE}/losses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ flow_rate: flowRate, head, rpm }),
  })
  if (!res.ok) throw new Error(`Losses failed: ${res.status}`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Version History API
// ---------------------------------------------------------------------------

export interface VersionEntry {
  id: string
  version_number: number
  label: string
  nq: number
  eta: number
  d2_mm: number
  npsh: number
  power_kw: number
  flow_rate: number
  head: number
  rpm: number
  created_at: string
}

export interface VersionDetail {
  id: string
  version_number: number
  label: string
  created_at: string
  operating_point: { flow_rate: number; head: number; rpm: number }
  sizing_result: Record<string, any>
}

export interface VersionCompareResult {
  a: { version: VersionEntry; sizing_result: Record<string, any> }
  b: { version: VersionEntry; sizing_result: Record<string, any> }
  deltas: Record<string, { a: number; b: number; delta: number; pct: number }>
}

export async function saveVersion(
  operatingPoint: { flow_rate: number; head: number; rpm: number },
  sizingResult: Record<string, any>,
  projectId?: string,
  label?: string,
): Promise<VersionEntry> {
  const res = await fetch(`${API_BASE}/versions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      project_id: projectId || null,
      operating_point: operatingPoint,
      sizing_result: sizingResult,
      label: label || null,
    }),
  })
  if (!res.ok) throw new Error(`Save version failed: ${res.status}`)
  return res.json()
}

export async function listVersions(projectId?: string, limit = 50): Promise<VersionEntry[]> {
  const params = new URLSearchParams()
  if (projectId) params.set('project_id', projectId)
  params.set('limit', String(limit))
  const res = await fetch(`${API_BASE}/versions?${params}`)
  if (!res.ok) throw new Error(`List versions failed: ${res.status}`)
  return res.json()
}

export async function getVersion(id: string): Promise<VersionDetail> {
  const res = await fetch(`${API_BASE}/versions/${id}`)
  if (!res.ok) throw new Error(`Get version failed: ${res.status}`)
  return res.json()
}

export async function compareVersions(a: string, b: string): Promise<VersionCompareResult> {
  const res = await fetch(`${API_BASE}/versions/compare`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ version_a: a, version_b: b }),
  })
  if (!res.ok) throw new Error(`Compare versions failed: ${res.status}`)
  return res.json()
}

export async function deleteVersion(id: string): Promise<void> {
  const res = await fetch(`${API_BASE}/versions/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`Delete version failed: ${res.status}`)
}

// ---------------------------------------------------------------------------
// CSV Export (client-side)
// ---------------------------------------------------------------------------

export function exportSizingCSV(sizing: any, opPoint: any): void {
  const rows = [
    ['Parâmetro', 'Valor', 'Unidade'],
    ['Vazão Q', (opPoint.flowRate).toFixed(2), 'm³/h'],
    ['Altura H', opPoint.head.toFixed(1), 'm'],
    ['Rotação n', opPoint.rpm.toFixed(0), 'rpm'],
    ['Nq', sizing.specific_speed_nq.toFixed(1), ''],
    ['D2', (sizing.impeller_d2*1000).toFixed(0), 'mm'],
    ['D1', (sizing.impeller_d1*1000).toFixed(0), 'mm'],
    ['b2', (sizing.impeller_b2*1000).toFixed(1), 'mm'],
    ['Z pas', sizing.blade_count, ''],
    ['beta1', sizing.beta1.toFixed(1), 'deg'],
    ['beta2', sizing.beta2.toFixed(1), 'deg'],
    ['eta total', (sizing.estimated_efficiency*100).toFixed(1), '%'],
    ['Potência', (sizing.estimated_power/1000).toFixed(1), 'kW'],
    ['NPSHr', sizing.estimated_npsh_r.toFixed(1), 'm'],
  ]
  const csv = rows.map(r => r.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a'); a.href = url; a.download = 'hpe-sizing.csv'; a.click()
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// HPE API v2.0 endpoints
// ---------------------------------------------------------------------------

export interface SizingV2Input {
  Q: number    // m³/s
  H: number    // m
  n: number    // rpm
  fluid?: string
  rho?: number
  nu?: number
}

export interface SizingV2Output {
  Ns: number; Nq: number; omega_s: number
  D1: number; D2: number; b2: number
  beta1: number; beta2: number; u2: number
  eta_hid: number; eta_total: number
  P_shaft: number; NPSHr: number
  warnings: string[]
  computation_time_ms: number
}

export async function runSizingV2(inp: SizingV2Input): Promise<SizingV2Output> {
  const res = await fetch(`${API_V2_BASE}/sizing/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp),
  })
  if (!res.ok) throw new Error(`SizingV2 failed: ${res.status}`)
  return res.json()
}

export interface GeometryOutput {
  params: {
    D2_mm: number; D1_mm: number; D1_hub_mm: number
    b2_mm: number; b1_mm: number
    beta1_deg: number; beta2_deg: number
    blade_count: number; blade_thickness_mm: number; wrap_angle_deg: number
  }
  meridional_hub_r_mm: number[]
  meridional_hub_z_mm: number[]
  meridional_shroud_r_mm: number[]
  meridional_shroud_z_mm: number[]
  blade_camber_r_mm: number[]
  blade_camber_theta_deg: number[]
  cad_available: boolean
  step_path: string | null
  warnings: string[]
  generation_time_ms: number
}

export async function runGeometry(inp: SizingV2Input): Promise<GeometryOutput> {
  const res = await fetch(`${API_V2_BASE}/geometry/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp),
  })
  if (!res.ok) throw new Error(`Geometry failed: ${res.status}`)
  return res.json()
}

export interface VoluteOutput {
  throat_area_mm2: number
  tongue_radius_mm: number
  exit_diameter_mm: number
  casing_width_mm: number
  D2_mm: number
  warnings: string[]
}

export async function runVolute(inp: SizingV2Input & { tongue_clearance?: number }): Promise<VoluteOutput> {
  const res = await fetch(`${API_V2_BASE}/volute/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp),
  })
  if (!res.ok) throw new Error(`Volute failed: ${res.status}`)
  return res.json()
}

export interface SurrogatePredictInput {
  Ns: number; D2: number; b2: number; beta2: number
  n: number; Q: number; H: number; n_stages?: number
}

export interface SurrogatePredictOutput {
  eta_hid: number; eta_total: number; H: number; P_shaft: number
  confidence: number; surrogate_version: string; latency_ms: number
}

export async function predictSurrogate(inp: SurrogatePredictInput): Promise<SurrogatePredictOutput> {
  const res = await fetch(`${API_V2_BASE}/surrogate/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp),
  })
  if (!res.ok) throw new Error(`Surrogate predict failed: ${res.status}`)
  return res.json()
}

export interface SimilarDesign {
  ns: number; d2_mm: number; eta_total: number; fonte: string; qualidade: number
  modelo_bomba?: string
}

export async function getSimilarDesigns(ns: number, d2_mm: number, limit?: number): Promise<SimilarDesign[]> {
  const params = new URLSearchParams({ ns: String(ns), d2_mm: String(d2_mm) })
  if (limit !== undefined) params.set('limit', String(limit))
  const res = await fetch(`${API_V2_BASE}/surrogate/similar?${params}`)
  if (!res.ok) throw new Error(`Similar designs failed: ${res.status}`)
  return res.json()
}

// Pipeline assíncrono
export interface PipelineRunResult {
  run_id: string
  mode: 'async' | 'sync'
  task_id?: string
  D2_mm?: number
  eta?: number
  elapsed_ms?: number
}

export async function startPipeline(inp: SizingV2Input): Promise<PipelineRunResult> {
  const res = await fetch(`${API_V2_BASE}/pipeline/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(inp),
  })
  if (!res.ok) throw new Error(`Pipeline start failed: ${res.status}`)
  return res.json()
}

// WebSocket status
export type PipelineStatus = {
  run_id: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'not_found'
  stage?: string
  progress?: number
  elapsed_s?: number
  message?: string
  result?: Record<string, unknown>
  error?: string
}

export function subscribePipelineStatus(
  run_id: string,
  onUpdate: (status: PipelineStatus) => void,
  onDone: (status: PipelineStatus) => void,
): () => void {
  const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const wsHost = window.location.host
  const ws = new WebSocket(`${wsProto}://${wsHost}/ws/pipeline/${run_id}`)

  ws.onmessage = (event) => {
    try {
      const status: PipelineStatus = JSON.parse(event.data)
      onUpdate(status)
      if (status.status === 'completed' || status.status === 'failed') {
        onDone(status)
        ws.close()
      }
    } catch {
      // ignore malformed messages
    }
  }

  ws.onerror = () => {
    const errorStatus: PipelineStatus = {
      run_id,
      status: 'failed',
      error: 'WebSocket connection error',
    }
    onDone(errorStatus)
  }

  // cleanup function
  return () => {
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close()
    }
  }
}

// Assistant RAG
export interface AssistantRequest {
  question: string
  context?: {
    Ns?: number; D2_mm?: number; eta?: number; NPSHr?: number
    warnings?: string[]
  }
}

export interface AssistantResponse {
  answer: string
  relevant_topics: string[]
  recommendations: string[]
  references: string[]
  confidence: number
  mode: string
}

export async function askAssistant(req: AssistantRequest): Promise<AssistantResponse> {
  const res = await fetch(`${API_V2_BASE}/assistant/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`Assistant ask failed: ${res.status}`)
  return res.json()
}
