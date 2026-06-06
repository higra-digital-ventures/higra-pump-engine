import React, { useState, useEffect, useCallback, useRef } from 'react'
import t from './i18n'
import { type Tab, WIDE_TABS } from './navigation'
import Layout from './components/Layout'
import LoginPage from './pages/LoginPage'
import ProjectsPage from './pages/ProjectsPage'
import SizingForm from './pages/SizingForm'
import ResultsView from './pages/ResultsView'
import CurvesChart from './components/CurvesChart'
import VelocityTriangle from './components/VelocityTriangle'
import LossBreakdownChart from './components/LossBreakdownChart'
import StressView from './components/StressView'
import ImpellerViewer from './components/ImpellerViewer'
import DesignComparison from './components/DesignComparison'
import AssistantChat from './components/AssistantChat'
import PipelinePanel from './components/PipelinePanel'
import CavitationPanel from './components/CavitationPanel'
import CFDSimPanel from './components/CFDSimPanel'
import BenchmarksPanel from './components/BenchmarksPanel'
import MeridionalView from './components/MeridionalView'
import OptimizePanel from './components/OptimizePanel'
import EfficiencyMap from './components/EfficiencyMap'
import LoadingEditor from './components/LoadingEditor'
import PressureDistribution from './components/PressureDistribution'
import MultiSpeedChart from './components/MultiSpeedChart'
import MeridionalEditor from './components/MeridionalEditor'
import SpanwiseLoadingChart from './components/SpanwiseLoadingChart'
import ReferencePanel from './components/ReferencePanel'
import ExportPanel from './components/ExportPanel'
import ImpellerMiniPreview from './components/ImpellerMiniPreview'
import RadarChart from './components/RadarChart'
import DoEPanel from './components/DoEPanel'
import ParetoPanel from './components/ParetoPanel'
import LeanSweepPanel from './components/LeanSweepPanel'
import LETEEditor from './components/LETEEditor'
import MeridionalDragEditor from './components/MeridionalDragEditor'
import OptimizationPresets from './components/OptimizationPresets'
import TemplateSelector from './components/TemplateSelector'
import StatusBar from './components/StatusBar'
import DesignDashboard from './components/DesignDashboard'
import ProgressStepper from './components/ProgressStepper'
import ResultsSkeleton from './components/ResultsSkeleton'
import CommandPalette from './components/CommandPalette'
import Toast from './components/Toast'
import HistoryPanel from './components/HistoryPanel'
import type { HistoryEntry } from './components/HistoryPanel'
import NextStepBanner from './components/NextStepBanner'
import VersionPanel from './components/VersionPanel'
import VersionCompareModal from './components/VersionCompareModal'
import GuidedTour from './components/GuidedTour'
import FloatingMetrics from './components/FloatingMetrics'
import DesignQualityBadge from './components/DesignQualityBadge'
import EvolutionSparkline from './components/EvolutionSparkline'
import ContextualHelp from './components/ContextualHelp'
import ExportCenter from './components/ExportCenter'
import WhatsNew from './components/WhatsNew'
import FeatureTip from './components/FeatureTip'
import ActionTimeline from './components/ActionTimeline'
import type { TimelineEntry } from './components/ActionTimeline'
import SectionGuide from './components/SectionGuide'
import Glossary from './components/Glossary'
import CompleteResultView from './components/CompleteResultView'
import GlobalContextMenu from './components/GlobalContextMenu'
import { incrementSizingCount } from './components/ProgressBadge'
import FeedbackStars from './components/FeedbackStars'
import QuickSummary from './components/QuickSummary'
import ProjectChecklist from './components/ProjectChecklist'
import { emailResults } from './utils/emailResults'
import useDynamicFavicon from './hooks/useDynamicFavicon'
import { useToast } from './hooks/useToast'
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts'
import { runSizing, getCurves, getLossBreakdown, runStressAnalysis, saveVersion, listVersions, getVersion, compareVersions, deleteVersion as apiDeleteVersion } from './services/api'
import type { VersionEntry, VersionCompareResult } from './services/api'

/* Simple inline panels for Noise and Batch until full components are built */
function NoisePanel({ flowRate, head, rpm, sizing }: any) {
  const [result, setResult] = React.useState<any>(null)
  const [loading, setLoading] = React.useState(false)
  const run = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/v1/analysis/noise', { method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ flow_rate: flowRate, head, rpm }) })
      if (r.ok) setResult(await r.json())
    } finally { setLoading(false) }
  }
  return (
    <div>
      <button className="btn-primary" onClick={run} disabled={loading} style={{ fontSize: 13, padding: '8px 16px' }}>
        {loading ? 'Calculando...' : 'Calcular Ruído'}
      </button>
      {result && (
        <div style={{ marginTop: 16, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, fontSize: 13 }}>
          <div>Lw total: <b style={{ color: 'var(--accent)' }}>{result.Lw_total_dB?.toFixed(1) ?? '—'} dB</b></div>
          <div>Lw(A): <b style={{ color: 'var(--accent)' }}>{result.Lw_A_weighted_dB?.toFixed(1) ?? '—'} dBA</b></div>
          <div>BPF: <b>{result.bpf_hz?.toFixed(0) ?? '—'} Hz</b></div>
          <div>Fonte dominante: <b>{result.dominant_source ?? '—'}</b></div>
        </div>
      )}
    </div>
  )
}

function BatchPanel({ baseFlowRate, baseHead, baseRpm }: any) {
  const [results, setResults] = React.useState<any[]>([])
  const [loading, setLoading] = React.useState(false)
  const [variable, setVariable] = React.useState('rpm')
  const run = async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/v1/batch/parametric', { method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ flow_rate: baseFlowRate, head: baseHead, rpm: baseRpm, sweep_variable: variable, n_points: 10 }) })
      if (r.ok) { const d = await r.json(); setResults(d.results || d) }
    } finally { setLoading(false) }
  }
  return (
    <div>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12 }}>
        <select value={variable} onChange={e => setVariable(e.target.value)} className="input" style={{ width: 160 }}>
          <option value="rpm">RPM</option><option value="flow_rate">Vazão</option><option value="head">Altura</option>
        </select>
        <button className="btn-primary" onClick={run} disabled={loading} style={{ fontSize: 13, padding: '8px 16px' }}>
          {loading ? 'Executando...' : 'Sweep Paramétrico'}
        </button>
      </div>
      {results.length > 0 && (
        <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
          <thead><tr style={{ borderBottom: '1px solid var(--border-primary)' }}>
            <th style={{ padding: 6, textAlign: 'left', color: 'var(--text-muted)' }}>Nq</th>
            <th style={{ padding: 6, textAlign: 'left', color: 'var(--text-muted)' }}>D2 [mm]</th>
            <th style={{ padding: 6, textAlign: 'left', color: 'var(--text-muted)' }}>η [%]</th>
            <th style={{ padding: 6, textAlign: 'left', color: 'var(--text-muted)' }}>Power [kW]</th>
          </tr></thead>
          <tbody>{results.map((r: any, i: number) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--border-subtle)' }}>
              <td style={{ padding: 6 }}>{(r.specific_speed_nq ?? r.nq)?.toFixed(1)}</td>
              <td style={{ padding: 6 }}>{((r.impeller_d2 ?? r.d2) * 1000)?.toFixed(0)}</td>
              <td style={{ padding: 6 }}>{((r.estimated_efficiency ?? r.eta) * 100)?.toFixed(1)}</td>
              <td style={{ padding: 6 }}>{((r.estimated_power ?? r.power) / 1000)?.toFixed(1)}</td>
            </tr>
          ))}</tbody>
        </table>
      )}
    </div>
  )
}

function EmptyStateHint({ label }: { label: string }) {
  return (
    <div style={{
      padding: 24, textAlign: 'center', color: 'var(--text-muted)',
      fontSize: 13, border: '1px dashed var(--border-primary)', borderRadius: 8,
    }}>
      {label}
    </div>
  )
}

export interface SizingResult {
  specific_speed_nq: number
  impeller_type: string
  impeller_d2: number
  impeller_d1: number
  impeller_b2: number
  blade_count: number
  beta1: number
  beta2: number
  estimated_efficiency: number
  estimated_power: number
  estimated_npsh_r: number
  sigma: number
  velocity_triangles: Record<string, any>
  meridional_profile: Record<string, any>
  warnings: string[]
  uncertainty?: Record<string, number>
}

export interface CurvePoint {
  flow_rate: number
  head: number
  efficiency: number
  power: number
  npsh_required: number
  is_unstable?: boolean
}

type Page = 'login' | 'projects' | 'design'
export type { Tab }

export default function App() {
  const [page, setPage] = useState<Page>('login')
  const [user, setUser] = useState<any>(null)
  const [token, setToken] = useState('')
  const [currentProject, setCurrentProject] = useState<any>(null)
  const [tab, setTab] = useState<Tab>('results')

  const [sizing, setSizing] = useState<SizingResult | null>(null)
  const [curves, setCurves] = useState<CurvePoint[]>([])
  const [losses, setLosses] = useState<any>(null)
  const [stress, setStress] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [opPoint, setOpPoint] = useState({ flowRate: 0, head: 0, rpm: 0 })
  // const [advancedMode, setAdvancedMode] = useState(false) // removed from header — available in sidebar
  const [saving, setSaving] = useState(false)
  const [savedId, setSavedId] = useState<string | null>(null)
  const [cmdOpen, setCmdOpen] = useState(false)
  const [shortcutsHelpOpen, setShortcutsHelpOpen] = useState(false)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [previousSizing, setPreviousSizing] = useState<SizingResult | null>(null)
  const resultsRef = useRef<HTMLDivElement | null>(null)
  const [tourActive, setTourActive] = useState(() => !localStorage.getItem('hpe_tour_completed'))
  const [completedSteps, setCompletedSteps] = useState<string[]>([])
  const [versions, setVersions] = useState<VersionEntry[]>([])
  const [currentVersionId, setCurrentVersionId] = useState<string | null>(null)
  const [compareData, setCompareData] = useState<VersionCompareResult | null>(null)
  const [helpOpen, setHelpOpen] = useState(false)
  const [exportCenterOpen, setExportCenterOpen] = useState(false)
  const [tabHistory, setTabHistory] = useState<Tab[]>([])
  const [timeline, setTimeline] = useState<TimelineEntry[]>([])
  const [timelineOpen, setTimelineOpen] = useState(false)
  const [showWhatsNew, setShowWhatsNew] = useState(() => localStorage.getItem('hpe_whats_new_seen') !== '0.2.0')
  const [glossaryOpen, setGlossaryOpen] = useState(false)
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number } | null>(null)
  const [editingProjectName, setEditingProjectName] = useState(false)
  const [editProjectNameVal, setEditProjectNameVal] = useState('')
  const [overviewTab, setOverviewTab] = useState(false)  // CompleteResultView toggle (#10)
  const [showAutoRestore, setShowAutoRestore] = useState(false)
  const { toasts, toast, dismiss } = useToast()

  // #1 — unsaved indicator
  const [isDirty, setIsDirty] = useState(false)
  // #2 — granular loading step
  const [loadingStep, setLoadingStep] = useState('')
  // #4 — what-changed diff banner
  const [changeDiff, setChangeDiff] = useState<string | null>(null)
  // #16 — focus / form-panel collapsed
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const CURRENT_VERSION = '0.2.1'

  // Dynamic favicon showing Nq value
  useDynamicFavicon(sizing?.specific_speed_nq || null)

  // Tab history tracking
  useEffect(() => {
    if (tab) setTabHistory(prev => {
      const next = [...prev, tab]
      return next.slice(-20)
    })
  }, [tab])

  const goBack = () => {
    if (tabHistory.length < 2) return
    const prev = tabHistory[tabHistory.length - 2]
    setTabHistory(h => h.slice(0, -1))
    setTab(prev)
  }

  // Action timeline logger
  const logAction = useCallback((action: string, detail?: string) => {
    setTimeline(prev => [...prev, { time: new Date(), action, detail }])
  }, [])

  // Helper to mark a progress step as completed
  const markStep = useCallback((step: string) => {
    setCompletedSteps(prev => prev.includes(step) ? prev : [...prev, step])
  }, [])

  useEffect(() => {
    const saved = localStorage.getItem('hpe_token')
    if (saved) { setToken(saved); setPage('projects') }
  }, [])

  // Track the last seen version (no intrusive banner — see WhatsNew modal).
  useEffect(() => {
    localStorage.setItem('hpe_last_version', CURRENT_VERSION)
  }, [])

  // #18 — Ctrl+P quick print
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'p' && sizing && page === 'design') {
        e.preventDefault()
        window.print()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [sizing, page])

  // Auto-save to localStorage every 30s (feature #10)
  useEffect(() => {
    const timer = setInterval(() => {
      if (sizing) {
        localStorage.setItem('hpe_autosave', JSON.stringify({
          opPoint, tab, projectName: currentProject?.name, timestamp: Date.now(),
        }))
      }
    }, 30000)
    return () => clearInterval(timer)
  }, [sizing, opPoint, tab, currentProject])

  // Check for auto-restore on mount (feature #10)
  useEffect(() => {
    try {
      const raw = localStorage.getItem('hpe_autosave')
      if (raw) {
        const state = JSON.parse(raw)
        if (Date.now() - state.timestamp < 86400000) setShowAutoRestore(true)
      }
    } catch { /* ignore */ }
  }, [])

  // Track progress steps based on state changes
  useEffect(() => {
    if (opPoint.flowRate > 0 && opPoint.head > 0 && opPoint.rpm > 0) markStep('dados')
  }, [opPoint, markStep])

  useEffect(() => {
    if (sizing) markStep('sizing')
  }, [sizing, markStep])

  useEffect(() => {
    if (tab === '3d' || tab === 'meridional-editor' || tab === 'meridional-drag') markStep('geometria')
    if (['curves', 'velocity', 'losses', 'stress', 'pressure', 'multispeed', 'spanwise', 'noise'].includes(tab)) markStep('analise')
    if (['optimize', 'doe', 'pareto'].includes(tab)) markStep('otimizacao')
  }, [tab, markStep])

  // Auto-restore handler (feature #10)
  const handleAutoRestore = useCallback(() => {
    try {
      const raw = localStorage.getItem('hpe_autosave')
      if (raw) {
        const state = JSON.parse(raw)
        if (state.opPoint) setOpPoint(state.opPoint)
        if (state.tab) setTab(state.tab)
        toast('Sessão anterior restaurada', 'success')
      }
    } catch { /* ignore */ }
    setShowAutoRestore(false)
  }, [toast])

  // Duplicate version handler (feature #4)
  const handleDuplicateVersion = useCallback((v: VersionEntry) => {
    const qH = v.flow_rate >= 1 ? v.flow_rate : v.flow_rate * 3600
    setOpPoint({ flowRate: qH, head: v.head, rpm: v.rpm })
    toast('Ponto de operação copiado — clique Executar para criar nova versão', 'info')
  }, [toast])

  // Warning count for sidebar badge (feature #8)
  const warningCount = sizing?.warnings?.length || 0

  const handleLogin = (userData: any, tok: string) => {
    setUser(userData); setToken(tok); setPage('projects')
  }

  const handleLogout = () => {
    localStorage.removeItem('hpe_token')
    setUser(null); setToken(''); setPage('login'); setSizing(null)
  }

  const renameCurrentProject = async (newName: string) => {
    if (!newName.trim() || !currentProject?.id) return
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (token) headers['Authorization'] = `Bearer ${token}`
      await fetch(`/api/v1/projects/${currentProject.id}`, {
        method: 'PUT', headers, body: JSON.stringify({ name: newName.trim() }),
      })
      setCurrentProject({ ...currentProject, name: newName.trim() })
    } catch {}
  }

  const handleNavigate = (p: 'projects' | 'design', t?: Tab) => {
    if (p === 'projects') {
      setPage('projects')
      setSizing(null); setCurves([]); setLosses(null); setStress(null)
    } else {
      setPage('design')
      if (t) setTab(t)
    }
  }

  // templateOp is stored so the useEffect below can pick it up
  const [pendingTemplateOp, setPendingTemplateOp] = useState<{ flowRate: number; head: number; rpm: number } | null>(null)
  // Incremented on every handleSelectProject call so the effect fires even when reopening the same project
  const [projectOpenKey, setProjectOpenKey] = useState(0)

  const handleSelectProject = (project: any, templateOp?: { flowRate: number; head: number; rpm: number; machine_type?: string }) => {
    // Reset all design state synchronously
    setSizing(null); setCurves([]); setLosses(null); setStress(null)
    setVersions([]); setCurrentVersionId(null)
    setOpPoint({ flowRate: 0, head: 0, rpm: 0 })
    setPendingTemplateOp(templateOp && templateOp.flowRate > 0 ? templateOp : null)
    setCurrentProject(project)
    setProjectOpenKey(k => k + 1)   // always triggers the useEffect, even for same project id
    setTab('results')
    setPage('design')
  }

  // Load last version whenever the active project is opened (React-idiomatic)
  useEffect(() => {
    if (page !== 'design' || !currentProject?.id) return

    // Template op → just pre-fill and run, no version to restore
    if (pendingTemplateOp) {
      const { flowRate, head, rpm } = pendingTemplateOp
      setOpPoint({ flowRate, head, rpm })
      setPendingTemplateOp(null)
      setTimeout(() => handleRunSizing(flowRate, head, rpm), 80)
      return
    }

    // Existing project → load versions and restore the latest
    let cancelled = false
    ;(async () => {
      try {
        const vers = await listVersions(currentProject.id, 50)
        if (cancelled) return
        if (vers.length === 0) return

        setVersions(vers)
        const latest = vers[0]   // newest-first from API
        setCurrentVersionId(latest.id)
        const qH = latest.flow_rate >= 1 ? latest.flow_rate : latest.flow_rate * 3600
        setOpPoint({ flowRate: qH, head: latest.head, rpm: latest.rpm })

        try {
          const detail = await getVersion(latest.id)
          if (cancelled) return
          const sr = detail.sizing_result
          if (sr && Object.keys(sr).length > 5) {
            setSizing(sr as any)
            setTab('overview')   // open on Visão Geral after restore
            const qm3s = qH / 3600
            getCurves(qm3s, latest.head, latest.rpm).then(c => { if (!cancelled) setCurves(c.points || []) }).catch(() => {})
            getLossBreakdown(qm3s, latest.head, latest.rpm).then(d => { if (!cancelled) setLosses(d) }).catch(() => {})
          } else {
            // sizing_result incomplete → recalculate without saving a new version
            if (!cancelled) handleRunSizing(qH, latest.head, latest.rpm, true)
          }
        } catch {
          if (!cancelled) handleRunSizing(qH, latest.head, latest.rpm, true)
        }
      } catch { /* project has no versions yet — form stays blank */ }
    })()

    return () => { cancelled = true }   // cleanup if project changes again quickly
  }, [projectOpenKey])                  // projectOpenKey increments on every handleSelectProject call

  const handleSaveDesign = async () => {
    if (!sizing || !currentProject) return
    setSaving(true)
    try {
      const body = {
        sizing_result: sizing as any,
        operating_point: { flow_rate: opPoint.flowRate / 3600, head: opPoint.head, rpm: opPoint.rpm },
        curve_points: curves.map(c => ({
          flow_rate: c.flow_rate, head: c.head, efficiency: c.efficiency,
          power: c.power, npsh_required: c.npsh_required, is_unstable: c.is_unstable ?? false,
        })),
      }
      const r = await fetch(`/api/v1/projects/${currentProject.id}/designs`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      })
      if (r.ok) { const d = await r.json(); setSavedId(d.id); setIsDirty(false); toast('Design salvo no projeto', 'success'); logAction('Design salvo', currentProject?.name) }
      else { toast('Erro ao salvar design', 'error') }
    } catch { toast('Erro ao salvar design', 'error') }
    finally { setSaving(false) }
  }

  // Full sizing run — sequential to avoid BaseHTTPMiddleware concurrency deadlock
  const handleRunSizing = async (q: number, h: number, n: number, skipVersionSave = false) => {
    setLoading(true)
    setChangeDiff(null)
    try {
      const qm3s = q / 3600
      // 1. Main sizing first
      setLoadingStep('Triângulos de velocidade...')
      const result = await runSizing(qm3s, h, n)
      // Save previous sizing for delta indicators
      const prevSizing = sizing
      setPreviousSizing(sizing)
      setSizing(result)
      setOpPoint({ flowRate: q, head: h, rpm: n })
      setSavedId(null)
      setIsDirty(true)

      // #4 — compute what changed vs previous run
      if (prevSizing) {
        const diffs: string[] = []
        const etaDelta = (result.estimated_efficiency - prevSizing.estimated_efficiency) * 100
        if (Math.abs(etaDelta) > 0.1) diffs.push(`${etaDelta >= 0 ? '↑' : '↓'} η ${etaDelta >= 0 ? '+' : ''}${etaDelta.toFixed(1)}%`)
        const npshDelta = result.estimated_npsh_r - prevSizing.estimated_npsh_r
        if (Math.abs(npshDelta) > 0.05) diffs.push(`${npshDelta >= 0 ? '↑' : '↓'} NPSHr ${npshDelta >= 0 ? '+' : ''}${npshDelta.toFixed(1)}m`)
        const powDelta = (result.estimated_power - prevSizing.estimated_power) / 1000
        if (Math.abs(powDelta) > 0.05) diffs.push(`${powDelta >= 0 ? '↑' : '↓'} P ${powDelta >= 0 ? '+' : ''}${powDelta.toFixed(1)}kW`)
        const d2Delta = (result.impeller_d2 - prevSizing.impeller_d2) * 1000
        if (Math.abs(d2Delta) > 0.5) diffs.push(`D2 ${d2Delta >= 0 ? '+' : ''}${d2Delta.toFixed(0)}mm`)
        if (diffs.length > 0) setChangeDiff(diffs.join(' · '))
      }

      // Push to calculation history (max 10, FIFO)
      setHistory(prev => {
        const entry: HistoryEntry = {
          id: Date.now(),
          timestamp: new Date(),
          flowRate: q,
          head: h,
          rpm: n,
          nq: result.specific_speed_nq,
          eta: result.estimated_efficiency,
          d2: result.impeller_d2 * 1000,
        }
        return [entry, ...prev].slice(0, 10)
      })

      // 2. Secondary data — sequential to avoid middleware serialization deadlock
      setLoadingStep('Curva H-Q...')
      const curvesData = await getCurves(qm3s, h, n).catch(() => ({ points: [] }))
      setCurves(curvesData.points || [])

      setLoadingStep('Análise de perdas...')
      const lossData = await getLossBreakdown(qm3s, h, n).catch(() => null)
      setLosses(lossData)

      setLoadingStep('Análise estrutural...')
      const stressData = await runStressAnalysis(qm3s, h, n).catch(() => null)
      setStress(stressData)

      // Auto-save as version (skip when restoring an existing version)
      if (!skipVersionSave) {
        setLoadingStep('Salvando versão...')
        try {
          const ver = await saveVersion(
            { flow_rate: qm3s, head: h, rpm: n },
            result,
            currentProject?.id || undefined,
          )
          setVersions(prev => [ver, ...prev])
          setCurrentVersionId(ver.id)
          setIsDirty(false)
        } catch { /* version save is best-effort */ }
      }

      // Always land on Visão Geral after a sizing run
      setTab('overview')

      toast('Dimensionamento concluido', 'success')
      incrementSizingCount()
      logAction('Dimensionamento executado', `Q=${q} m3/h H=${h}m n=${n}rpm`)

      // Proactive engineering warnings — only actionable, physics-based hints.
      // (Random "encouragement" / tutorial toasts were removed to cut UI noise.)
      setTimeout(() => {
        if (result.estimated_efficiency < 0.75) {
          toast('η abaixo de 75%. Tente aumentar o número de pás ou ajustar β₂.', 'warning')
        }
        if (result.estimated_npsh_r > 8) {
          toast(`NPSHr=${result.estimated_npsh_r.toFixed(1)}m está alto. Reduza RPM para ~${Math.round(n * 0.85)}.`, 'warning')
        }
        if (result.specific_speed_nq < 10) {
          toast('Nq muito baixo — considere multi-estágio ou aumente RPM.', 'info')
        }
        if (result.specific_speed_nq > 200) {
          toast('Nq alto — considere bomba axial ou mixed-flow.', 'info')
        }
      }, 1500)
    } catch {
      toast('Erro ao calcular', 'error')
    } finally {
      setLoading(false)
      setLoadingStep('')
    }
  }

  // Keyboard shortcuts
  const handleRunSizingShortcut = useCallback(() => {
    if (opPoint.flowRate > 0 && opPoint.head > 0 && opPoint.rpm > 0) handleRunSizing(opPoint.flowRate, opPoint.head, opPoint.rpm)
  }, [opPoint])

  useKeyboardShortcuts({
    onRunSizing: handleRunSizingShortcut,
    onSave: handleSaveDesign,
    onCmdPalette: () => setCmdOpen(true),
    onNavigate: handleNavigate,
    onEscape: () => { setCmdOpen(false); setShortcutsHelpOpen(false); setHelpOpen(false); setTimelineOpen(false) },
    onF1Help: () => setHelpOpen(v => !v),
    onExport: () => setExportCenterOpen(true),
  })

  // History restore handler
  const handleHistoryRestore = (entry: HistoryEntry) => {
    setOpPoint({ flowRate: entry.flowRate, head: entry.head, rpm: entry.rpm })
    handleRunSizing(entry.flowRate, entry.head, entry.rpm)
  }

  // Version handlers
  const handleVersionSelect = (v: VersionEntry) => {
    setCurrentVersionId(v.id)
    // flow_rate from backend is in m3/s, convert to m3/h for the form
    const qH = v.flow_rate >= 1 ? v.flow_rate : v.flow_rate * 3600
    setOpPoint({ flowRate: qH, head: v.head, rpm: v.rpm })
    // skipVersionSave=true: restoring an existing version must NOT create a new one
    handleRunSizing(qH, v.head, v.rpm, true)
  }

  const handleVersionCompare = async (a: VersionEntry, b: VersionEntry) => {
    try {
      const result = await compareVersions(a.id, b.id)
      setCompareData(result)
    } catch {
      toast('Erro ao comparar versões', 'error')
    }
  }

  const handleVersionDelete = async (id: string) => {
    try {
      await apiDeleteVersion(id)
      setVersions(prev => prev.filter(v => v.id !== id))
      if (currentVersionId === id) setCurrentVersionId(null)
    } catch {
      toast('Erro ao excluir versão', 'error')
    }
  }

  /* Shared overlay elements rendered in all authenticated pages */
  const overlays = (
    <>
      {ctxMenu && (
        <GlobalContextMenu
          x={ctxMenu.x} y={ctxMenu.y}
          onClose={() => setCtxMenu(null)}
          onRecalculate={() => handleRunSizing(opPoint.flowRate, opPoint.head, opPoint.rpm)}
          onExport={() => setExportCenterOpen(true)}
          onHelp={() => setHelpOpen(v => !v)}
        />
      )}
      <CommandPalette
        open={cmdOpen}
        onClose={() => setCmdOpen(false)}
        onNavigate={handleNavigate}
        onRunSizing={handleRunSizingShortcut}
        onStartTour={() => setTourActive(true)}
        onGlossary={() => setGlossaryOpen(true)}
        sizing={sizing}
      />
      <Toast messages={toasts} onDismiss={dismiss} />
      {showWhatsNew && page !== 'login' && (
        <WhatsNew onClose={() => { setShowWhatsNew(false); localStorage.setItem('hpe_whats_new_seen', '0.2.0') }} />
      )}
      {showAutoRestore && page !== 'login' && (
        <div style={{
          position: 'fixed', bottom: 48, left: '50%', transform: 'translateX(-50%)',
          background: 'var(--bg-elevated)', border: '1px solid var(--border-primary)',
          borderRadius: 8, padding: '10px 16px', zIndex: 2000,
          display: 'flex', alignItems: 'center', gap: 12, fontSize: 13,
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
        }}>
          <span style={{ color: 'var(--text-secondary)' }}>Sessão anterior encontrada. Restaurar?</span>
          <button className="btn-primary" style={{ fontSize: 11, padding: '4px 12px' }} onClick={handleAutoRestore}>Restaurar</button>
          <button onClick={() => setShowAutoRestore(false)} style={{
            background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 13,
          }}>x</button>
        </div>
      )}
      {shortcutsHelpOpen && <ShortcutsHelpModal onClose={() => setShortcutsHelpOpen(false)} />}
      <GuidedTour active={tourActive} onComplete={() => setTourActive(false)} onNavigate={(p, t) => handleNavigate(p, t as Tab)} />
      {compareData && <VersionCompareModal data={compareData} onClose={() => setCompareData(null)} />}
      <FloatingMetrics sizing={sizing} resultsRef={resultsRef} />
      <ContextualHelp open={helpOpen} onClose={() => setHelpOpen(false)} currentTab={tab} />
      <ActionTimeline entries={timeline} open={timelineOpen} onClose={() => setTimelineOpen(false)} />
      <Glossary open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
      <ExportCenter
        open={exportCenterOpen}
        onClose={() => setExportCenterOpen(false)}
        sizing={sizing}
        opPoint={sizing ? opPoint : undefined}
        projectName={currentProject?.name}
        onExport={(format) => {
          setExportCenterOpen(false)
          const q = opPoint.flowRate / 3600
          const h = opPoint.head
          const n = opPoint.rpm
          const downloadBlob = (blob: Blob, filename: string) => {
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url; a.download = filename; a.click()
            URL.revokeObjectURL(url)
          }
          const downloadText = (text: string, filename: string) => {
            downloadBlob(new Blob([text], { type: 'text/plain' }), filename)
          }
          switch (format) {
            case 'step':
              fetch(`/api/v1/geometry/export/step?flow_rate=${q}&head=${h}&rpm=${n}`)
                .then(r => r.blob()).then(b => downloadBlob(b, 'impeller.step'))
                .catch(() => toast('Erro ao exportar STEP', 'error'))
              break
            case 'iges':
              fetch(`/api/v1/geometry/export/iges?flow_rate=${q}&head=${h}&rpm=${n}`)
                .then(r => r.blob()).then(b => downloadBlob(b, 'impeller.iges'))
                .catch(() => toast('Erro ao exportar IGES', 'error'))
              break
            case 'stl':
              fetch(`/api/v1/geometry/export/stl?flow_rate=${q}&head=${h}&rpm=${n}`)
                .then(r => r.blob()).then(b => downloadBlob(b, 'impeller.stl'))
                .catch(() => toast('Erro ao exportar STL', 'error'))
              break
            case 'gltf':
              handleNavigate('design', '3d')
              toast('Use o botao glTF no viewer 3D', 'info')
              break
            case 'bladegen':
              fetch(`/api/v1/geometry/export/bladegen?flow_rate=${q}&head=${h}&rpm=${n}`)
                .then(r => r.json()).then(d => { if (d.inf) downloadText(d.inf, 'blade.inf') })
                .catch(() => toast('Erro ao exportar BladeGen', 'error'))
              break
            case 'geo':
              fetch(`/api/v1/geometry/export/geo?flow_rate=${q}&head=${h}&rpm=${n}`)
                .then(r => r.json()).then(d => { if (d.ps) downloadText(d.ps, 'blade_ps.geo') })
                .catch(() => toast('Erro ao exportar GEO', 'error'))
              break
            case 'cfx-package':
              fetch('/api/v1/cfd/cfx/package', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ flow_rate: q, head: h, rpm: n }) })
                .then(r => r.blob()).then(b => downloadBlob(b, 'cfx_package.zip'))
                .catch(() => toast('Erro ao exportar CFX', 'error'))
              break
            case 'fluent':
              fetch('/api/v1/cfd/fluent/journal', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ flow_rate: q, head: h, rpm: n }) })
                .then(r => r.text()).then(t => downloadText(t, 'setup.jou'))
                .catch(() => toast('Erro ao exportar Fluent', 'error'))
              break
            case 'openfoam':
              fetch('/api/v1/cfd/openfoam/case', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ flow_rate: q, head: h, rpm: n }) })
                .then(r => r.blob()).then(b => downloadBlob(b, 'openfoam_case.zip'))
                .catch(() => toast('Erro ao exportar OpenFOAM', 'error'))
              break
            case 'pdf':
              fetch('/api/v1/report/pdf', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ flow_rate: q, head: h, rpm: n }) })
                .then(r => r.blob()).then(b => downloadBlob(b, 'hpe_relatorio.pdf'))
                .catch(() => toast('Erro ao exportar PDF', 'error'))
              break
            case 'csv':
              if (sizing) {
                import('./services/api').then(({ exportSizingCSV }) => exportSizingCSV(sizing, opPoint))
              }
              break
            case 'png':
              handleNavigate('design', '3d')
              toast('Use o botao PNG no viewer 3D', 'info')
              break
            case 'email':
              if (sizing) emailResults(sizing, opPoint, currentProject?.name)
              break
            default:
              toast(`Formato ${format} em desenvolvimento`, 'info')
          }
          markStep('exportar')
        }}
      />
    </>
  )

  // Recent tabs for SubTabBar dot indicators (last 3 unique, excluding current)
  const recentTabs = [...new Set(tabHistory.slice(-6).reverse())].filter(t => t !== tab).slice(0, 3)

  // === LOGIN ===
  if (page === 'login') {
    return <LoginPage onLogin={handleLogin} />
  }

  // === PROJECTS ===
  if (page === 'projects') {
    return (
      <Layout page="projects" activeTab={null} userName={user?.name || t.user}
        onNavigate={handleNavigate} onLogout={handleLogout} warningCount={warningCount}>
        <ProjectsPage onSelectProject={handleSelectProject} token={token} />
        <StatusBar sizing={sizing} previousSizing={previousSizing} opPoint={sizing ? opPoint : undefined} savedId={savedId} onShortcutsHelp={() => setShortcutsHelpOpen(true)} onTimeline={() => setTimelineOpen(v => !v)} />
        {overlays}
      </Layout>
    )
  }

  const canRun = opPoint.flowRate > 0 && opPoint.head > 0 && opPoint.rpm > 0

  // === DESIGN — 3D viewer (now with sub-tabs visible) ===
  if (tab === '3d') {
    return (
      <Layout page="design" activeTab={tab} userName={user?.name || t.user}
        projectName={currentProject?.name} onNavigate={handleNavigate} onLogout={handleLogout} warningCount={warningCount} recentTabs={recentTabs}>
        <ImpellerViewer
          flowRate={opPoint.flowRate}
          head={opPoint.head}
          rpm={opPoint.rpm}
          fullscreen
          loading={loading}
          sizing={sizing}
          onRunSizing={handleRunSizing}
          onToast={toast}
        />
        <StatusBar sizing={sizing} previousSizing={previousSizing} opPoint={sizing ? opPoint : undefined} savedId={savedId} onShortcutsHelp={() => setShortcutsHelpOpen(true)} onTimeline={() => setTimelineOpen(v => !v)} />
        {overlays}
      </Layout>
    )
  }

  // === DESIGN — wide tabs (editors, optimization, etc.) ===
  if (WIDE_TABS.includes(tab) && (tab as string) !== '3d') {
    return (
      <Layout page="design" activeTab={tab} userName={user?.name || t.user}
        projectName={currentProject?.name} onNavigate={handleNavigate} onLogout={handleLogout} warningCount={warningCount} recentTabs={recentTabs}>
        <div>
          {tab === 'templates' && (
            <TemplateSelector loading={loading} onSelect={(tmpl: any) => {
              if (tmpl.flow_rate && tmpl.head && tmpl.rpm) {
                // Navigate to results tab so user sees the computation
                handleNavigate('design', 'results')
                handleRunSizing(tmpl.flow_rate, tmpl.head, tmpl.rpm)
              }
            }} />
          )}
          {tab === 'meridional-drag' && sizing && (
            <MeridionalDragEditor d1={sizing.impeller_d1 * 1000} d2={sizing.impeller_d2 * 1000} b2={sizing.impeller_b2 * 1000} />
          )}
          {tab === 'meridional-editor' && sizing && (
            <MeridionalEditor d1={sizing.impeller_d1} d2={sizing.impeller_d2} b2={sizing.impeller_b2} />
          )}
          {tab === 'lete' && sizing && (
            <LETEEditor nq={sizing.specific_speed_nq} flowRate={opPoint.flowRate / 3600} head={opPoint.head} rpm={opPoint.rpm} />
          )}
          {tab === 'lean-sweep' && (
            <LeanSweepPanel defaultFlowRate={opPoint.flowRate / 3600} defaultHead={opPoint.head} defaultRpm={opPoint.rpm} />
          )}
          {tab === 'doe' && <DoEPanel />}
          {tab === 'pareto' && (
            <ParetoPanel defaultFlowRate={opPoint.flowRate} defaultHead={opPoint.head} defaultRpm={opPoint.rpm} />
          )}
          {tab === 'batch' && (
            <div className="card" style={{ padding: 20 }}>
              <h3 style={{ color: 'var(--accent)', marginTop: 0, fontSize: 15 }}>Batch / Paramétrico</h3>
              <BatchPanel baseFlowRate={opPoint.flowRate / 3600} baseHead={opPoint.head} baseRpm={opPoint.rpm} />
            </div>
          )}
          {tab === 'compare' && <DesignComparison />}
          {tab === 'optimize' && (
            <>
              <OptimizationPresets defaultFlowRate={opPoint.flowRate} defaultHead={opPoint.head} defaultRpm={opPoint.rpm} />
              <OptimizePanel defaultFlowRate={opPoint.flowRate} defaultHead={opPoint.head} defaultRpm={opPoint.rpm} />
              <FeedbackStars tab="optimize" />
            </>
          )}
          {tab === 'benchmarks' && (
            <div style={{ maxWidth: 900, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: 'var(--text-primary)' }}>Benchmarks de Validação</h3>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>
                  SHF Combes 1999, ERCOFTAC TC6, TUD radial — MAPE contra bancada experimental.
                </p>
              </div>
              <BenchmarksPanel />
            </div>
          )}
          {tab === 'cavitation' && (
            <div style={{ maxWidth: 760, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: 'var(--text-primary)' }}>Análise de Cavitação</h3>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>
                  NPSHr, margem de segurança, curva NPSHr–Q e recomendações (Gülich §6.10).
                </p>
              </div>
              {canRun ? (
                <CavitationPanel
                  flowRate={opPoint.flowRate}
                  head={opPoint.head}
                  rpm={opPoint.rpm}
                  sizing={sizing ? {
                    specific_speed_nq:    sizing.specific_speed_nq,
                    estimated_npsh_r:     sizing.estimated_npsh_r,
                    sigma:                sizing.sigma,
                    impeller_d2:          sizing.impeller_d2,
                    estimated_efficiency: sizing.estimated_efficiency,
                  } : undefined}
                />
              ) : (
                <EmptyStateHint label="Preencha Q, H e n para analisar cavitação." />
              )}
            </div>
          )}
          {tab === 'cfd_sim' && (
            <div style={{ maxWidth: 820, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: 'var(--text-primary)' }}>Simulação CFD</h3>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>
                  Configurar e executar caso OpenFOAM — ponto único ou sweep H-Q completo.
                </p>
              </div>
              {canRun ? (
                <CFDSimPanel
                  flowRate={opPoint.flowRate}
                  head={opPoint.head}
                  rpm={opPoint.rpm}
                  sizing={sizing ? {
                    specific_speed_nq:    sizing.specific_speed_nq,
                    impeller_d2:          sizing.impeller_d2,
                    impeller_b2:          sizing.impeller_b2,
                    beta1:                sizing.beta1,
                    beta2:                sizing.beta2,
                    blade_count:          sizing.blade_count,
                    estimated_efficiency: sizing.estimated_efficiency,
                  } : undefined}
                />
              ) : (
                <EmptyStateHint label="Preencha Q, H e n para configurar simulação CFD." />
              )}
            </div>
          )}
          {tab === 'pipeline' && (
            <div style={{ maxWidth: 720, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
              <div>
                <h3 style={{ margin: '0 0 4px', fontSize: 15, color: 'var(--text-primary)' }}>Pipeline Completo</h3>
                <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>
                  Sizing 1D → Geometria → Surrogate → Orquestrador Celery.
                  Progresso em tempo real via WebSocket.
                </p>
              </div>
              {opPoint.flowRate > 0 && opPoint.head > 0 && opPoint.rpm > 0 ? (
                <PipelinePanel
                  input={{ Q: opPoint.flowRate / 3600, H: opPoint.head, n: opPoint.rpm }}
                  onComplete={(result) => {
                    const eta = result.eta as number | undefined
                    const d2 = result.D2_mm as number | undefined
                    toast(
                      `Pipeline concluído — η=${eta !== undefined ? (eta * 100).toFixed(1) + '%' : '—'}  D2=${d2 !== undefined ? d2.toFixed(0) + ' mm' : '—'}`,
                      'success',
                    )
                    logAction('pipeline_complete', `eta=${eta ?? '?'}`)
                  }}
                />
              ) : (
                <EmptyStateHint label="Preencha Q, H e n no formulário para executar o pipeline." />
              )}
            </div>
          )}
          {!sizing && tab !== 'templates' && tab !== 'pipeline' && tab !== 'cavitation' && tab !== 'cfd_sim' && (
            <EmptyStateHint label="Execute um dimensionamento primeiro para usar esta funcionalidade." />
          )}
        </div>
        <StatusBar sizing={sizing} previousSizing={previousSizing} opPoint={sizing ? opPoint : undefined} savedId={savedId} onShortcutsHelp={() => setShortcutsHelpOpen(true)} onTimeline={() => setTimelineOpen(v => !v)} />
        {overlays}
      </Layout>
    )
  }

  // === DESIGN — standard 2-column layout (sizing form + results) ===
  return (
    <Layout page="design" activeTab={tab} userName={user?.name || t.user}
      projectName={currentProject?.name} onNavigate={handleNavigate} onLogout={handleLogout}
      onRecalculate={sizing ? () => handleRunSizing(opPoint.flowRate, opPoint.head, opPoint.rpm) : undefined}
      onExport={sizing ? () => setExportCenterOpen(true) : undefined}
      onContextMenu={(e: React.MouseEvent) => { e.preventDefault(); setCtxMenu({ x: e.clientX, y: e.clientY }) }}
      sizing={sizing}>

      {/* Progress Stepper — removed: sub-tabs already serve the same purpose */}

      <div className="content-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {/* #16 — Focus mode toggle */}
          <button
            type="button"
            title={sidebarCollapsed ? 'Mostrar painel (⊞)' : 'Modo foco — ocultar painel (⊟)'}
            onClick={() => setSidebarCollapsed(v => !v)}
            style={{ background: 'none', border: '1px solid var(--border-primary)', borderRadius: 4, padding: '3px 7px', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 11, transition: 'all 0.15s', flexShrink: 0 }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-primary)'; e.currentTarget.style.color = 'var(--text-muted)' }}
          >{sidebarCollapsed ? '⊞' : '⊟'}</button>
          <button
            type="button"
            onClick={() => handleNavigate('projects')}
            style={{
              background: 'none', border: 'none', color: 'var(--text-muted)',
              cursor: 'pointer', fontSize: 13, padding: 0,
              fontFamily: 'var(--font-family)', whiteSpace: 'nowrap',
            }}
            onMouseEnter={e => { (e.currentTarget as HTMLElement).style.color = 'var(--accent)' }}
            onMouseLeave={e => { (e.currentTarget as HTMLElement).style.color = 'var(--text-muted)' }}
          >
            &larr; Projetos
          </button>
          <span style={{ color: 'var(--border-primary)', fontSize: 14, userSelect: 'none' }}>/</span>
          {editingProjectName ? (
            <input
              autoFocus
              value={editProjectNameVal}
              onChange={e => setEditProjectNameVal(e.target.value)}
              onBlur={() => { renameCurrentProject(editProjectNameVal); setEditingProjectName(false) }}
              onKeyDown={e => { if (e.key === 'Enter') { renameCurrentProject(editProjectNameVal); setEditingProjectName(false) } if (e.key === 'Escape') setEditingProjectName(false) }}
              className="input"
              style={{ fontSize: 18, fontWeight: 600, padding: '2px 8px', minWidth: 200 }}
            />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <h1
                style={{ fontSize: 18, margin: 0, cursor: currentProject ? 'text' : 'default' }}
                onDoubleClick={() => { if (currentProject) { setEditingProjectName(true); setEditProjectNameVal(currentProject.name) } }}
                title={currentProject ? 'Duplo-clique para renomear' : undefined}
              >{currentProject?.name || t.quickDesign}</h1>
              {/* #1 — unsaved dot */}
              {isDirty && (
                <span title="Alterações não salvas — Ctrl+S para salvar"
                  style={{ width: 8, height: 8, borderRadius: '50%', background: '#f59e0b', display: 'inline-block', flexShrink: 0, boxShadow: '0 0 6px #f59e0b80' }}
                />
              )}
              {/* #14 — pencil icon */}
              {currentProject && (
                <button
                  type="button"
                  title="Renomear projeto"
                  onClick={() => { setEditingProjectName(true); setEditProjectNameVal(currentProject.name) }}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '2px 3px', color: 'var(--text-muted)', opacity: 0.45, lineHeight: 1, display: 'flex', alignItems: 'center' }}
                  onMouseEnter={e => { e.currentTarget.style.opacity = '1'; e.currentTarget.style.color = 'var(--accent)' }}
                  onMouseLeave={e => { e.currentTarget.style.opacity = '0.45'; e.currentTarget.style.color = 'var(--text-muted)' }}
                >
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                  </svg>
                </button>
              )}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {sizing && <DesignQualityBadge sizing={sizing} />}
          <VersionPanel
            versions={versions}
            currentVersionId={currentVersionId}
            onSelect={handleVersionSelect}
            onCompare={handleVersionCompare}
            onDelete={handleVersionDelete}
            onDuplicate={handleDuplicateVersion}
          />
          {/* Inline progress next to Exportar */}
          {sizing && (
            <ProjectChecklist
              hasSizing={completedSteps.includes('sizing')}
              hasViewedGeometry={completedSteps.includes('geometria')}
              hasViewedAnalysis={completedSteps.includes('analise')}
              hasCheckedNpsh={completedSteps.includes('sizing')}
              hasExported={completedSteps.includes('exportar')}
            />
          )}
          <button
            type="button"
            onClick={() => setExportCenterOpen(true)}
            style={{
              padding: '4px 10px', fontSize: 11, borderRadius: 4,
              border: '1px solid var(--border-primary)', background: 'transparent',
              color: 'var(--text-muted)', cursor: 'pointer',
            }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)'; e.currentTarget.style.color = 'var(--accent)' }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-primary)'; e.currentTarget.style.color = 'var(--text-muted)' }}
          >
            Exportar
          </button>
        </div>
      </div>

      {/* Two-column design layout: left = form + export, right = results + analysis tabs */}
      <div style={{ display: 'grid', gridTemplateColumns: sidebarCollapsed ? '0 1fr' : '320px 1fr', gap: sidebarCollapsed ? 0 : 24, transition: 'grid-template-columns 0.25s ease' }}>

        {/* LEFT PANEL — collapsible via focus mode */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, overflow: sidebarCollapsed ? 'hidden' : 'visible', opacity: sidebarCollapsed ? 0 : 1, transition: 'opacity 0.2s', minWidth: 0 }}>
          <SizingForm
            onResult={async (result, curvePoints, lossData, stressData, op) => {
              setPreviousSizing(sizing)
              setSizing(result); setCurves(curvePoints)
              setLosses(lossData); setStress(stressData)
              if (op) setOpPoint(op)
              setSavedId(null)

              // Auto-save version on every calculation
              try {
                const qm3s = op ? op.flowRate / 3600 : opPoint.flowRate / 3600
                const h = op ? op.head : opPoint.head
                const n = op ? op.rpm : opPoint.rpm
                const ver = await saveVersion(
                  { flow_rate: qm3s, head: h, rpm: n },
                  result,
                  currentProject?.id || undefined,
                )
                setVersions(prev => [ver, ...prev])
                setCurrentVersionId(ver.id)
              } catch {}

              toast('Dimensionamento concluído', 'success')
              incrementSizingCount()
            }}
            loading={loading}
            setLoading={setLoading}
            extFlowRate={opPoint.flowRate}
            extHead={opPoint.head}
            extRpm={opPoint.rpm}
          />
        </div>

        {/* RIGHT PANEL — results area */}
        <div ref={resultsRef} style={{ position: 'relative' }}>

          {/* #2 — granular loading step banner */}
          {loading && (
            <div style={{ marginBottom: 12, padding: '10px 14px', borderRadius: 8, background: 'rgba(0,160,223,0.08)', border: '1px solid rgba(0,160,223,0.2)', display: 'flex', alignItems: 'center', gap: 10 }}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" style={{ animation: 'spin 1s linear infinite', flexShrink: 0 }}>
                <path d="M21 12a9 9 0 11-6.219-8.56" />
              </svg>
              <span style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 500 }}>{loadingStep || 'Calculando...'}</span>
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
                {['Triângulos de velocidade...','Curva H-Q...','Análise de perdas...','Análise estrutural...','Salvando versão...'].map((step, i) => {
                  const steps = ['Triângulos de velocidade...','Curva H-Q...','Análise de perdas...','Análise estrutural...','Salvando versão...']
                  const currentIdx = steps.indexOf(loadingStep)
                  return <div key={i} style={{ width: 6, height: 6, borderRadius: '50%', background: i <= currentIdx ? 'var(--accent)' : 'var(--border-primary)', transition: 'background 0.3s' }} />
                })}
              </div>
            </div>
          )}

          {/* #4 — "o que mudou" diff banner */}
          {changeDiff && !loading && (
            <div style={{ marginBottom: 10, padding: '8px 14px', borderRadius: 8, background: 'rgba(34,197,94,0.07)', border: '1px solid rgba(34,197,94,0.25)', display: 'flex', alignItems: 'center', gap: 8 }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2.5" strokeLinecap="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17" /><polyline points="16 7 22 7 22 13" /></svg>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)', flex: 1 }}>{changeDiff}</span>
              <button onClick={() => setChangeDiff(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', fontSize: 16, padding: '0 2px', lineHeight: 1 }}>×</button>
            </div>
          )}

          {/* Section guide (#2) */}
          <SectionGuide tab={tab} />

          {/* Quick Summary tab (#9) */}
          {tab === 'overview' && sizing && (
            <QuickSummary sizing={sizing} curves={curves} opPoint={opPoint} onNavigate={(t) => handleNavigate('design', t as Tab)} />
          )}

          {sizing ? (
            <>
              {/* Results: 2-column layout — left = metrics+detail, right = 3D+health sticky */}
              {tab === 'results' && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 310px', gap: 16, alignItems: 'start' }}>

                  {/* ── Left column ── */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <DesignDashboard
                      sizing={sizing}
                      previousSizing={previousSizing}
                      opPoint={opPoint}
                      onNavigate={(t) => handleNavigate('design', t)}
                      onRunSizing={handleRunSizing}
                      onWhatIf={(newD2mm) => {
                        toast(`D2 alterado para ${newD2mm.toFixed(0)}mm — execute novamente para aplicar`, 'info')
                      }}
                    />
                    <ResultsView sizing={sizing} previousSizing={previousSizing} />
                    <ReferencePanel sizing={sizing} />
                  </div>

                  {/* ── Right column: sticky visual panel ── */}
                  <div style={{ position: 'sticky', top: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
                    <ImpellerMiniPreview
                      flowRate={opPoint.flowRate}
                      head={opPoint.head}
                      rpm={opPoint.rpm}
                      onExpand={() => handleNavigate('design', '3d')}
                    />
                    {/* Gauge + Status + Radar condensed */}
                    {(() => {
                      const eta = (sizing.estimated_efficiency * 100)
                      const ec = eta >= 80 ? '#2563eb' : eta >= 70 ? '#d97706' : '#dc2626'
                      const dr = (sizing as any).diffusion_ratio || 0
                      const u2 = sizing.velocity_triangles?.outlet?.u || 0
                      const r = 32, cx = 42, cy = 42
                      const sa = -210, ea = 30, arc = ea - sa
                      const toRad = (d: number) => d * Math.PI / 180
                      const ax = (a: number) => cx + r * Math.cos(toRad(a))
                      const ay = (a: number) => cy + r * Math.sin(toRad(a))
                      const dArc = (s: number, e: number) => {
                        const lg = e - s > 180 ? 1 : 0
                        return `M ${ax(s)} ${ay(s)} A ${r} ${r} 0 ${lg} 1 ${ax(e)} ${ay(e)}`
                      }
                      const angle = sa + arc * Math.min(1, eta / 100)
                      return (
                        <div className="card" style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
                          {/* Gauge row */}
                          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                            <svg width={84} height={64} viewBox="0 0 84 64">
                              <path d={dArc(sa, ea)} fill="none" stroke="var(--border-primary)" strokeWidth={6} strokeLinecap="round" />
                              <path d={dArc(sa, angle)} fill="none" stroke={ec} strokeWidth={6} strokeLinecap="round" />
                              <text x={cx} y={cx - 4} textAnchor="middle" fill={ec} fontSize={15} fontWeight={700}>{eta.toFixed(1)}</text>
                              <text x={cx} y={cx + 9} textAnchor="middle" fill="var(--text-muted)" fontSize={8}>η %</text>
                            </svg>
                            <div style={{ flex: 1 }}>
                              <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>Status</div>
                              {[
                                { label: 'De Haller', val: dr > 0 ? dr.toFixed(3) : '—', ok: dr >= 0.7, warn: dr >= 0.6 },
                                { label: 'u₂', val: `${u2.toFixed(1)} m/s`, ok: u2 < 35, warn: u2 < 45 },
                                { label: 'NPSHr', val: `${sizing.estimated_npsh_r.toFixed(1)} m`, ok: sizing.estimated_npsh_r < 5, warn: sizing.estimated_npsh_r < 10 },
                              ].map(s => (
                                <div key={s.label} style={{ display: 'flex', alignItems: 'center', fontSize: 11, marginBottom: 4 }}>
                                  <span style={{ color: s.ok ? '#22c55e' : s.warn ? '#f59e0b' : '#ef4444', marginRight: 5, fontSize: 10, width: 10, textAlign: 'center' }}>
                                    {s.ok ? '✓' : s.warn ? '⚠' : '✕'}
                                  </span>
                                  <span style={{ color: 'var(--text-muted)', flex: 1 }}>{s.label}</span>
                                  <span style={{ fontWeight: 700, color: 'var(--text-primary)' }}>{s.val}</span>
                                </div>
                              ))}
                            </div>
                          </div>
                          {/* Radar */}
                          <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
                            <div style={{ fontSize: 9, color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Radar</div>
                            <RadarChart data={[
                              { label: 'η', value: sizing.estimated_efficiency, min: 0.5, max: 0.95, higherBetter: true },
                              { label: 'NP', value: sizing.estimated_npsh_r, min: 0, max: 15, higherBetter: false },
                              { label: 'Pot', value: sizing.estimated_power / 1000, min: 0, max: 50, higherBetter: false },
                              { label: 'D2', value: sizing.impeller_d2 * 1000, min: 100, max: 500, higherBetter: false },
                              { label: 'Hal', value: dr || 0.75, min: 0.5, max: 1.0, higherBetter: true },
                            ]} size={100} />
                          </div>
                        </div>
                      )
                    })()}
                  </div>

                </div>
              )}
              {tab === 'curves' && (
                <>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, alignItems: 'start' }}>
                    <CurvesChart points={curves} designFlow={opPoint.flowRate / 3600} designHead={opPoint.head} />
                    <EfficiencyMap flowRate={opPoint.flowRate} head={opPoint.head} rpm={opPoint.rpm} />
                  </div>
                  <FeedbackStars tab="curves" />
                </>
              )}
              {tab === 'velocity' && <VelocityTriangle triangles={sizing.velocity_triangles} />}
              {tab === 'losses' && <LossBreakdownChart losses={losses} />}
              {tab === 'stress' && <StressView stress={stress} />}
              {/* compare and optimize are now in wide-tab layout */}
              {tab === 'assistant' && <AssistantChat sizing={sizing} />}
              {tab === 'loading' && <LoadingEditor />}
              {tab === 'pressure' && <PressureDistribution sizing={sizing} />}
              {tab === 'multispeed' && <MultiSpeedChart flowRate={opPoint.flowRate} head={opPoint.head} rpm={opPoint.rpm} />}
              {/* meridional-editor is now in wide-tab layout */}
              {tab === 'spanwise' && <SpanwiseLoadingChart sizing={sizing} />}
              {/* doe, pareto, lean-sweep, lete, meridional-drag, batch are now in wide-tab layout */}
              {tab === 'noise' && (
                <div className="card" style={{ padding: 20 }}>
                  <h3 style={{ color: 'var(--accent)', marginTop: 0, fontSize: 15 }}>Predição de Ruído</h3>
                  <NoisePanel flowRate={opPoint.flowRate / 3600} head={opPoint.head} rpm={opPoint.rpm} sizing={sizing} />
                </div>
              )}
            </>
          ) : loading ? (
            /* Loading skeleton — shown while sizing is computing */
            <ResultsSkeleton />
          ) : (
            /* #8 — Empty state with template suggestion cards */
            <div style={{ padding: '32px 0' }}>
              <div style={{ textAlign: 'center', marginBottom: 24 }}>
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5" style={{ opacity: 0.4, marginBottom: 10 }}>
                  <path d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <div style={{ fontSize: 15, color: 'var(--text-secondary)', marginBottom: 4, fontWeight: 600 }}>Pronto para calcular</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Preencha o formulário à esquerda ou inicie com um exemplo:</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
                {[
                  { label: 'Irrigação', desc: 'Q=120 · H=18m · n=1150', q: 120, h: 18, n: 1150, color: '#22c55e' },
                  { label: 'Industrial', desc: 'Q=300 · H=32m · n=1750', q: 300, h: 32, n: 1750, color: '#00a0df' },
                  { label: 'Alta Pressão', desc: 'Q=50 · H=80m · n=2900', q: 50, h: 80, n: 2900, color: '#a855f7' },
                ].map(tmpl => (
                  <button key={tmpl.label} type="button"
                    onClick={() => handleRunSizing(tmpl.q, tmpl.h, tmpl.n)}
                    style={{
                      padding: '14px 12px', borderRadius: 9, cursor: 'pointer', textAlign: 'left',
                      background: 'var(--card-bg)', border: `1px solid ${tmpl.color}30`,
                      borderTop: `2px solid ${tmpl.color}`, fontFamily: 'var(--font-family)',
                      transition: 'all 0.18s',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.transform = 'translateY(-2px)'; e.currentTarget.style.boxShadow = `0 6px 18px ${tmpl.color}22` }}
                    onMouseLeave={e => { e.currentTarget.style.transform = 'none'; e.currentTarget.style.boxShadow = 'none' }}
                  >
                    <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 3 }}>{tmpl.label}</div>
                    <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 8 }}>{tmpl.desc}</div>
                    <div style={{ fontSize: 11, color: tmpl.color, fontWeight: 600 }}>Calcular →</div>
                  </button>
                ))}
              </div>
            </div>
          )}
          <NextStepBanner currentTab={tab} hasSizing={!!sizing} onNavigate={(t) => handleNavigate('design', t)} />

          {/* #7 — Floating recalculate button */}
          {sizing && (
            <div style={{ position: 'sticky', bottom: 12, display: 'flex', justifyContent: 'flex-end', pointerEvents: 'none', marginTop: 16 }}>
              <button
                onClick={() => handleRunSizing(opPoint.flowRate, opPoint.head, opPoint.rpm)}
                disabled={loading || !opPoint.flowRate}
                title="Recalcular (F5)"
                style={{
                  pointerEvents: 'all', display: 'flex', alignItems: 'center', gap: 7,
                  padding: '9px 18px', borderRadius: 24, fontSize: 12, fontWeight: 700,
                  background: 'var(--accent)', color: '#fff', border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
                  boxShadow: '0 4px 20px rgba(0,160,223,0.45)',
                  opacity: loading || !opPoint.flowRate ? 0.6 : 1,
                  transition: 'all 0.18s', fontFamily: 'var(--font-family)',
                }}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="1 4 1 10 7 10" /><path d="M3.51 15a9 9 0 1 0 .49-4" />
                </svg>
                {loading ? (loadingStep || 'Calculando…') : '↺ Recalcular'}
              </button>
            </div>
          )}
        </div>
      </div>
      <StatusBar sizing={sizing} previousSizing={previousSizing} opPoint={sizing ? opPoint : undefined} savedId={savedId} onShortcutsHelp={() => setShortcutsHelpOpen(true)} onTimeline={() => setTimelineOpen(v => !v)} />
      {overlays}
    </Layout>
  )
}

/* ── Shortcuts Help Modal ─────────────────────────────────────────────────── */
function ShortcutsHelpModal({ onClose }: { onClose: () => void }) {
  const shortcuts = [
    { keys: 'Ctrl+K', desc: 'Abrir command palette' },
    { keys: 'Ctrl+S', desc: 'Salvar design' },
    { keys: 'F5 / Ctrl+Enter', desc: 'Executar dimensionamento' },
    { keys: '1-7', desc: 'Navegar entre seções' },
    { keys: 'Ctrl+E', desc: 'Abrir Export Center' },
    { keys: 'F1', desc: 'Ajuda contextual' },
    { keys: 'Esc', desc: 'Fechar modal/palette' },
  ]

  return (
    <div
      style={{
        position: 'fixed', inset: 0, zIndex: 2100,
        background: 'rgba(0,0,0,0.6)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          width: 360, background: 'var(--bg-elevated)',
          borderRadius: 12, border: '1px solid var(--border-primary)',
          boxShadow: 'var(--shadow-md)', padding: 24,
        }}
        onClick={e => e.stopPropagation()}
      >
        <h3 style={{ margin: '0 0 16px', fontSize: 16, color: 'var(--text-primary)' }}>Atalhos de Teclado</h3>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {shortcuts.map(s => (
            <div key={s.keys} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>{s.desc}</span>
              <kbd style={{
                fontSize: 11, padding: '2px 8px',
                background: 'var(--bg-surface)', borderRadius: 4,
                border: '1px solid var(--border-primary)',
                color: 'var(--text-muted)', fontFamily: 'var(--font-family)',
              }}>{s.keys}</kbd>
            </div>
          ))}
        </div>
        <button
          onClick={onClose}
          style={{
            marginTop: 20, width: '100%', padding: '8px 0',
            background: 'var(--bg-surface)', border: '1px solid var(--border-primary)',
            borderRadius: 6, color: 'var(--text-secondary)', cursor: 'pointer',
            fontSize: 13, fontFamily: 'var(--font-family)',
          }}
        >
          Fechar (Esc)
        </button>
      </div>
    </div>
  )
}
