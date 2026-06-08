import React, { useEffect, useState } from 'react'
import {
  listGeometryBackends, generateGeometryMode, type GeometryBackendInfo,
} from '../services/api'

interface Props {
  flowRate: number   // m³/h
  head: number       // m
  rpm: number
}

/**
 * Geometry mode selector — Classic (parametric B-rep) vs Free (implicit SDF).
 * Calls POST /api/v1/geometry/generate with the chosen mode and renders the
 * resulting artifact (params for classic, voxel stats for free).
 */
export default function GeometryModePanel({ flowRate, head, rpm }: Props) {
  const [backends, setBackends] = useState<GeometryBackendInfo[]>([])
  const [mode, setMode] = useState<'classic' | 'free'>('classic')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    listGeometryBackends().then(setBackends).catch(() => {})
  }, [])

  const canRun = flowRate > 0 && head > 0 && rpm > 0

  const run = async () => {
    setLoading(true); setError(null); setResult(null)
    try {
      const r = await generateGeometryMode(flowRate / 3600, head, rpm, mode)
      setResult(r)
    } catch (e: any) {
      setError(e?.message || 'Erro ao gerar geometria')
    } finally {
      setLoading(false)
    }
  }

  const caps = backends.find(b => b.mode === mode)?.capabilities

  return (
    <div style={{ maxWidth: 820, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div>
        <h3 style={{ margin: '0 0 4px', fontSize: 15, color: 'var(--text-primary)' }}>
          Modo de Geometria
        </h3>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--text-muted)' }}>
          Clássico = geometria paramétrica B-rep (CadQuery) para produção.
          Livre = geometria implícita SDF/voxel (estilo Noyron) para manufatura aditiva.
        </p>
      </div>

      {/* Mode selector */}
      <div style={{ display: 'flex', gap: 10 }}>
        {([
          { key: 'classic', label: 'Clássico', desc: 'Paramétrico · STEP/IGES · malha body-fitted' },
          { key: 'free', label: 'Livre', desc: 'Implícito SDF · STL/voxel · malha cut-cell', badge: 'experimental' },
        ] as const).map(opt => {
          const active = mode === opt.key
          return (
            <button
              key={opt.key}
              onClick={() => { setMode(opt.key); setResult(null); setError(null) }}
              style={{
                flex: 1, textAlign: 'left', padding: '12px 14px', borderRadius: 8, cursor: 'pointer',
                border: `1px solid ${active ? 'var(--accent)' : 'var(--border-primary)'}`,
                background: active ? 'rgba(0,160,223,0.10)' : 'transparent',
                fontFamily: 'var(--font-family)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <span style={{ fontWeight: 700, fontSize: 13, color: active ? 'var(--accent)' : 'var(--text-primary)' }}>
                  {opt.label}
                </span>
                {'badge' in opt && opt.badge && (
                  <span style={{
                    fontSize: 9, fontWeight: 700, padding: '1px 6px', borderRadius: 8,
                    background: '#f59e0b22', color: '#f59e0b', border: '1px solid #f59e0b55',
                  }}>{opt.badge}</span>
                )}
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{opt.desc}</div>
            </button>
          )
        })}
      </div>

      {/* Capabilities */}
      {caps && (
        <div style={{ fontSize: 11, color: 'var(--text-muted)', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
          <span>Malha: <b style={{ color: 'var(--text-secondary)' }}>{caps.mesh_strategy}</b></span>
          <span>STEP: {caps.supports_step ? '✓' : '—'}</span>
          <span>STL: {caps.supports_stl ? '✓' : '—'}</span>
          <span>Voxel: {caps.supports_voxel ? '✓' : '—'}</span>
          <span>Canais internos: {caps.supports_internal_channels ? '✓' : '—'}</span>
        </div>
      )}

      <div>
        <button className="btn-primary" onClick={run} disabled={loading || !canRun}
          style={{ fontSize: 13, padding: '8px 16px' }}>
          {loading ? 'Gerando...' : 'Gerar geometria'}
        </button>
        {!canRun && (
          <span style={{ marginLeft: 10, fontSize: 12, color: 'var(--text-muted)' }}>
            Preencha Q, H e n primeiro.
          </span>
        )}
      </div>

      {error && (
        <div style={{
          padding: 12, borderRadius: 8, fontSize: 13,
          background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.3)', color: '#ef4444',
        }}>{error}</div>
      )}

      {result && <ResultView result={result} />}
    </div>
  )
}

function ResultView({ result }: { result: any }) {
  const art = result.artifact || {}
  const isFree = result.mode === 'free'
  const params: Record<string, any> = art.params || {}
  const voxel: Record<string, any> = art.extra?.voxel || {}

  return (
    <div className="card" style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
        <span style={{ color: 'var(--text-muted)' }}>Backend:</span>
        <b style={{ color: 'var(--accent)' }}>{result.backend}</b>
        <span style={{ color: 'var(--text-muted)' }}>· malha {result.mesh_strategy}</span>
      </div>

      {/* Params table */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
        {Object.entries(isFree ? params : params).map(([k, v]) => (
          <div key={k} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', padding: '3px 0' }}>
            <span style={{ color: 'var(--text-muted)' }}>{k}</span>
            <b style={{ color: 'var(--text-primary)' }}>{typeof v === 'number' ? v.toFixed(2) : String(v)}</b>
          </div>
        ))}
      </div>

      {/* Free-mode voxel stats */}
      {isFree && Object.keys(voxel).length > 0 && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          <div style={{ fontWeight: 600, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', fontSize: 10, letterSpacing: '0.05em' }}>
            Campo voxel (SDF)
          </div>
          <div>Grade: <b>{(voxel.grid_shape || []).join(' × ')}</b></div>
          <div>Voxels internos: <b>{voxel.inside_voxels}</b></div>
          <div>Volume sólido: <b>{voxel.solid_volume_cm3} cm³</b></div>
        </div>
      )}

      {/* Warnings */}
      {(art.warnings || []).length > 0 && (
        <div style={{ fontSize: 11, color: '#f59e0b' }}>
          {art.warnings.map((w: string, i: number) => <div key={i}>⚠ {w}</div>)}
        </div>
      )}

      {/* Feasibility */}
      {result.summary && (
        <div style={{ fontSize: 12, color: result.summary.feasible ? '#22c55e' : '#f59e0b' }}>
          {result.summary.feasible ? '✓ Viável' : `⚠ Restrições: ${(result.summary.failed_constraints || []).join(', ') || '—'}`}
        </div>
      )}
    </div>
  )
}
