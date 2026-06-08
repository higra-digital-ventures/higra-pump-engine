/**
 * AnsysSceneViewer — visualização 3D equivalente Ansys CFX-Post.
 *
 * Renderiza a cena retornada por /api/v1/cfd/advanced/ansys_scene em
 * Three.js: surfaces (volute + impeller + hub) com pressure colormap +
 * 3D streamlines coloridas por |U|.
 *
 * Usa @react-three/fiber se disponível; fallback SVG 2D quando R3F não
 * está instalado.
 */
import React, { useState, useEffect, useMemo, useRef } from 'react'

interface SurfaceMesh {
  name: string
  n_vertices: number
  n_triangles: number
  vertices: number[]
  indices: number[]
  field_name: string
  field_values: number[]
  field_min: number
  field_max: number
}

interface AnsysScene {
  surfaces: SurfaceMesh[]
  streamlines: { points: number[]; velocities: number[]; n_points: number; vel_max: number; vel_min: number }[]
  bounding_box: { min: number[]; max: number[] }
  field_global_min: number
  field_global_max: number
  field_name: string
  units: string
  n_streamlines: number
}

interface Props {
  flowRate?: number      // m³/s
  head?: number          // m
  rpm?: number
  field?: 'pressure' | 'velocity'
  height?: number
}

// Try to load R3F at runtime. `require` only exists in CommonJS contexts; in
// the browser/ESM build it is undefined, so this safely falls back to canvas2d.
let R3F: any = null
let Three: any = null
try {
  const _require = (globalThis as any).require as undefined | ((m: string) => any)
  if (_require) {
    R3F = _require('@react-three/fiber')
    Three = _require('three')
  }
} catch { /* fallback to canvas2d */ }

// ===========================================================================

export default function AnsysSceneViewer({
  flowRate = 0.05, head = 30, rpm = 1750, field = 'pressure', height = 540,
}: Props) {
  const [scene, setScene] = useState<AnsysScene | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showStreamlines, setShowStreamlines] = useState(true)
  const [showSurfaces, setShowSurfaces] = useState(true)
  const [colormap, setColormap] = useState<'viridis' | 'jet'>('jet')

  useEffect(() => {
    if (flowRate <= 0 || head <= 0 || rpm <= 0) return
    setLoading(true)
    setError(null)
    fetch('/api/v1/cfd/advanced/ansys_scene', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        flow_rate: flowRate, head, rpm, field,
        n_streamlines: 150, n_streamline_steps: 70,
      }),
    })
      .then(r => r.ok ? r.json() : Promise.reject(`HTTP ${r.status}`))
      .then(data => setScene(data))
      .catch(err => setError(String(err)))
      .finally(() => setLoading(false))
  }, [flowRate, head, rpm, field])

  return (
    <div style={{
      border: '1px solid var(--border-primary)', borderRadius: 8,
      padding: 16, background: 'var(--card-bg)',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12,
      }}>
        <h4 style={{ margin: 0, fontSize: 14, fontWeight: 600, color: 'var(--text-primary)' }}>
          Ansys CFX-Post Equivalent — Surface Pressure + Streamlines
        </h4>
        <div style={{ display: 'flex', gap: 12, fontSize: 11 }}>
          <label style={checkLabel}>
            <input type="checkbox" checked={showSurfaces}
                   onChange={e => setShowSurfaces(e.target.checked)} />
            Surfaces
          </label>
          <label style={checkLabel}>
            <input type="checkbox" checked={showStreamlines}
                   onChange={e => setShowStreamlines(e.target.checked)} />
            Streamlines
          </label>
          <select value={colormap} onChange={e => setColormap(e.target.value as any)}
                  style={{ fontSize: 11, padding: '2px 4px' }}>
            <option value="jet">Jet</option>
            <option value="viridis">Viridis</option>
          </select>
        </div>
      </div>

      {loading && (
        <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center',
                      color: 'var(--text-muted)' }}>
          Carregando cena CFD…
        </div>
      )}

      {error && (
        <div style={{ color: '#ef4444', fontSize: 12 }}>Erro: {error}</div>
      )}

      {scene && !loading && (
        <SceneRenderer scene={scene} height={height}
                       showSurfaces={showSurfaces}
                       showStreamlines={showStreamlines}
                       colormap={colormap} />
      )}

      {scene && (
        <div style={{
          display: 'flex', justifyContent: 'space-between',
          fontSize: 11, color: 'var(--text-muted)', marginTop: 8,
        }}>
          <span>{scene.surfaces.length} surfaces · {scene.n_streamlines} streamlines</span>
          <span>{scene.field_name}: {scene.field_global_min.toFixed(0)} … {scene.field_global_max.toFixed(0)} {scene.units}</span>
        </div>
      )}
    </div>
  )
}

// ===========================================================================
// Renderer (R3F if available, else canvas2d projection)
// ===========================================================================

function SceneRenderer(props: {
  scene: AnsysScene
  height: number
  showSurfaces: boolean
  showStreamlines: boolean
  colormap: 'viridis' | 'jet'
}) {
  // Always use canvas2d projection (R3F may not be installed)
  return <Canvas2DRenderer {...props} />
}

// ---------------------------------------------------------------------------
// Canvas 2D fallback — projeção isométrica básica
// ---------------------------------------------------------------------------

function Canvas2DRenderer({ scene, height, showSurfaces, showStreamlines, colormap }: {
  scene: AnsysScene
  height: number
  showSurfaces: boolean
  showStreamlines: boolean
  colormap: 'viridis' | 'jet'
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [rotation, setRotation] = useState({ x: -0.5, y: 0.4 })
  const [zoom, setZoom] = useState(1.0)
  const dragRef = useRef<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const W = canvas.width
    const H = canvas.height
    ctx.clearRect(0, 0, W, H)

    // Linear gradient background (Ansys-like)
    const grad = ctx.createLinearGradient(0, 0, 0, H)
    grad.addColorStop(0, '#cfe8ff')
    grad.addColorStop(1, '#fbfdff')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, H)

    const bb = scene.bounding_box
    const sx = (bb.max[0] - bb.min[0]) / 2 || 1
    const sy = (bb.max[1] - bb.min[1]) / 2 || 1
    const sz = (bb.max[2] - bb.min[2]) / 2 || 1
    const scale = Math.min(W, H) / 2.4 / Math.max(sx, sy, sz) * zoom

    // 3D → 2D rotated projection
    const cx = Math.cos(rotation.x), sxR = Math.sin(rotation.x)
    const cy = Math.cos(rotation.y), syR = Math.sin(rotation.y)

    function project(x: number, y: number, z: number): [number, number, number] {
      // Rotate around Y then X (camera orbit)
      const x1 = x * cy + z * syR
      const z1 = -x * syR + z * cy
      const y1 = y * cx - z1 * sxR
      const z2 = y * sxR + z1 * cx

      const px = W / 2 + x1 * scale
      const py = H / 2 - y1 * scale
      return [px, py, z2]
    }

    const fmin = scene.field_global_min
    const fmax = scene.field_global_max

    function colorOf(v: number): string {
      const t = (v - fmin) / (fmax - fmin || 1)
      return colormap === 'jet' ? jetColor(t) : viridisColor(t)
    }

    // ── Draw surfaces ────────────────────────────────────────────────
    if (showSurfaces) {
      for (const surf of scene.surfaces) {
        const triangles: { z: number; pts: [number, number][]; color: string }[] = []

        for (let t = 0; t < surf.indices.length; t += 3) {
          const i0 = surf.indices[t]
          const i1 = surf.indices[t + 1]
          const i2 = surf.indices[t + 2]
          const v0 = project(surf.vertices[i0 * 3], surf.vertices[i0 * 3 + 1], surf.vertices[i0 * 3 + 2])
          const v1 = project(surf.vertices[i1 * 3], surf.vertices[i1 * 3 + 1], surf.vertices[i1 * 3 + 2])
          const v2 = project(surf.vertices[i2 * 3], surf.vertices[i2 * 3 + 1], surf.vertices[i2 * 3 + 2])

          const avgZ = (v0[2] + v1[2] + v2[2]) / 3
          const fAvg = (surf.field_values[i0] + surf.field_values[i1] + surf.field_values[i2]) / 3
          triangles.push({
            z: avgZ,
            pts: [[v0[0], v0[1]], [v1[0], v1[1]], [v2[0], v2[1]]],
            color: colorOf(fAvg),
          })
        }

        // Painter's algorithm: sort back to front
        triangles.sort((a, b) => a.z - b.z)
        for (const tri of triangles) {
          ctx.fillStyle = tri.color
          ctx.beginPath()
          ctx.moveTo(tri.pts[0][0], tri.pts[0][1])
          ctx.lineTo(tri.pts[1][0], tri.pts[1][1])
          ctx.lineTo(tri.pts[2][0], tri.pts[2][1])
          ctx.closePath()
          ctx.fill()
        }
      }
    }

    // ── Draw streamlines ────────────────────────────────────────────
    if (showStreamlines) {
      const vMax = Math.max(...scene.streamlines.map(s => s.vel_max), 1)
      ctx.lineWidth = 0.8
      for (const sl of scene.streamlines) {
        if (sl.n_points < 2) continue
        for (let i = 1; i < sl.n_points; i++) {
          const p0 = project(sl.points[(i - 1) * 3], sl.points[(i - 1) * 3 + 1], sl.points[(i - 1) * 3 + 2])
          const p1 = project(sl.points[i * 3], sl.points[i * 3 + 1], sl.points[i * 3 + 2])
          const v = sl.velocities[i] || 0
          ctx.strokeStyle = colormap === 'jet' ? jetColor(v / vMax) : viridisColor(v / vMax)
          ctx.beginPath()
          ctx.moveTo(p0[0], p0[1])
          ctx.lineTo(p1[0], p1[1])
          ctx.stroke()
        }
      }
    }

    // ── Colorbar ────────────────────────────────────────────────────
    const cbX = W - 60, cbY = 30, cbW = 16, cbH = H - 60
    for (let i = 0; i < cbH; i++) {
      const t = 1 - i / cbH
      ctx.fillStyle = colormap === 'jet' ? jetColor(t) : viridisColor(t)
      ctx.fillRect(cbX, cbY + i, cbW, 1)
    }
    ctx.strokeStyle = '#333'
    ctx.lineWidth = 0.5
    ctx.strokeRect(cbX, cbY, cbW, cbH)

    ctx.fillStyle = '#222'
    ctx.font = '10px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(`${scene.field_global_max.toFixed(0)}`, cbX + cbW + 4, cbY + 8)
    ctx.fillText(`${((scene.field_global_max + scene.field_global_min) / 2).toFixed(0)}`, cbX + cbW + 4, cbY + cbH / 2)
    ctx.fillText(`${scene.field_global_min.toFixed(0)}`, cbX + cbW + 4, cbY + cbH - 2)
    ctx.save()
    ctx.translate(cbX - 4, cbY + cbH / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillText(`${scene.field_name} [${scene.units}]`, 0, 0)
    ctx.restore()

    // Axis indicator
    ctx.font = '11px sans-serif'
    ctx.textAlign = 'left'
    ctx.fillStyle = '#333'
    ctx.fillText('drag = rotate, wheel = zoom', 12, H - 12)
  }, [scene, rotation, zoom, showSurfaces, showStreamlines, colormap])

  // Mouse handlers
  const onMouseDown = (e: React.MouseEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY }
  }
  const onMouseMove = (e: React.MouseEvent) => {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.x
    const dy = e.clientY - dragRef.current.y
    setRotation(r => ({
      x: r.x + dy * 0.01,
      y: r.y + dx * 0.01,
    }))
    dragRef.current = { x: e.clientX, y: e.clientY }
  }
  const onMouseUp = () => { dragRef.current = null }
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault()
    setZoom(z => Math.max(0.3, Math.min(3, z * (e.deltaY < 0 ? 1.1 : 0.9))))
  }

  return (
    <canvas
      ref={canvasRef}
      width={1100}
      height={height}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={onMouseUp}
      onWheel={onWheel}
      style={{
        width: '100%', height,
        border: '1px solid var(--border-subtle)',
        borderRadius: 6, cursor: dragRef.current ? 'grabbing' : 'grab',
        background: '#cfe8ff',
      }}
    />
  )
}

// ===========================================================================
// Color maps (matching backend)
// ===========================================================================

function viridisColor(t: number): string {
  t = Math.max(0, Math.min(1, t))
  const stops = [
    [68, 1, 84], [59, 82, 139], [33, 144, 141], [94, 201, 98], [253, 231, 37],
  ]
  const idx = t * (stops.length - 1)
  const i0 = Math.floor(idx)
  const i1 = Math.min(stops.length - 1, i0 + 1)
  const f = idx - i0
  const r = Math.round(stops[i0][0] * (1 - f) + stops[i1][0] * f)
  const g = Math.round(stops[i0][1] * (1 - f) + stops[i1][1] * f)
  const b = Math.round(stops[i0][2] * (1 - f) + stops[i1][2] * f)
  return `rgb(${r},${g},${b})`
}

function jetColor(t: number): string {
  t = Math.max(0, Math.min(1, t))
  let r = 0, g = 0, b = 0
  if (t < 0.125)      { r = 0;             g = 0;             b = 0.5 + t * 4 }
  else if (t < 0.375) { r = 0;             g = (t - 0.125)*4; b = 1 }
  else if (t < 0.625) { r = (t - 0.375)*4; g = 1;             b = 1 - (t-0.375)*4 }
  else if (t < 0.875) { r = 1;             g = 1 - (t-0.625)*4; b = 0 }
  else                { r = 1 - (t-0.875)*4; g = 0;           b = 0 }
  return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`
}

const checkLabel: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 4,
  cursor: 'pointer', color: 'var(--text-primary)',
}
