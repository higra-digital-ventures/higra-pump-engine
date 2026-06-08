/**
 * Central navigation config — SINGLE SOURCE OF TRUTH for tabs and sections.
 *
 * Before this module the tab→section mapping, the sub-tab lists, the sidebar
 * items and the "wide layout" list lived in 3-4 different files that had to be
 * kept in sync by hand. Everything navigation-related now derives from the
 * `SUB_TABS` table below.
 *
 * Sidebar.tsx re-exports Tab / Section / sectionForTab / SUB_TABS for backward
 * compatibility with the components that still import them from './Sidebar'.
 */

/* ── Section = a primary destination in the sidebar ──────────────────────── */
export type Section =
  | 'projects' | 'templates' | 'design' | 'geometry'
  | 'analysis' | 'optimization' | 'assistant'

/* ── Tab = every sub-page the app can show ───────────────────────────────── */
export type Tab =
  | 'overview' | 'results' | 'curves' | '3d' | 'velocity' | 'losses' | 'stress'
  | 'compare' | 'assistant' | 'optimize' | 'loading' | 'pressure'
  | 'multispeed' | 'meridional-editor' | 'spanwise'
  | 'templates' | 'doe' | 'pareto' | 'lean-sweep' | 'lete'
  | 'meridional-drag' | 'noise' | 'batch' | 'pipeline'
  | 'cavitation' | 'cfd_sim' | 'benchmarks' | 'geometry-modes'

export interface SubTab { key: Tab; label: string }

/* ── Sub-tabs per section (drives the horizontal SubTabBar) ───────────────── */
export const SUB_TABS: Record<Section, SubTab[]> = {
  projects: [],
  templates: [],
  assistant: [],
  design: [
    { key: 'overview', label: 'Resumo' },
    { key: 'results', label: 'Dimensionamento' },
    { key: 'curves', label: 'Curvas H-Q' },
    { key: 'multispeed', label: 'Multi-Velocidade' },
  ],
  geometry: [
    { key: '3d', label: 'Rotor 3D' },
    { key: 'geometry-modes', label: 'Modo (Clássico/Livre)' },
    { key: 'meridional-drag', label: 'Editor Meridional' },
    { key: 'meridional-editor', label: 'Meridional Avançado' },
    { key: 'lete', label: 'LE / TE' },
    { key: 'lean-sweep', label: 'Lean / Sweep / Bow' },
  ],
  analysis: [
    { key: 'velocity', label: 'Velocidades' },
    { key: 'losses', label: 'Perdas' },
    { key: 'pressure', label: 'Pressão PS/SS' },
    { key: 'loading', label: 'Carregamento rVθ' },
    { key: 'spanwise', label: 'Spanwise' },
    { key: 'noise', label: 'Ruído' },
    { key: 'stress', label: 'Tensões' },
    { key: 'compare', label: 'Comparação' },
    { key: 'cavitation', label: 'Cavitação' },
    { key: 'benchmarks', label: 'Benchmarks' },
  ],
  optimization: [
    { key: 'optimize', label: 'NSGA-II / Bayesian' },
    { key: 'pareto', label: 'Fronteira Pareto' },
    { key: 'doe', label: 'DoE / Surrogate' },
    { key: 'batch', label: 'Batch / Paramétrico' },
    { key: 'pipeline', label: 'Pipeline Completo' },
    { key: 'cfd_sim', label: 'Simulação CFD' },
  ],
}

/* ── Tabs that own the whole content area (no 2-column sizing-form layout) ── */
export const WIDE_TABS: Tab[] = [
  '3d', 'meridional-drag', 'meridional-editor', 'lete', 'lean-sweep',
  'doe', 'pareto', 'batch', 'templates', 'compare', 'optimize',
  'pipeline', 'cavitation', 'cfd_sim', 'benchmarks', 'geometry-modes',
]

export function isWideTab(tab: Tab): boolean {
  return WIDE_TABS.includes(tab)
}

/* ── tab → section, derived from SUB_TABS (no hand-maintained switch) ─────── */
const TAB_TO_SECTION: Record<string, Section> = (() => {
  const m: Record<string, Section> = {}
  ;(Object.keys(SUB_TABS) as Section[]).forEach(section => {
    SUB_TABS[section].forEach(st => { m[st.key] = section })
  })
  // Single-tab sections that have no SubTabBar entries:
  m['assistant'] = 'assistant'
  m['templates'] = 'templates'
  return m
})()

export function sectionForTab(tab: Tab): Section {
  return TAB_TO_SECTION[tab] ?? 'design'
}

/* ── Sidebar primary destinations (ALL sections are reachable) ────────────── */
export interface NavSection {
  key: Section
  label: string
  iconPath: string
  isPage?: 'projects'
  defaultTab?: Tab
  description: string
}

export const NAV_SECTIONS: NavSection[] = [
  {
    key: 'projects', label: 'Projetos', isPage: 'projects',
    description: 'Gerenciar projetos salvos e criar novos',
    iconPath: 'M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z',
  },
  {
    key: 'design', label: 'Design', defaultTab: 'overview',
    description: 'Dimensionamento 1D, curvas H-Q e multi-velocidade',
    iconPath: 'M4 21v-7 M4 10V3 M12 21v-9 M12 8V3 M20 21v-5 M20 12V3 M1 14h6 M9 8h6 M17 16h6',
  },
  {
    key: 'geometry', label: 'Geometria', defaultTab: '3d',
    description: 'Rotor 3D e editores meridional / LE-TE',
    iconPath: 'M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z M3.27 6.96L12 12.01l8.73-5.05 M12 22.08V12',
  },
  {
    key: 'analysis', label: 'Análise', defaultTab: 'velocity',
    description: 'Velocidades, perdas, pressão, cavitação, ruído e comparação',
    iconPath: 'M3 12h4l3-9 4 18 3-9h4',
  },
  {
    key: 'optimization', label: 'Otimização', defaultTab: 'optimize',
    description: 'NSGA-II, Bayesian, DoE, Pareto, pipeline e CFD',
    iconPath: 'M13 10V3L4 14h7v7l9-11h-7z',
  },
  {
    key: 'assistant', label: 'Assistente', defaultTab: 'assistant',
    description: 'Assistente de engenharia (RAG + regras Gülich)',
    iconPath: 'M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z',
  },
]
