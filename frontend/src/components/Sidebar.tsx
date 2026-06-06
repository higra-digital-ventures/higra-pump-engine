import React, { useState, useEffect } from 'react'
import t, { setLang, getCurrentLang, type LangKey } from '../i18n'
import {
  type Tab, type Section, type NavSection,
  sectionForTab, SUB_TABS, NAV_SECTIONS,
} from '../navigation'

/* Re-exported for components that still import these from './Sidebar'. */
export { sectionForTab, SUB_TABS }
export type { Tab, Section }

/* ── Props ─────────────────────────────────────────────────────────────────── */
interface Props {
  page: 'projects' | 'design'
  activeTab: Tab | null
  userName: string
  isCollapsed: boolean
  onToggleCollapse: () => void
  onNavigate: (page: 'projects' | 'design', tab?: Tab) => void
  onLogout: () => void
  warningCount?: number
}

/* ── SVG icon helper — supports multi-subpath `d` strings ──────────────────── */
const I = ({ d }: { d: string }) => (
  <svg viewBox="0 0 24 24">
    {d.split(' M').map((seg, i) => (
      <path key={i} d={i === 0 ? seg : `M${seg}`} />
    ))}
  </svg>
)

/* Nav items come straight from the central config — all sections are shown. */
const NAV_ITEMS: NavSection[] = NAV_SECTIONS

/* ── Component ─────────────────────────────────────────────────────────────── */

export default function Sidebar({
  page, activeTab, userName, isCollapsed, onToggleCollapse, onNavigate, onLogout, warningCount,
}: Props) {

  const activeSection = activeTab ? sectionForTab(activeTab) : (page === 'projects' ? 'projects' : 'design')

  const handleClick = (item: NavSection) => {
    if (item.isPage === 'projects') {
      onNavigate('projects')
    } else if (item.defaultTab) {
      onNavigate('design', item.defaultTab)
    }
  }

  // On the projects page only "Projetos" is relevant; on design show all sections.
  const visibleItems = page === 'projects'
    ? NAV_ITEMS.filter(i => i.key === 'projects')
    : NAV_ITEMS

  return (
    <div className={`sidebar${isCollapsed ? ' collapsed' : ''}`}>
      {/* ── Header ───────────────────────────────────────────────────────── */}
      <div className="sidebar-header">
        <div className="logo-icon">H</div>
        {!isCollapsed && <span className="logo-text">HPE</span>}
      </div>

      {/* ── Quick-access buttons (design mode only) ───────────────────── */}
      {page === 'design' && !isCollapsed && (
        <div style={{ display: 'flex', gap: 4, padding: '8px 8px 0' }}>
          {([
            { label: '3D', tab: '3d' as Tab },
            { label: 'Curvas', tab: 'curves' as Tab },
            { label: 'Otim.', tab: 'optimize' as Tab },
          ]).map(q => (
            <button
              key={q.tab}
              onClick={() => onNavigate('design', q.tab)}
              style={{
                flex: 1, padding: '4px 0', borderRadius: 4, fontSize: 10, fontWeight: 600,
                cursor: 'pointer', transition: 'all 0.15s',
                border: `1px solid ${activeTab === q.tab ? 'var(--accent)' : 'var(--border-primary)'}`,
                background: activeTab === q.tab ? 'rgba(0,160,223,0.15)' : 'transparent',
                color: activeTab === q.tab ? 'var(--accent)' : 'var(--text-muted)',
                fontFamily: 'var(--font-family)',
              }}
            >
              {q.label}
            </button>
          ))}
        </div>
      )}

      {/* ── Navigation ───────────────────────────────────────────────────── */}
      <nav className="sidebar-nav">
        {visibleItems.map(item => (
          <button
            key={item.key}
            className={`menu-item${activeSection === item.key ? ' active' : ''}`}
            onClick={() => handleClick(item)}
            title={isCollapsed ? item.label : (item.description || item.label)}
            style={{ position: 'relative' }}
          >
            <span className="icon"><I d={item.iconPath} /></span>
            {!isCollapsed && <span>{item.label}</span>}
            {item.key === 'design' && !!warningCount && warningCount > 0 && (
              <span style={{
                position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
                background: '#ef4444', color: '#fff', borderRadius: '50%',
                width: 16, height: 16, fontSize: 9, fontWeight: 700,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>{warningCount}</span>
            )}
          </button>
        ))}
      </nav>

      {/* ── Footer ───────────────────────────────────────────────────────── */}
      <div className="sidebar-footer" style={{ flexDirection: isCollapsed ? 'column' : 'row', gap: isCollapsed ? 8 : 10 }}>
        {!isCollapsed && (
          <>
            <div className="avatar">{userName.charAt(0).toUpperCase()}</div>
            <div className="user-info">
              <div className="user-name">{userName.length > 15 ? userName.slice(0, 15) + '...' : userName}</div>
              <div className="user-role" style={{ cursor: 'pointer' }} onClick={onLogout}>{t.logout}</div>
            </div>
            <LangSelector />
            <ThemeToggle />
          </>
        )}
        <button
          className="collapse-btn"
          onClick={onToggleCollapse}
          title={isCollapsed ? 'Expandir menu' : 'Recolher menu'}
          style={{
            width: isCollapsed ? 40 : undefined,
            height: isCollapsed ? 40 : undefined,
            borderRadius: isCollapsed ? 8 : undefined,
            background: isCollapsed ? 'var(--bg-surface)' : undefined,
            border: isCollapsed ? '1px solid var(--border-primary)' : undefined,
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            {isCollapsed ? <path d="M9 18l6-6-6-6" /> : <path d="M15 18l-6-6 6-6" />}
          </svg>
        </button>
        {isCollapsed && <LangSelector />}
        {isCollapsed && <ThemeToggle />}
      </div>
    </div>
  )
}

/* ── Language Selector (flag buttons) ─────────────────────────────────────── */
function LangSelector() {
  const current = getCurrentLang()
  const langs: { key: LangKey; flag: string }[] = [
    { key: 'pt-br', flag: '\uD83C\uDDE7\uD83C\uDDF7' },
    { key: 'en', flag: '\uD83C\uDDFA\uD83C\uDDF8' },
    { key: 'es', flag: '\uD83C\uDDEA\uD83C\uDDF8' },
  ]
  return (
    <div style={{ display: 'flex', gap: 2 }}>
      {langs.map(l => (
        <button
          key={l.key}
          onClick={() => { setLang(l.key); window.location.reload() }}
          title={l.key.toUpperCase()}
          style={{
            background: current === l.key ? 'var(--bg-surface)' : 'none',
            border: current === l.key ? '1px solid var(--border-primary)' : '1px solid transparent',
            borderRadius: 4, cursor: 'pointer', padding: '2px 4px',
            fontSize: 14, lineHeight: 1,
          }}
        >
          {l.flag}
        </button>
      ))}
    </div>
  )
}

/* ── Theme Toggle (dark / light / high-contrast) ─────────────────────────── */
const THEMES = ['dark', 'light', 'high-contrast'] as const
type Theme = typeof THEMES[number]

const THEME_LABELS: Record<Theme, string> = {
  dark: 'Escuro',
  light: 'Claro',
  'high-contrast': 'Alto Contraste',
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('hpe_theme') as Theme | null
    return saved && THEMES.includes(saved) ? saved : 'dark'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('hpe_theme', theme)
  }, [theme])

  // Also set on mount
  useEffect(() => {
    const saved = localStorage.getItem('hpe_theme') as Theme | null
    if (saved && THEMES.includes(saved)) {
      document.documentElement.setAttribute('data-theme', saved)
    }
  }, [])

  const nextTheme = () => {
    setTheme(prev => {
      const idx = THEMES.indexOf(prev)
      return THEMES[(idx + 1) % THEMES.length]
    })
  }

  return (
    <button
      onClick={nextTheme}
      title={THEME_LABELS[theme]}
      style={{
        background: 'none', border: 'none', cursor: 'pointer',
        color: 'var(--text-muted)', padding: 4, display: 'flex', alignItems: 'center',
      }}
    >
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        {theme === 'dark' ? (
          /* Sun icon — clicking will go to light */
          <>
            <circle cx="12" cy="12" r="5" />
            <line x1="12" y1="1" x2="12" y2="3" />
            <line x1="12" y1="21" x2="12" y2="23" />
            <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
            <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
            <line x1="1" y1="12" x2="3" y2="12" />
            <line x1="21" y1="12" x2="23" y2="12" />
            <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
            <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
          </>
        ) : theme === 'light' ? (
          /* Half-circle icon — clicking will go to high-contrast */
          <circle cx="12" cy="12" r="9" fill="currentColor" fillOpacity="0.5" />
        ) : (
          /* Moon icon — clicking will go to dark */
          <path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
        )}
      </svg>
    </button>
  )
}
