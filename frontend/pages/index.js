/* ═══════════════════════════════════════════════════
   pages/index.js — SentiView · YouTube Sentiment AI
   Complete redesign: warm cream editorial theme.
   ═══════════════════════════════════════════════════ */

import { useState, useEffect, useRef } from 'react'
import Head from 'next/head'
import LoadingAnimation from '../components/LoadingAnimation'
import ResultsDashboard from '../components/ResultsDashboard'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

/* ─── Topbar ─────────────────────────────────────── */
function Topbar({ onReset, inResults }) {
  return (
    <header className="topbar">
      <div
        style={{
          maxWidth      : 900,
          margin        : '0 auto',
          padding       : '0 24px',
          height        : '100%',
          display       : 'flex',
          alignItems    : 'center',
          justifyContent: 'space-between',
        }}
      >
        {/* Logo */}
        <div
          style={{
            display    : 'flex',
            alignItems : 'center',
            gap        : 10,
            cursor     : inResults ? 'pointer' : 'default',
            userSelect : 'none',
          }}
          onClick={inResults ? onReset : undefined}
          title={inResults ? 'Back to search' : undefined}
        >
          <div
            style={{
              width          : 34,
              height         : 34,
              borderRadius   : 10,
              background     : '#0F172A',
              display        : 'flex',
              alignItems     : 'center',
              justifyContent : 'center',
              flexShrink     : 0,
            }}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="white">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14.5v-9l6 4.5-6 4.5z" />
            </svg>
          </div>
          <div>
            <div
              style={{
                fontSize     : 15,
                fontWeight   : 800,
                letterSpacing: '-0.028em',
                color        : '#0F172A',
                lineHeight   : 1.2,
                fontFamily   : "'Plus Jakarta Sans', sans-serif",
              }}
            >
              SentiView
            </div>
            <div
              className="font-mono"
              style={{ fontSize: 9.5, color: '#9CA3AF', letterSpacing: '0.06em' }}
            >
              YT Comment Analyzer
            </div>
          </div>
        </div>

        {/* Pills (desktop only, hide in results) */}
        {!inResults && (
        <div className="sm-hide" style={{ display: 'flex', gap: 6 }}>
          {['BiLSTM + ML Ensemble', 'EN · NE · HI', '125k+ Training Comments'].map(tag => (
            <span
              key={tag}
              style={{
                fontSize  : 11.5,
                fontWeight: 500,
                padding   : '4px 11px',
                borderRadius: 20,
                background: '#EDE3D8',
                color     : '#6B7280',
              }}
            >
              {tag}
            </span>
          ))}
        </div>
        )}

        {/* Back button when in results */}
        {inResults && (
          <button
            className="btn-ghost"
            onClick={onReset}
            style={{ fontSize: 13, color: '#6B7280', fontWeight: 600 }}
          >
            ← New search
          </button>
        )}
      </div>
    </header>
  )
}

/* ─── Error banner ───────────────────────────────── */
function ErrorBanner({ message, onDismiss }) {
  return (
    <div
      style={{
        width       : '100%',
        maxWidth    : 630,
        marginBottom: 16,
        background  : '#FEF2F2',
        border      : '1px solid #FECACA',
        borderRadius: 14,
        padding     : '14px 18px',
        display     : 'flex',
        alignItems  : 'flex-start',
        gap         : 12,
      }}
    >
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none" style={{ flexShrink: 0, marginTop: 1 }}>
        <circle cx="8" cy="8" r="8" fill="#DC2626" />
        <path d="M8 4.5v4M8 11v.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
      <div style={{ flex: 1, fontSize: 13, color: '#991B1B', lineHeight: 1.5 }}>
        <strong>Analysis failed — </strong>{message}
      </div>
      <button
        className="btn-ghost"
        onClick={onDismiss}
        style={{ color: '#991B1B', fontSize: 18, lineHeight: 1, marginTop: -1 }}
      >
        ×
      </button>
    </div>
  )
}

/* ─── Search form card ───────────────────────────── */
function SearchForm({ onSubmit, loading }) {
  const [url,       setUrl]       = useState('')
  const [model,     setModel]     = useState('ensemble')
  const [maxN,      setMaxN]      = useState(2000)
  const [showAdv,   setShowAdv]   = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    const t = setTimeout(() => inputRef.current?.focus(), 300)
    return () => clearTimeout(t)
  }, [])

  const isValid = url.trim().length > 0

  const submit = () => {
    if (!isValid || loading) return
    onSubmit({ url: url.trim(), model, max_comments: maxN })
  }

  const MODELS = [
    { id: 'ensemble', label: 'ML + DL Ensemble', sub: 'Best accuracy'      },
    { id: 'dl',       label: 'BiLSTM Only',      sub: 'Best for code-mix'  },
    { id: 'ml',       label: 'ML Ensemble Only', sub: 'Fastest'            },
  ]

  return (
    <div
      className="card anim-u3"
      style={{ width: '100%', maxWidth: 630, overflow: 'hidden' }}
    >
      {/* Input row */}
      <div style={{ padding: '20px 20px 0', display: 'flex', gap: 10 }}>

        {/* URL input */}
        <div style={{ flex: 1, position: 'relative' }}>
          {/* YT icon */}
          <svg
            width="17" height="17" viewBox="0 0 24 24"
            style={{
              position : 'absolute',
              left     : 13,
              top      : '50%',
              transform: 'translateY(-50%)',
              pointerEvents: 'none',
            }}
          >
            <rect x="2" y="6" width="20" height="14" rx="3" fill="#FF0000" opacity="0.9" />
            <path d="M10 9.5l5 3-5 3V9.5z" fill="white" />
          </svg>
          <input
            ref={inputRef}
            className="sv-input"
            type="text"
            value={url}
            onChange={e => setUrl(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && submit()}
            placeholder="https://youtube.com/watch?v=..."
            style={{ paddingRight: url ? 36 : 14 }}
          />
          {url && (
            <button
              className="btn-ghost"
              onClick={() => setUrl('')}
              style={{
                position : 'absolute',
                right    : 11,
                top      : '50%',
                transform: 'translateY(-50%)',
                color    : '#9CA3AF',
                fontSize : 17,
                lineHeight: 1,
              }}
            >
              ×
            </button>
          )}
        </div>

        {/* Analyze button */}
        <button
          className="btn-primary"
          onClick={submit}
          disabled={!isValid || loading}
          style={{ height: 48, padding: '0 22px' }}
        >
          {loading ? (
            <>
              <span
                style={{
                  width       : 14,
                  height      : 14,
                  border      : '2px solid rgba(255,255,255,0.3)',
                  borderTopColor: '#fff',
                  borderRadius: '50%',
                  display     : 'inline-block',
                  animation   : 'sv-spin 0.7s linear infinite',
                }}
              />
              Analyzing
            </>
          ) : 'Analyze →'}
        </button>
      </div>

      {/* Advanced toggle */}
      <div style={{ padding: '10px 20px 0' }}>
        <button
          className="btn-ghost"
          onClick={() => setShowAdv(!showAdv)}
          style={{ color: '#9CA3AF', fontSize: 12, fontWeight: 500 }}
        >
          <svg
            width="11" height="11" viewBox="0 0 11 11" fill="none"
            style={{
              transform : showAdv ? 'rotate(180deg)' : 'none',
              transition: 'transform 0.2s',
            }}
          >
            <path
              d="M2 4l3.5 3.5L9 4"
              stroke="currentColor" strokeWidth="1.5"
              strokeLinecap="round" strokeLinejoin="round"
            />
          </svg>
          Advanced options
        </button>
      </div>

      {/* Advanced panel */}
      {showAdv && (
        <div
          style={{
            margin      : '12px 20px 0',
            padding     : '16px 18px',
            background  : '#F8FAFC',
            borderRadius: 12,
            border      : '1px solid #F1F5F9',
          }}
        >
          {/* Model selector */}
          <div style={{ marginBottom: 18 }}>
            <div className="label-caps" style={{ marginBottom: 9 }}>
              Model
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {MODELS.map(opt => (
                <button
                  key={opt.id}
                  className="model-btn"
                  onClick={() => setModel(opt.id)}
                  style={{
                    border    : `1.5px solid ${model === opt.id ? '#0F172A' : '#E2E8F0'}`,
                    background: model === opt.id ? '#0F172A' : '#FFF',
                  }}
                >
                  <div
                    style={{
                      fontSize  : 13,
                      fontWeight: 600,
                      color     : model === opt.id ? '#FFF' : '#334155',
                    }}
                  >
                    {opt.label}
                  </div>
                  <div
                    style={{
                      fontSize  : 10.5,
                      marginTop : 1,
                      color     : model === opt.id ? 'rgba(255,255,255,0.5)' : '#94A3B8',
                    }}
                  >
                    {opt.sub}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Max comments */}
          <div>
            <div className="label-caps" style={{ marginBottom: 9 }}>
              Max Comments —{' '}
              <span
                className="font-mono"
                style={{ color: '#0F172A', fontWeight: 600, textTransform: 'none' }}
              >
                {maxN.toLocaleString()}
              </span>
            </div>
            <input
              type="range"
              min={50}
              max={20000}
              step={50}
              value={maxN}
              onChange={e => setMaxN(Number(e.target.value))}
            />
          </div>
        </div>
      )}

      {/* Footer strip */}
      <div className="input-footer" style={{ marginTop: 14 }}>
        <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
          {['BiLSTM + Attention', 'ML Ensemble', 'Romanized NE & HI', '125k Training Comments'].map(tag => (
            <span
              key={tag}
              style={{
                fontSize  : 11.5,
                color     : '#9CA3AF',
                fontWeight: 500,
                display   : 'flex',
                alignItems: 'center',
                gap       : 5,
              }}
            >
              <span
                style={{
                  width: 3, height: 3,
                  borderRadius: '50%',
                  background: '#D1D5DB',
                  display: 'inline-block',
                }}
              />
              {tag}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ─── Idle / hero page ───────────────────────────── */
function HeroPage({ onSubmit, loading, error, onDismissError }) {
  return (
    <div
      style={{
        minHeight     : '100vh',
        display       : 'flex',
        flexDirection : 'column',
        alignItems    : 'center',
        justifyContent: 'center',
        padding       : '110px 20px 100px',
      }}
    >
      {/* Live badge */}
      <div
        className="anim-u0"
        style={{
          display    : 'inline-flex',
          alignItems : 'center',
          gap        : 7,
          background : '#ECFDF5',
          color      : '#065F46',
          border     : '1px solid #A7F3D0',
          fontSize   : 12,
          fontWeight : 600,
          padding    : '5px 14px',
          borderRadius: 30,
          marginBottom: 24,
        }}
      >
        <span
          style={{
            width      : 6,
            height     : 6,
            borderRadius: '50%',
            background : '#059669',
            display    : 'inline-block',
            animation  : 'sv-pulse-dot 2s ease-in-out infinite',
          }}
        />
        AI-powered · 125,000+ multilingual training comments
      </div>

      {/* Hero heading */}
      <h1
        className="font-serif hero-title anim-u1"
        style={{
          fontSize    : 'clamp(28px, 4.2vw, 50px)',
          lineHeight  : 1.14,
          color       : '#0F172A',
          textAlign   : 'center',
          marginBottom: 18,
          fontStyle   : 'italic',
          maxWidth    : 780,
          letterSpacing: '-0.015em',
          fontWeight  : 500,
        }}
      >
        Vibe Check: A Deep Dive into{' '}
        <em style={{ color: '#059669', fontStyle: 'italic' }}>YouTube Audience Sentiment</em>
      </h1>

      <p
        className="anim-u2"
        style={{
          fontSize    : 16,
          color       : '#6B7280',
          lineHeight  : 1.65,
          textAlign   : 'center',
          maxWidth    : 490,
          marginBottom: 46,
        }}
      >
        Paste any YouTube link and our BiLSTM + Ensemble AI maps comment
        sentiment across English, Nepali, and Hindi — including romanized script.
      </p>

      {/* Error */}
      {error && (
        <ErrorBanner message={error} onDismiss={onDismissError} />
      )}

      {/* Search card */}
      <SearchForm onSubmit={onSubmit} loading={loading} />

      {/* Footer tagline */}
      <p
        className="font-mono anim-u4"
        style={{
          fontSize  : 11.5,
          color     : '#94A3B8',
          marginTop : 36,
          textAlign : 'center',
          maxWidth  : 440,
          lineHeight: 1.6,
        }}
      >
        Handles Romanized Nepali & Hindi · Trained on 125k+ YouTube comments ·
        BiLSTM + Attention + ML Ensemble
      </p>
    </div>
  )
}

/* ─── Main page ──────────────────────────────────── */
export default function Home() {
  const [state,   setState]   = useState('idle')   // idle | loading | results
  const [result,  setResult]  = useState(null)
  const [error,   setError]   = useState('')
  const [lastUrl, setLastUrl] = useState('')
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef(null)

  const startTimer = () => {
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(p => p + 1), 1000)
  }
  const stopTimer = () => {
    clearInterval(timerRef.current)
    timerRef.current = null
  }

  const handleAnalyze = async ({ url, model, max_comments }) => {
    setState('loading')
    setLastUrl(url)
    setError('')
    setResult(null)
    startTimer()

    try {
      const resp = await fetch(`${API_BASE}/api/analyze`, {
        method : 'POST',
        headers: { 'Content-Type': 'application/json' },
        body   : JSON.stringify({ url, model, max_comments }),
      })
      const data = await resp.json()
      stopTimer()

      if (!resp.ok || !data.success)
        throw new Error(data.error || data.detail || 'Server error')

      setResult(data)
      setState('results')
    } catch (err) {
      stopTimer()
      setError(err.message || 'Failed to connect to backend. Is the server running?')
      setState('idle')
    }
  }

  const handleReset = () => {
    setState('idle')
    setResult(null)
    setError('')
  }

  return (
    <>
      <Head>
        <title>SentiView — YouTube Comment Sentiment</title>
        <meta
          name="description"
          content="AI-powered YouTube comment sentiment analysis. BiLSTM + Ensemble models supporting English, Nepali, and Hindi."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div style={{ background: '#F5EDE3', minHeight: '100vh' }}>

        {/* Fixed topbar */}
        <Topbar
          onReset={handleReset}
          inResults={state === 'results'}
        />

        {/* Pages */}
        {state === 'idle' && (
          <HeroPage
            onSubmit={handleAnalyze}
            loading={false}
            error={error}
            onDismissError={() => setError('')}
          />
        )}

        {state === 'loading' && (
          <LoadingAnimation videoUrl={lastUrl} elapsed={elapsed} />
        )}

        {state === 'results' && result && (
          <ResultsDashboard result={result} onReset={handleReset} />
        )}
      </div>
    </>
  )
}
