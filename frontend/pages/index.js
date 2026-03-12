import { useState, useRef } from 'react'
import Head from 'next/head'
import LoadingAnimation from '../components/LoadingAnimation'
import ResultsDashboard from '../components/ResultsDashboard'

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ── Background decorations ──────────────────────────────────────────────────
function BackgroundFX() {
  return (
    <>
      {/* Gradient orbs */}
      <div
        className="fixed top-[-30%] left-[-10%] w-[700px] h-[700px] rounded-full pointer-events-none"
        style={{
          background: 'radial-gradient(circle, rgba(108,99,255,0.06) 0%, transparent 70%)',
          filter: 'blur(40px)',
        }}
      />
      <div
        className="fixed bottom-[-20%] right-[-10%] w-[600px] h-[600px] rounded-full pointer-events-none"
        style={{
          background: 'radial-gradient(circle, rgba(0,229,160,0.05) 0%, transparent 70%)',
          filter: 'blur(40px)',
        }}
      />
      {/* Grid */}
      <div className="fixed inset-0 bg-grid pointer-events-none opacity-60" />
    </>
  )
}

// ── Hero header ─────────────────────────────────────────────────────────────
function HeroHeader() {
  return (
    <div className="text-center mb-10 animate-fade-in">
      {/* Logo mark */}
      <div className="flex justify-center mb-5">
        <div
          className="relative w-16 h-16 rounded-2xl flex items-center justify-center"
          style={{
            background: 'linear-gradient(135deg, rgba(108,99,255,0.2) 0%, rgba(0,229,160,0.2) 100%)',
            border: '1.5px solid rgba(108,99,255,0.3)',
            boxShadow: '0 0 30px rgba(108,99,255,0.15)',
          }}
        >
          <span className="text-2xl animate-float">🧠</span>
        </div>
      </div>

      {/* Title */}
      <h1 className="font-display font-black text-4xl sm:text-5xl lg:text-6xl text-white leading-none mb-3 tracking-tight">
        YouTube{' '}
        <span className="gradient-text">Sentiment</span>
        <br />
        <span style={{ fontSize: '0.75em', opacity: 0.7, letterSpacing: '0.04em' }}>
          ANALYZER
        </span>
      </h1>

      <p className="font-body text-sm sm:text-base text-white/35 max-w-md mx-auto leading-relaxed">
        Paste any YouTube link. Our BiLSTM + Ensemble AI analyzes comments across{' '}
        <span className="text-white/60 font-medium">English, Nepali & Hindi</span>{' '}
        to reveal the true sentiment of your audience.
      </p>

      {/* Tech badges */}
      <div className="flex flex-wrap justify-center gap-2 mt-5">
        {['BiLSTM + Attention', 'ML Ensemble', 'Romanized Nepali/Hindi', '125k+ Training Comments'].map(tag => (
          <span
            key={tag}
            className="font-mono text-[10px] px-2.5 py-1 rounded-full text-white/35"
            style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── Search / input form ──────────────────────────────────────────────────────
function SearchForm({ onSubmit, loading }) {
  const [url, setUrl]       = useState('')
  const [model, setModel]   = useState('ensemble')
  const [maxN, setMaxN]     = useState(2000)
  const [showAdv, setShowAdv] = useState(false)
  const inputRef = useRef(null)

  const handleSubmit = (e) => {
    e?.preventDefault()
    if (!url.trim() || loading) return
    onSubmit({ url: url.trim(), model, max_comments: maxN })
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSubmit()
  }

  const isYouTubeUrl = url.includes('youtube') || url.includes('youtu.be') || /^[A-Za-z0-9_-]{11}$/.test(url)

  return (
    <div className="glass-card p-6 w-full max-w-2xl mx-auto animate-slide-up" style={{ borderColor: 'rgba(108,99,255,0.28)', boxShadow: '0 0 40px rgba(108,99,255,0.08), 0 2px 20px rgba(0,0,0,0.4)' }}>
      {/* URL input */}
      <div className="relative mb-4">
        {/* YouTube icon */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 pointer-events-none">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
            <rect x="2" y="6" width="20" height="14" rx="3" fill="#ff0000" opacity={isYouTubeUrl ? '1' : '0.3'} />
            <path d="M10 9.5l5 3-5 3V9.5z" fill="white" />
          </svg>
        </div>
        <input
          ref={inputRef}
          type="text"
          value={url}
          onChange={e => setUrl(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="https://youtube.com/watch?v=... or video ID"
          className="url-input"
          style={{ paddingLeft: 46 }}
          autoFocus
        />
        {url && (
          <button
            onClick={() => setUrl('')}
            className="absolute right-4 top-1/2 -translate-y-1/2 text-white/20 hover:text-white/50 transition-colors"
          >
            ✕
          </button>
        )}
      </div>

      {/* Action row */}
      <div className="flex gap-3 items-stretch">
        <button
          onClick={handleSubmit}
          disabled={!url.trim() || loading}
          className="analyze-btn flex-1"
        >
          {loading ? (
            <span className="flex items-center justify-center gap-2">
              <span className="inline-block w-3 h-3 border-2 border-white/40 border-t-white rounded-full animate-spin" />
              Analyzing...
            </span>
          ) : (
            '⚡ Analyze Sentiment'
          )}
        </button>

        {/* Advanced toggle */}
        <button
          onClick={() => setShowAdv(!showAdv)}
          className="px-4 rounded-xl text-sm font-display font-semibold transition-all duration-200"
          style={{
            background: showAdv ? 'rgba(108,99,255,0.2)' : 'rgba(255,255,255,0.04)',
            border    : showAdv ? '1px solid rgba(108,99,255,0.4)' : '1px solid rgba(255,255,255,0.08)',
            color     : showAdv ? '#fff' : 'rgba(255,255,255,0.4)',
          }}
          title="Advanced options"
        >
          ⚙️
        </button>
      </div>

      {/* Advanced options */}
      {showAdv && (
        <div
          className="mt-4 p-4 rounded-xl space-y-4 animate-fade-in"
          style={{ background: 'rgba(0,0,0,0.3)', border: '1px solid rgba(255,255,255,0.06)' }}
        >
          {/* Model selector */}
          <div>
            <label className="font-mono text-xs text-white/30 uppercase tracking-wider block mb-2">
              Model
            </label>
            <div className="flex gap-2 flex-wrap">
              {[
                { id: 'ensemble', label: '🔀 ML + DL Ensemble', sub: 'Best accuracy' },
                { id: 'dl',       label: '🧠 BiLSTM Only',       sub: 'Best for code-mix' },
                { id: 'ml',       label: '🌲 ML Ensemble Only',  sub: 'Fastest' },
              ].map(opt => (
                <button
                  key={opt.id}
                  onClick={() => setModel(opt.id)}
                  className="px-3 py-2 rounded-lg text-xs font-display font-semibold transition-all duration-200 text-left"
                  style={{
                    background: model === opt.id ? 'rgba(108,99,255,0.2)' : 'rgba(255,255,255,0.03)',
                    border    : model === opt.id ? '1px solid rgba(108,99,255,0.5)' : '1px solid rgba(255,255,255,0.06)',
                    color     : model === opt.id ? '#fff' : 'rgba(255,255,255,0.4)',
                  }}
                >
                  <div>{opt.label}</div>
                  <div className="font-mono font-normal text-[10px] opacity-50 mt-0.5">{opt.sub}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Max comments */}
          <div>
            <label className="font-mono text-xs text-white/30 uppercase tracking-wider block mb-2">
              Max Comments to Analyze
            </label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={50} max={20000} step={50}
                value={maxN}
                onChange={e => setMaxN(Number(e.target.value))}
                className="flex-1"
                style={{ accentColor: '#6c63ff' }}
              />
              <span className="font-mono text-sm text-white/60 w-12 text-right">{maxN}</span>
            </div>
          </div>
        </div>
      )}


    </div>
  )
}

// ── Error message ────────────────────────────────────────────────────────────
function ErrorBanner({ message, onDismiss }) {
  return (
    <div
      className="glass-card p-5 w-full max-w-2xl mx-auto animate-slide-up"
      style={{ background: 'rgba(255,61,110,0.06)', borderColor: 'rgba(255,61,110,0.3)' }}
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <span className="text-xl mt-0.5">⚠️</span>
          <div>
            <p className="font-display font-bold text-sm text-red-400">Analysis Failed</p>
            <p className="font-mono text-xs text-white/40 mt-1">{message}</p>
          </div>
        </div>
        <button
          onClick={onDismiss}
          className="text-white/30 hover:text-white/60 transition-colors text-sm shrink-0"
        >
          ✕
        </button>
      </div>
    </div>
  )
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function Home() {
  const [state, setState]       = useState('idle')  // idle | loading | results | error
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState('')
  const [lastUrl, setLastUrl]   = useState('')
  const [elapsed, setElapsed]   = useState(0)
  const timerRef                = useRef(null)

  const startTimer = () => {
    setElapsed(0)
    timerRef.current = setInterval(() => {
      setElapsed(prev => prev + 1)
    }, 1000)
  }

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
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

      if (!resp.ok || !data.success) {
        throw new Error(data.error || data.detail || 'Server error')
      }

      setResult(data)
      setState('results')

    } catch (err) {
      stopTimer()
      console.error('Analysis error:', err)
      setError(err.message || 'Failed to connect to backend. Is the server running?')
      setState('error')
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
        <title>YouTube Sentiment Analyzer</title>
        <meta name="description" content="AI-powered YouTube comment sentiment analysis with BiLSTM + Ensemble models" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <div className="min-h-screen relative noise-overlay">
        <BackgroundFX />

        <main className="relative z-10 min-h-screen flex flex-col items-center justify-start px-4 pt-16 pb-24">

          {/* Always-visible header when not in results */}
          {state !== 'results' && <HeroHeader />}

          {/* Idle state — show input */}
          {state === 'idle' && (
            <SearchForm onSubmit={handleAnalyze} loading={false} />
          )}

          {/* Error state */}
          {state === 'error' && (
            <div className="w-full max-w-2xl mx-auto space-y-4">
              <ErrorBanner message={error} onDismiss={handleReset} />
              <SearchForm onSubmit={handleAnalyze} loading={false} />
            </div>
          )}

          {/* Loading state */}
          {state === 'loading' && (
            <LoadingAnimation videoUrl={lastUrl} elapsed={elapsed} />
          )}

          {/* Results state */}
          {state === 'results' && result && (
            <ResultsDashboard result={result} onReset={handleReset} />
          )}

          {/* Footer */}
          {state === 'idle' && (
            <footer className="mt-16 text-center">
              <p className="font-mono text-[11px] text-white/45 max-w-sm mx-auto">
                Powered by BiLSTM + Attention + ML Ensemble · Trained on 125k+ multilingual YouTube comments · Handles Nepali/Hindi Roman script
              </p>
            </footer>
          )}
        </main>
      </div>
    </>
  )
}
