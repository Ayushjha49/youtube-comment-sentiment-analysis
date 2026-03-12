import { useEffect, useState } from 'react'

// Steps now have REALISTIC time estimates — no fake 1.2s / 2.4s / 3.6s delays.
// The active step advances based on real elapsed time passed in from the parent.
const STEPS = [
  {
    id        : 'fetch',
    icon      : '📡',
    label     : 'Fetching YouTube comments',
    sub       : 'Connecting to YouTube Data API v3 · ~30–90s depending on count',
    color     : '#6c63ff',
    startAt   : 0,    // becomes active at 0s elapsed
  },
  {
    id        : 'clean',
    icon      : '🧹',
    label     : 'Cleaning & normalizing text',
    sub       : 'Handling Romanized Nepali/Hindi + English + emojis',
    color     : '#4ecdc4',
    startAt   : 15,   // becomes active at ~15s elapsed (still fetching but also cleaning)
  },
  {
    id        : 'model',
    icon      : '🧠',
    label     : 'Running sentiment model',
    sub       : 'BiLSTM + Attention · ML Ensemble · per-comment predictions',
    color     : '#f0b429',
    startAt   : 40,   // model runs after fetch completes
  },
  {
    id        : 'results',
    icon      : '📊',
    label     : 'Aggregating & generating results',
    sub       : 'Computing distribution · picking top comments',
    color     : '#00e5a0',
    startAt   : 70,   // final stage
  },
]

function PulsingRing({ color }) {
  return (
    <div className="relative flex items-center justify-center" style={{ width: 36, height: 36 }}>
      <div className="absolute inset-0 rounded-full animate-ping" style={{ background: color, opacity: 0.2 }} />
      <div className="absolute inset-1 rounded-full animate-pulse" style={{ background: color, opacity: 0.4 }} />
      <div className="relative w-3 h-3 rounded-full" style={{ background: color, boxShadow: `0 0 12px ${color}` }} />
    </div>
  )
}

function CheckIcon({ color }) {
  return (
    <div className="flex items-center justify-center rounded-full"
      style={{ width: 36, height: 36, background: `${color}22`, border: `1.5px solid ${color}` }}>
      <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
        <path d="M3 8l3.5 3.5L13 5" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </div>
  )
}

function PendingIcon() {
  return (
    <div className="flex items-center justify-center rounded-full"
      style={{ width: 36, height: 36, background: 'rgba(255,255,255,0.04)', border: '1.5px solid rgba(255,255,255,0.1)' }}>
      <div className="w-2 h-2 rounded-full bg-white/20" />
    </div>
  )
}

function NeuralBackground() {
  return (
    <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
      <svg className="absolute inset-0 w-full h-full opacity-10" viewBox="0 0 400 300">
        {[[60,60],[60,150],[60,240],[160,40],[160,110],[160,190],[160,260],[260,70],[260,150],[260,230],[340,100],[340,200]]
          .map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r="5" fill="#6c63ff" opacity="0.6">
            <animate attributeName="opacity" values="0.3;0.8;0.3" dur={`${1.5 + i*0.2}s`} repeatCount="indefinite" />
          </circle>
        ))}
        {[[60,60,160,40],[60,60,160,110],[60,150,160,110],[60,150,160,190],[60,240,160,190],[60,240,160,260],
          [160,40,260,70],[160,110,260,70],[160,110,260,150],[160,190,260,150],[160,190,260,230],[160,260,260,230],
          [260,70,340,100],[260,150,340,100],[260,150,340,200],[260,230,340,200]]
          .map(([x1,y1,x2,y2], i) => (
          <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#6c63ff" strokeWidth="1">
            <animate attributeName="opacity" values="0.1;0.4;0.1" dur={`${2 + i*0.1}s`} repeatCount="indefinite" />
          </line>
        ))}
      </svg>
    </div>
  )
}

// Format seconds into a human-readable string: 0–59s → "Xs", 60+ → "Xm Ys"
function formatElapsed(sec) {
  if (sec < 60) return `${sec}s`
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return s > 0 ? `${m}m ${s}s` : `${m}m`
}

export default function LoadingAnimation({ videoUrl, elapsed = 0 }) {
  const [dots, setDots] = useState('')

  // Determine active step from real elapsed time
  const activeStepIndex = STEPS.reduce((best, step, i) => {
    return elapsed >= step.startAt ? i : best
  }, 0)

  const completedStepIds = STEPS
    .filter((_, i) => i < activeStepIndex)
    .map(s => s.id)

  // Animated dots
  useEffect(() => {
    const interval = setInterval(() => {
      setDots(d => d.length >= 3 ? '' : d + '.')
    }, 500)
    return () => clearInterval(interval)
  }, [])

  // Smooth progress: within a step, interpolate toward the next step's startAt
  const currentStep  = STEPS[activeStepIndex]
  const nextStep     = STEPS[activeStepIndex + 1]
  const stepProgress = nextStep
    ? Math.min(1, (elapsed - currentStep.startAt) / (nextStep.startAt - currentStep.startAt))
    : Math.min(1, (elapsed - currentStep.startAt) / 30)
  const overallPct = Math.min(98, ((activeStepIndex + stepProgress) / STEPS.length) * 100)

  return (
    <div className="glass-card relative overflow-hidden p-8 w-full max-w-2xl mx-auto animate-slide-up">
      <NeuralBackground />

      {/* Header */}
      <div className="relative z-10 text-center mb-8">
        <div className="flex items-center justify-center gap-3 mb-3">
          <div className="relative w-10 h-10">
            <svg className="w-10 h-10 -rotate-90" viewBox="0 0 36 36">
              <circle cx="18" cy="18" r="14" fill="none" stroke="rgba(108,99,255,0.15)" strokeWidth="3"/>
              <circle
                cx="18" cy="18" r="14" fill="none"
                stroke="url(#gradient-loader)" strokeWidth="3"
                strokeLinecap="round"
                strokeDasharray="60 88"
                style={{ animation: 'spin 1s linear infinite' }}
              />
              <defs>
                <linearGradient id="gradient-loader" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#6c63ff" />
                  <stop offset="100%" stopColor="#00e5a0" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <h3 className="font-display font-bold text-xl text-white">
            Analyzing comments{dots}
          </h3>
        </div>

        {/* Live elapsed timer — the key UX fix */}
        <div className="flex items-center justify-center gap-2 mt-1">
          <span className="font-mono text-xs text-white/25">elapsed:</span>
          <span
            className="font-mono text-sm font-bold"
            style={{ color: elapsed > 60 ? '#f0b429' : '#6c63ff' }}
          >
            {formatElapsed(elapsed)}
          </span>
          {elapsed > 45 && (
            <span className="font-mono text-[10px] text-white/20">
              (large videos take 1–2 min)
            </span>
          )}
        </div>

        {videoUrl && (
          <p className="font-mono text-xs text-white/30 truncate max-w-xs mx-auto mt-2">
            {videoUrl}
          </p>
        )}
      </div>

      {/* Steps — driven by real elapsed time */}
      <div className="relative z-10 space-y-2">
        {STEPS.map((step, i) => {
          const isDone    = completedStepIds.includes(step.id)
          const isActive  = activeStepIndex === i
          const isPending = !isDone && !isActive

          return (
            <div
              key={step.id}
              className={`loading-step ${isActive ? 'active' : ''} ${isDone ? 'done' : ''} ${isPending ? 'pending' : ''}`}
              style={{ transitionDelay: `${i * 50}ms` }}
            >
              {isActive  ? <PulsingRing color={step.color} /> : null}
              {isDone    ? <CheckIcon color={step.color} />   : null}
              {isPending ? <PendingIcon />                    : null}

              <div className="flex-1 min-w-0">
                <div
                  className="font-display font-semibold text-sm"
                  style={{ color: isActive ? step.color : isDone ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.2)' }}
                >
                  {step.label}
                </div>
                <div className="text-xs text-white/25 mt-0.5 font-mono">{step.sub}</div>
              </div>

              <div
                className="font-mono text-xs shrink-0"
                style={{ color: isActive ? step.color : 'rgba(255,255,255,0.1)' }}
              >
                {String(i + 1).padStart(2, '0')}
              </div>
            </div>
          )
        })}
      </div>

      {/* Progress bar driven by real elapsed time */}
      <div className="relative z-10 mt-6">
        <div className="progress-bar-track">
          <div
            className="progress-bar-fill shimmer"
            style={{
              width     : `${overallPct}%`,
              background: 'linear-gradient(90deg, #6c63ff, #00e5a0)',
              transition: 'width 1s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
          />
        </div>
        <div className="flex justify-between mt-2 font-mono text-xs text-white/20">
          <span>Processing</span>
          <span>{Math.round(overallPct)}%</span>
        </div>
      </div>

      <style jsx>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  )
}
