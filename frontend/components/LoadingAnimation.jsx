/* ═══════════════════════════════════════════════════
   LoadingAnimation.jsx
   Step-by-step progress tracker while backend runs.
   ═══════════════════════════════════════════════════ */

/* ── Checkmark icon ─────────────────────────────── */
function CheckIcon() {
  return (
    <svg
      width="11" height="11" viewBox="0 0 11 11" fill="none"
      style={{ animation: 'sv-checkpop 0.35s cubic-bezier(0.16,1,0.3,1) both' }}
    >
      <path
        d="M2 5.5l2.5 2.5L9 3"
        stroke="white" strokeWidth="1.7"
        strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  )
}

/* ── Individual step row ────────────────────────── */
function StepRow({ label, status, isLast }) {
  const done   = status === 'done'
  const active = status === 'active'

  return (
    <div
      style={{
        display       : 'flex',
        alignItems    : 'center',
        gap           : 14,
        padding       : '13px 22px',
        borderBottom  : isLast ? 'none' : '1px solid #F1F5F9',
        background    : active ? '#FAFAF8' : 'transparent',
        transition    : 'background 0.3s',
      }}
    >
      {/* Circle status */}
      <div
        style={{
          width           : 26,
          height          : 26,
          borderRadius    : '50%',
          flexShrink      : 0,
          display         : 'flex',
          alignItems      : 'center',
          justifyContent  : 'center',
          background      : done ? '#0F172A' : active ? 'transparent' : '#F1F5F9',
          border          : active ? '1.5px solid #E2E8F0' : 'none',
          transition      : 'background 0.3s, border 0.3s',
        }}
      >
        {done && <CheckIcon />}
        {active && (
          <span
            style={{
              width: 7, height: 7,
              borderRadius: '50%',
              background: '#CBD5E1',
              animation: 'sv-pulse-dot 1.2s ease-in-out infinite',
            }}
          />
        )}
        {!done && !active && (
          <span
            style={{
              width: 5, height: 5,
              borderRadius: '50%',
              background: '#E2E8F0',
            }}
          />
        )}
      </div>

      {/* Label */}
      <span
        style={{
          flex       : 1,
          fontSize   : 13.5,
          fontWeight : done ? 600 : active ? 500 : 400,
          color      : done ? '#0F172A' : active ? '#475569' : '#CBD5E1',
          transition : 'color 0.3s, font-weight 0.2s',
        }}
      >
        {label}
      </span>

      {/* Done tick */}
      {done && (
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style={{ flexShrink: 0 }}>
          <path
            d="M2.5 7l3 3 6-6"
            stroke="#059669" strokeWidth="1.6"
            strokeLinecap="round" strokeLinejoin="round"
          />
        </svg>
      )}
    </div>
  )
}

/* ── Main export ────────────────────────────────── */
export default function LoadingAnimation({ videoUrl, elapsed, model }) {
  const inferenceLabel =
    model === 'dl' ? 'Running BiLSTM inference' :
    model === 'ml' ? 'Running ML Ensemble inference' :
    'Running BiLSTM + ML inference'

  const STEPS = [
    { label: 'Fetching comments from YouTube',   threshold: 15 },
    { label: 'Preprocessing & tokenizing text',  threshold: 25 },
    { label: inferenceLabel,                     threshold: 55 },
    { label: 'Aggregating sentiment scores',     threshold: 75 },
  ]

  function getStatus(step, i) {
    const prevThreshold = i === 0 ? 0 : STEPS[i - 1].threshold
    if (elapsed > step.threshold) return 'done'
    if (elapsed >= prevThreshold)  return 'active'
    return 'pending'
  }

  return (
    <div
      style={{
        minHeight      : '100vh',
        display        : 'flex',
        flexDirection  : 'column',
        alignItems     : 'center',
        justifyContent : 'center',
        padding        : '110px 20px 80px',
      }}
    >
      <div style={{ width: '100%', maxWidth: 440, textAlign: 'center' }}>

        {/* Spinner */}
        <div
          style={{
            width        : 52,
            height       : 52,
            margin       : '0 auto 28px',
            border       : '3px solid #E4E3DB',
            borderTopColor: '#0F172A',
            borderRadius : '50%',
            animation    : 'sv-spin 0.9s linear infinite',
          }}
        />

        {/* Heading */}
        <h2
          className="font-serif"
          style={{
            fontSize    : 26,
            fontStyle   : 'italic',
            color       : '#0F172A',
            marginBottom: 6,
            lineHeight  : 1.2,
          }}
        >
          Analyzing comments…
        </h2>
        <p style={{ fontSize: 13.5, color: '#9CA3AF', marginBottom: 30, lineHeight: 1.55 }}>
          This may take a moment depending on the comment count.
        </p>

        {/* Step card */}
        <div className="card" style={{ textAlign: 'left', overflow: 'hidden', marginBottom: 18 }}>
          {STEPS.map((step, i) => (
            <StepRow
              key={i}
              label={step.label}
              status={getStatus(step, i)}
              isLast={i === STEPS.length - 1}
            />
          ))}
        </div>

        {/* Elapsed */}
        <p
          className="font-mono"
          style={{ fontSize: 12, color: '#6B7280' }}
        >
          {elapsed}s elapsed
        </p>
      </div>
    </div>
  )
}
