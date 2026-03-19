/* ═══════════════════════════════════════════════════
   ResultsDashboard.jsx
   Full results view — verdict, chart, stats, comments.
   ═══════════════════════════════════════════════════ */

import SentimentChart from './SentimentChart'

const SC = {
  positive: {
    label    : 'Positive',
    color    : '#059669',
    bg       : '#ECFDF5',
    border   : '#A7F3D0',
    textDark : '#065F46',
  },
  negative: {
    label    : 'Negative',
    color    : '#DC2626',
    bg       : '#FEF2F2',
    border   : '#FECACA',
    textDark : '#7F1D1D',
  },
  neutral: {
    label    : 'Neutral',
    color    : '#D97706',
    bg       : '#FFFBEB',
    border   : '#FDE68A',
    textDark : '#78350F',
  },
}

/* ── Video info + verdict hero ──────────────────── */
function VerdictHero({ result }) {
  const cfg      = SC[result.overall_sentiment] || SC.neutral
  const conf     = Math.round(result.overall_confidence * 100)
  const confText = conf >= 75 ? 'High confidence'
                 : conf >= 55 ? 'Moderate confidence'
                 : 'Low confidence'

  return (
    <div
      className="card anim-u0"
      style={{
        padding : 26,
        display : 'flex',
        gap     : 22,
        flexWrap: 'wrap',
        alignItems: 'flex-start',
      }}
    >
      {/* Thumbnail */}
      {result.thumbnail ? (
        <img
          src={result.thumbnail}
          alt=""
          style={{
            width      : 108,
            height     : 74,
            objectFit  : 'cover',
            borderRadius: 10,
            flexShrink : 0,
            border     : '1px solid #E4E3DB',
          }}
        />
      ) : (
        <div
          style={{
            width          : 108,
            height         : 74,
            borderRadius   : 10,
            background     : '#FF0000',
            display        : 'flex',
            alignItems     : 'center',
            justifyContent : 'center',
            flexShrink     : 0,
          }}
        >
          <svg width="26" height="26" viewBox="0 0 24 24" fill="white">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      )}

      {/* Title + channel */}
      <div style={{ flex: 1, minWidth: 180 }}>
        <div className="label-caps" style={{ marginBottom: 6 }}>
          Analyzed Video
        </div>
        <div
          style={{
            fontSize    : 17,
            fontWeight  : 700,
            color       : '#0F172A',
            lineHeight  : 1.4,
            marginBottom: 5,
          }}
        >
          {result.video_title || 'YouTube Video'}
        </div>
        {result.channel && (
          <div style={{ fontSize: 13, color: '#6B7280' }}>
            {result.channel}
          </div>
        )}
      </div>

      {/* Verdict pill */}
      <div
        style={{
          padding      : '18px 26px',
          borderRadius : 16,
          background   : cfg.bg,
          border       : `1.5px solid ${cfg.border}`,
          textAlign    : 'center',
          flexShrink   : 0,
          minWidth     : 152,
        }}
      >
        <div className="label-caps" style={{ marginBottom: 7 }}>
          Overall Verdict
        </div>
        <div
          style={{
            fontSize     : 26,
            fontWeight   : 800,
            color        : cfg.color,
            letterSpacing: '-0.03em',
            lineHeight   : 1,
          }}
        >
          {cfg.label}
        </div>
        <div
          className="font-mono"
          style={{
            fontSize  : 12,
            color     : cfg.color,
            marginTop : 6,
            opacity   : 0.72,
          }}
        >
          {conf}% · {confText}
        </div>
      </div>
    </div>
  )
}

/* ── Stats grid card ────────────────────────────── */
function StatItem({ label, value, sub }) {
  return (
    <div
      style={{
        padding     : '14px 16px',
        borderRadius: 12,
        background  : '#F8FAFC',
        border      : '1px solid #F1F5F9',
      }}
    >
      <div className="label-caps" style={{ marginBottom: 5 }}>
        {label}
      </div>
      <div
        className="font-mono"
        style={{
          fontSize     : 21,
          fontWeight   : 700,
          color        : '#0F172A',
          letterSpacing: '-0.025em',
        }}
      >
        {value}
      </div>
      {sub && (
        <div style={{ fontSize: 11, color: '#9CA3AF', marginTop: 2 }}>
          {sub}
        </div>
      )}
    </div>
  )
}

function StatsCard({ result }) {
  const modelLabel =
    result.model_used === 'ml+dl_ensemble' ? 'ML + DL Ensemble'
    : result.model_used === 'dl_bilstm'    ? 'BiLSTM Only'
    : result.model_used === 'ml_ensemble'  ? 'ML Ensemble Only'
    : result.model_used

  const coverage = Math.round(
    (result.analyzed_count / result.total_comments_video) * 100
  )

  return (
    <div className="card anim-u2" style={{ padding: 26 }}>
      <div
        style={{
          fontSize    : 15,
          fontWeight  : 700,
          color       : '#0F172A',
          marginBottom: 18,
        }}
      >
        Analysis Stats
      </div>

      <div
        style={{
          display              : 'grid',
          gridTemplateColumns  : '1fr 1fr',
          gap                  : 12,
          marginBottom         : 14,
        }}
      >
        <StatItem
          label="Analyzed"
          value={result.analyzed_count.toLocaleString()}
          sub="comments"
        />
        <StatItem
          label="Total on Video"
          value={result.total_comments_video.toLocaleString()}
          sub="comments"
        />
        <StatItem
          label="Model"
          value={modelLabel}
          sub="prediction engine"
        />
        <StatItem
          label="Processing Time"
          value={`${result.processing_time_s.toFixed(1)}s`}
          sub="fetch + predict"
        />
      </div>

      {/* Coverage bar */}
      <div
        style={{
          padding     : '12px 16px',
          borderRadius: 12,
          background  : '#F8FAFC',
          border      : '1px solid #F1F5F9',
          display     : 'flex',
          alignItems  : 'center',
          justifyContent: 'space-between',
          gap         : 12,
        }}
      >
        <div>
          <div className="label-caps" style={{ marginBottom: 3 }}>
            Coverage
          </div>
          <div style={{ fontSize: 12, color: '#6B7280' }}>
            Comments analyzed vs total
          </div>
        </div>
        <div
          className="font-mono"
          style={{
            fontSize     : 20,
            fontWeight   : 700,
            color        : '#0F172A',
            letterSpacing: '-0.025em',
            flexShrink   : 0,
          }}
        >
          {coverage}%
        </div>
      </div>
    </div>
  )
}

/* ── Comment chip ───────────────────────────────── */
function CommentChip({ text, type }) {
  const c = SC[type] || SC.neutral
  return (
    <div
      style={{
        padding     : '10px 14px',
        borderRadius: '0 10px 10px 0',
        background  : c.bg,
        borderLeft  : `3px solid ${c.color}`,
        fontSize    : 13,
        color       : '#334155',
        lineHeight  : 1.55,
        wordBreak   : 'break-word',
      }}
    >
      {text}
    </div>
  )
}

/* ── Comments section ───────────────────────────── */
function CommentsSection({ top_positive, top_negative }) {
  if (!top_positive?.length && !top_negative?.length) return null

  const cols = [
    { list: top_positive, type: 'positive', heading: 'Most Positive' },
    { list: top_negative, type: 'negative', heading: 'Most Negative' },
  ]

  return (
    <div className="card anim-u3" style={{ padding: 26 }}>
      <div
        style={{
          fontSize    : 15,
          fontWeight  : 700,
          color       : '#0F172A',
          marginBottom: 22,
        }}
      >
        Sample Comments
      </div>

      <div
        style={{
          display             : 'grid',
          gridTemplateColumns : '1fr 1fr',
          gap                 : 28,
        }}
        className="sm-stack"
      >
        {cols.map(({ list, type, heading }) => {
          const c = SC[type]
          return (
            <div key={type}>
              {/* Column heading */}
              <div
                style={{
                  display     : 'flex',
                  alignItems  : 'center',
                  gap         : 7,
                  marginBottom: 12,
                }}
              >
                <div
                  style={{
                    width: 7, height: 7,
                    borderRadius: '50%',
                    background: c.color,
                    flexShrink: 0,
                  }}
                />
                <span
                  style={{
                    fontSize      : 10.5,
                    fontWeight    : 700,
                    color         : c.color,
                    textTransform : 'uppercase',
                    letterSpacing : '0.08em',
                  }}
                >
                  {heading}
                </span>
              </div>

              {/* Chips */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
                {list?.slice(0, 3).map((text, i) => (
                  <CommentChip key={i} text={text} type={type} />
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

/* ── Main export ────────────────────────────────── */
export default function ResultsDashboard({ result, onReset }) {
  const { distribution, top_positive, top_negative } = result

  return (
    <div
      style={{
        maxWidth: 900,
        margin  : '0 auto',
        padding : '90px 20px 80px',
        display : 'flex',
        flexDirection: 'column',
        gap     : 18,
      }}
    >
      {/* Verdict hero */}
      <VerdictHero result={result} />

      {/* Two-column: chart + stats */}
      <div
        style={{
          display             : 'grid',
          gridTemplateColumns : 'minmax(0,1.1fr) minmax(0,0.9fr)',
          gap                 : 18,
        }}
        className="sm-stack anim-u1"
      >
        <SentimentChart distribution={distribution} />
        <StatsCard result={result} />
      </div>

      {/* Comments */}
      <CommentsSection
        top_positive={top_positive}
        top_negative={top_negative}
      />

      {/* Reset CTA */}
      <div className="anim-u4" style={{ textAlign: 'center', paddingTop: 6 }}>
        <button
          className="btn-primary"
          onClick={onReset}
          style={{ padding: '14px 32px', fontSize: 14 }}
        >
          ← Analyze another video
        </button>
        <p
          className="font-mono"
          style={{ fontSize: 11.5, color: '#9CA3AF', marginTop: 14 }}
        >
          Powered by BiLSTM + Attention + ML Ensemble · EN / NE / HI
        </p>
      </div>
    </div>
  )
}
