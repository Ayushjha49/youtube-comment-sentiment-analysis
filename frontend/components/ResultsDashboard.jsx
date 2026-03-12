import SentimentChart from './SentimentChart'

const SENTIMENT_CONFIG = {
  positive: {
    color  : '#00e5a0',
    glow   : 'rgba(0, 229, 160, 0.2)',
    border : 'rgba(0, 229, 160, 0.3)',
    bg     : 'rgba(0, 229, 160, 0.06)',
    emoji  : '😊',
    label  : 'Positive',
    gradient: 'linear-gradient(135deg, rgba(0,229,160,0.15) 0%, rgba(0,229,160,0.05) 100%)',
  },
  negative: {
    color  : '#ff3d6e',
    glow   : 'rgba(255, 61, 110, 0.2)',
    border : 'rgba(255, 61, 110, 0.3)',
    bg     : 'rgba(255, 61, 110, 0.06)',
    emoji  : '😞',
    label  : 'Negative',
    gradient: 'linear-gradient(135deg, rgba(255,61,110,0.15) 0%, rgba(255,61,110,0.05) 100%)',
  },
  neutral: {
    color  : '#f0b429',
    glow   : 'rgba(240, 180, 41, 0.2)',
    border : 'rgba(240, 180, 41, 0.3)',
    bg     : 'rgba(240, 180, 41, 0.06)',
    emoji  : '😐',
    label  : 'Neutral',
    gradient: 'linear-gradient(135deg, rgba(240,180,41,0.15) 0%, rgba(240,180,41,0.05) 100%)',
  },
}

// ── Overall Sentiment Hero Banner ───────────────────────────────────────────
function OverallSentimentBanner({ sentiment, confidence, videoTitle, channel, thumbnail }) {
  const cfg = SENTIMENT_CONFIG[sentiment] || SENTIMENT_CONFIG.neutral

  const confidencePct = Math.round(confidence * 100)
  const confidenceLabel =
    confidencePct >= 70 ? 'High confidence' :
    confidencePct >= 50 ? 'Moderate confidence' :
    'Low confidence'

  return (
    <div
      className="glass-card relative overflow-hidden p-8 animate-slide-up"
      style={{ background: cfg.gradient, borderColor: cfg.border }}
    >
      {/* Glow orb */}
      <div
        className="absolute -top-20 -right-20 w-64 h-64 rounded-full blur-3xl pointer-events-none"
        style={{ background: cfg.glow, opacity: 0.5 }}
      />
      <div
        className="absolute -bottom-20 -left-20 w-48 h-48 rounded-full blur-3xl pointer-events-none"
        style={{ background: cfg.glow, opacity: 0.3 }}
      />

      <div className="relative z-10">
        {/* Video info */}
        {videoTitle && (
          <div className="flex items-start gap-4 mb-6">
            {thumbnail && (
              <img
                src={thumbnail}
                alt="Thumbnail"
                className="w-20 h-14 object-cover rounded-lg shrink-0"
                style={{ border: `1px solid ${cfg.border}` }}
              />
            )}
            <div className="min-w-0">
              <p className="font-display font-bold text-white text-base leading-snug line-clamp-2">
                {videoTitle}
              </p>
              {channel && (
                <p className="text-sm text-white/40 mt-1 font-mono">{channel}</p>
              )}
            </div>
          </div>
        )}

        {/* Overall result */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <p className="font-mono text-xs tracking-widest uppercase text-white/30 mb-2">
              Overall Sentiment
            </p>
            <div className="flex items-center gap-3">
              <span className="text-4xl">{cfg.emoji}</span>
              <h2
                className="font-display font-black text-5xl sm:text-6xl uppercase tracking-tight"
                style={{ color: cfg.color, textShadow: `0 0 40px ${cfg.glow}` }}
              >
                {cfg.label}
              </h2>
            </div>
          </div>

          {/* Confidence meter */}
          <div
            className="p-4 rounded-2xl text-center min-w-[130px]"
            style={{ background: 'rgba(0,0,0,0.3)', border: `1px solid ${cfg.border}` }}
          >
            <div
              className="font-mono font-black text-4xl"
              style={{ color: cfg.color }}
            >
              {confidencePct}%
            </div>
            <div className="font-display text-xs text-white/40 mt-1">
              {confidenceLabel}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Stat card ───────────────────────────────────────────────────────────────
function StatCard({ label, value, sub, color }) {
  return (
    <div
      className="glass-card-light p-4 flex flex-col gap-1"
      style={{ borderColor: color ? `${color}22` : undefined }}
    >
      <span className="font-mono text-xs text-white/25 uppercase tracking-widest">{label}</span>
      <span
        className="font-display font-bold text-2xl"
        style={{ color: color || '#fff' }}
      >
        {value}
      </span>
      {sub && <span className="font-mono text-xs text-white/25">{sub}</span>}
    </div>
  )
}

// ── Per-sentiment percentage card ────────────────────────────────────────────
function SentimentPercentCard({ sentiment, pct, animDelay = 0 }) {
  const cfg = SENTIMENT_CONFIG[sentiment]
  return (
    <div
      className="glass-card p-5 flex flex-col items-center gap-2 animate-slide-up"
      style={{
        background    : cfg.bg,
        borderColor   : cfg.border,
        animationDelay: `${animDelay}ms`,
      }}
    >
      <span className="text-2xl">{cfg.emoji}</span>
      <span
        className="font-display font-black text-3xl"
        style={{ color: cfg.color, textShadow: `0 0 20px ${cfg.glow}` }}
      >
        {pct.toFixed(1)}%
      </span>
      <span className="font-display font-semibold text-xs text-white/50 uppercase tracking-wider">
        {cfg.label}
      </span>
      {/* Mini bar */}
      <div className="w-full mt-1 progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{
            width     : `${pct}%`,
            background: cfg.color,
            boxShadow : `0 0 8px ${cfg.glow}`,
          }}
        />
      </div>
    </div>
  )
}

// ── Sample comments ──────────────────────────────────────────────────────────
function CommentChip({ text, sentiment }) {
  const cfg = SENTIMENT_CONFIG[sentiment]
  return (
    <div
      className="px-3 py-2 rounded-lg text-sm text-white/70 font-body line-clamp-2"
      style={{
        background : cfg.bg,
        borderLeft : `3px solid ${cfg.color}`,
      }}
    >
      {text}
    </div>
  )
}

// ── Main results dashboard ───────────────────────────────────────────────────
export default function ResultsDashboard({ result, onReset }) {
  const { distribution, top_positive, top_negative } = result

  return (
    <div className="w-full max-w-3xl mx-auto space-y-5">

      {/* Overall sentiment */}
      <OverallSentimentBanner
        sentiment   = {result.overall_sentiment}
        confidence  = {result.overall_confidence}
        videoTitle  = {result.video_title}
        channel     = {result.channel}
        thumbnail   = {result.thumbnail}
      />

      {/* 3 percentage cards */}
      <div className="grid grid-cols-3 gap-3">
        <SentimentPercentCard sentiment="positive" pct={distribution.positive} animDelay={50}  />
        <SentimentPercentCard sentiment="negative" pct={distribution.negative} animDelay={100} />
        <SentimentPercentCard sentiment="neutral"  pct={distribution.neutral}  animDelay={150} />
      </div>

      {/* Chart */}
      <SentimentChart distribution={distribution} />

      {/* Stats row */}
      <div
        className="grid grid-cols-2 sm:grid-cols-4 gap-3 animate-slide-up"
        style={{ animationDelay: '0.3s' }}
      >
        <StatCard
          label="Analyzed"
          value={result.analyzed_count.toLocaleString()}
          sub="comments"
        />
        <StatCard
          label="Total Comments"
          value={result.total_comments_video.toLocaleString()}
          sub="on video"
        />
        <StatCard
          label="Model"
          value={result.model_used.replace('ml+dl_', '').replace('_', ' ')}
          sub={result.model_used}
        />
        <StatCard
          label="Time"
          value={`${result.processing_time_s.toFixed(1)}s`}
          sub="processing"
        />
      </div>

      {/* Sample comments */}
      {(top_positive?.length > 0 || top_negative?.length > 0) && (
        <div
          className="glass-card p-5 animate-slide-up"
          style={{ animationDelay: '0.4s' }}
        >
          <h3 className="font-display font-bold text-sm text-white/70 mb-4">
            Sample Comments
          </h3>
          <div className="grid sm:grid-cols-2 gap-3">
            {/* Positive samples */}
            <div>
              <p className="font-mono text-xs text-white/25 uppercase tracking-widest mb-2">
                😊 Most Positive
              </p>
              <div className="space-y-2">
                {top_positive?.slice(0, 3).map((text, i) => (
                  <CommentChip key={i} text={text} sentiment="positive" />
                ))}
              </div>
            </div>
            {/* Negative samples */}
            <div>
              <p className="font-mono text-xs text-white/25 uppercase tracking-widest mb-2">
                😞 Most Negative
              </p>
              <div className="space-y-2">
                {top_negative?.slice(0, 3).map((text, i) => (
                  <CommentChip key={i} text={text} sentiment="negative" />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Reset button */}
      <div className="flex justify-center pt-2 pb-8">
        <button
          onClick={onReset}
          className="font-display font-semibold text-sm text-white/30 hover:text-white/70 transition-colors duration-200 flex items-center gap-2 px-5 py-3 rounded-xl"
          style={{ border: '1px solid rgba(255,255,255,0.08)' }}
        >
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path
              d="M12 7A5 5 0 1 1 7 2M7 2L5 4M7 2L9 4"
              stroke="currentColor" strokeWidth="1.5"
              strokeLinecap="round" strokeLinejoin="round"
            />
          </svg>
          Analyze another video
        </button>
      </div>
    </div>
  )
}
