/* ═══════════════════════════════════════════════════
   SentimentChart.jsx
   All 3 chart types: Bar (horizontal) · Pie · Radial
   ═══════════════════════════════════════════════════ */

import { useState, useEffect } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  RadialBarChart, RadialBar,
} from 'recharts'

const BARS = [
  { key: 'positive', label: 'Positive', color: '#059669', bg: '#D1FAE5' },
  { key: 'negative', label: 'Negative', color: '#DC2626', bg: '#FEE2E2' },
  { key: 'neutral',  label: 'Neutral',  color: '#D97706', bg: '#FEF3C7' },
]

const COLORS = {
  positive: '#059669',
  negative: '#DC2626',
  neutral : '#D97706',
}

function buildChartData(distribution) {
  return [
    { name: 'Positive', value: distribution.positive, key: 'positive' },
    { name: 'Negative', value: distribution.negative, key: 'negative' },
    { name: 'Neutral',  value: distribution.neutral,  key: 'neutral'  },
  ]
}

/* ── Chart type toggle ──────────────────────────── */
function ChartToggle({ type, onChange }) {
  const opts = [
    { id: 'bar',    label: '▬ Bar'    },
    { id: 'pie',    label: '◕ Pie'    },
    { id: 'radial', label: '◎ Radial' },
  ]
  return (
    <div
      style={{
        display    : 'flex',
        padding    : 3,
        borderRadius: 10,
        background : '#F1F5F9',
        border     : '1px solid #E4E3DB',
        gap        : 2,
      }}
    >
      {opts.map(btn => (
        <button
          key={btn.id}
          onClick={() => onChange(btn.id)}
          style={{
            padding     : '5px 12px',
            borderRadius: 7,
            border      : type === btn.id ? '1px solid #CBD5E1' : '1px solid transparent',
            background  : type === btn.id ? '#FFFFFF' : 'transparent',
            color       : type === btn.id ? '#0F172A' : '#9CA3AF',
            fontSize    : 12,
            fontWeight  : type === btn.id ? 600 : 500,
            cursor      : 'pointer',
            fontFamily  : "'Plus Jakarta Sans', sans-serif",
            transition  : 'all 0.15s',
            whiteSpace  : 'nowrap',
            boxShadow   : type === btn.id ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
          }}
        >
          {btn.label}
        </button>
      ))}
    </div>
  )
}

/* ── Custom tooltip (shared by Pie + Radial) ─────── */
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const item  = payload[0]
  const key   = item.payload?.key || item.dataKey
  const color = COLORS[key] || '#334155'

  return (
    <div
      style={{
        background  : '#FFFFFF',
        border      : `1px solid ${color}40`,
        borderRadius: 10,
        padding     : '10px 14px',
        minWidth    : 120,
        boxShadow   : '0 4px 16px rgba(0,0,0,0.10)',
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 600, color: '#6B7280', marginBottom: 3 }}>
        {item.name || key}
      </div>
      <div
        style={{
          fontFamily   : "'JetBrains Mono', monospace",
          fontSize     : 20,
          fontWeight   : 700,
          color,
          letterSpacing: '-0.02em',
        }}
      >
        {item.value?.toFixed(1)}%
      </div>
    </div>
  )
}

/* ── Pie label ──────────────────────────────────── */
function PieLabel({ cx, cy, midAngle, outerRadius, value, payload }) {
  const RADIAN = Math.PI / 180
  const r = outerRadius + 28
  const x = cx + r * Math.cos(-midAngle * RADIAN)
  const y = cy + r * Math.sin(-midAngle * RADIAN)
  if (value < 5) return null
  return (
    <text
      x={x} y={y}
      fill={COLORS[payload.key] || '#334155'}
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      fontSize={12}
      fontFamily="'JetBrains Mono', monospace"
      fontWeight="600"
    >
      {value?.toFixed(1)}%
    </text>
  )
}

/* ── Horizontal animated bar ─────────────────────── */
function HBar({ label, value, color, bg, delay }) {
  const [width, setWidth] = useState(0)

  useEffect(() => {
    const t = setTimeout(() => setWidth(value), 120 + delay)
    return () => clearTimeout(t)
  }, [value, delay])

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
      <div style={{ width: 80, flexShrink: 0 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: '#334155', marginBottom: 2 }}>
          {label}
        </div>
        <div
          className="font-mono"
          style={{ fontSize: 14, fontWeight: 700, color, letterSpacing: '-0.02em' }}
        >
          {value.toFixed(1)}%
        </div>
      </div>

      <div
        style={{
          flex        : 1,
          height      : 42,
          background  : bg,
          borderRadius: 10,
          overflow    : 'hidden',
          position    : 'relative',
        }}
      >
        <div
          style={{
            position    : 'absolute',
            left: 0, top: 0, bottom: 0,
            width       : `${width}%`,
            background  : color,
            borderRadius: 10,
            opacity     : 0.78,
            transition  : `width 1.1s cubic-bezier(0.16,1,0.3,1) ${delay}ms`,
          }}
        />
        <div
          style={{
            position      : 'absolute',
            left          : 14,
            top           : '50%',
            transform     : 'translateY(-50%)',
            fontFamily    : "'JetBrains Mono', monospace",
            fontSize      : 12,
            fontWeight    : 700,
            color         : 'white',
            letterSpacing : '-0.02em',
            pointerEvents : 'none',
            opacity       : width > 18 ? 1 : 0,
            transition    : `opacity 0.4s ${delay + 600}ms`,
          }}
        >
          {value.toFixed(1)}%
        </div>
      </div>

      <div
        style={{
          width          : 52,
          height         : 28,
          display        : 'flex',
          alignItems     : 'center',
          justifyContent : 'center',
          borderRadius   : 6,
          background     : bg,
          border         : `1px solid ${color}30`,
          flexShrink     : 0,
        }}
      >
        <span className="font-mono" style={{ fontSize: 11.5, fontWeight: 700, color }}>
          {value.toFixed(1)}%
        </span>
      </div>
    </div>
  )
}

/* ── Main export ────────────────────────────────── */
export default function SentimentChart({ distribution }) {
  const [chartType, setChartType] = useState('bar')
  const data = buildChartData(distribution)

  const dominant = BARS.reduce((a, b) =>
    distribution[a.key] > distribution[b.key] ? a : b
  )

  return (
    <div className="card" style={{ padding: 26 }}>

      {/* Header row */}
      <div
        style={{
          display        : 'flex',
          alignItems     : 'flex-start',
          justifyContent : 'space-between',
          marginBottom   : 24,
          gap            : 12,
          flexWrap       : 'wrap',
        }}
      >
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: '#0F172A', marginBottom: 3 }}>
            Sentiment Distribution
          </div>
          <div style={{ fontSize: 12.5, color: '#9CA3AF' }}>
            % of analyzed comments
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          <div
            style={{
              padding     : '5px 12px',
              borderRadius: 30,
              background  : dominant.bg,
              border      : `1px solid ${dominant.color}40`,
              display     : 'flex',
              alignItems  : 'center',
              gap         : 6,
            }}
          >
            <span
              style={{
                width: 6, height: 6, borderRadius: '50%',
                background: dominant.color, display: 'inline-block',
              }}
            />
            <span style={{ fontSize: 11.5, fontWeight: 600, color: dominant.color }}>
              {dominant.label} dominant
            </span>
          </div>
          <ChartToggle type={chartType} onChange={setChartType} />
        </div>
      </div>

      {/* ── Bar view ── */}
      {chartType === 'bar' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {BARS.map((bar, i) => (
            <HBar
              key={bar.key}
              label={bar.label}
              value={distribution[bar.key]}
              color={bar.color}
              bg={bar.bg}
              delay={i * 160}
            />
          ))}
        </div>
      )}

      {/* ── Pie view ── */}
      {chartType === 'pie' && (
        <ResponsiveContainer width="100%" height={270}>
          <PieChart>
            <Pie
              data={data}
              cx="50%" cy="50%"
              innerRadius={55}
              outerRadius={95}
              paddingAngle={4}
              dataKey="value"
              labelLine={false}
              label={<PieLabel />}
              animationBegin={0}
              animationDuration={900}
            >
              {data.map(entry => (
                <Cell
                  key={entry.key}
                  fill={COLORS[entry.key]}
                  stroke="white"
                  strokeWidth={2}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              iconType="circle"
              iconSize={8}
              formatter={(value, entry) => (
                <span
                  style={{
                    color     : COLORS[entry.payload.key],
                    fontFamily: "'Plus Jakarta Sans', sans-serif",
                    fontSize  : 13,
                    fontWeight: 500,
                  }}
                >
                  {value}
                </span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      )}

      {/* ── Radial view ── */}
      {chartType === 'radial' && (
        <ResponsiveContainer width="100%" height={250}>
          <RadialBarChart
            cx="50%" cy="50%"
            innerRadius={30}
            outerRadius={105}
            data={data}
            startAngle={180}
            endAngle={-180}
          >
            <RadialBar
              minAngle={5}
              dataKey="value"
              cornerRadius={6}
              background={{ fill: '#F1F5F9' }}
            >
              {data.map(entry => (
                <Cell key={entry.key} fill={COLORS[entry.key]} />
              ))}
            </RadialBar>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              iconType="circle"
              iconSize={8}
              formatter={(value, entry) => (
                <span
                  style={{
                    color     : COLORS[entry.payload?.key],
                    fontFamily: "'Plus Jakarta Sans', sans-serif",
                    fontSize  : 13,
                    fontWeight: 500,
                  }}
                >
                  {entry.payload?.name} — {entry.payload?.value?.toFixed(1)}%
                </span>
              )}
            />
          </RadialBarChart>
        </ResponsiveContainer>
      )}

      {/* Legend footer (bar view only) */}
      {chartType === 'bar' && (
        <div
          style={{
            marginTop : 22,
            paddingTop: 16,
            borderTop : '1px solid #F1F5F9',
            display   : 'flex',
            gap       : 18,
            flexWrap  : 'wrap',
          }}
        >
          {BARS.map(bar => (
            <div key={bar.key} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <div
                style={{
                  width: 8, height: 8,
                  borderRadius: '50%',
                  background: bar.color,
                  flexShrink: 0,
                }}
              />
              <span style={{ fontSize: 12, color: '#6B7280', fontWeight: 500 }}>
                {bar.label}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
