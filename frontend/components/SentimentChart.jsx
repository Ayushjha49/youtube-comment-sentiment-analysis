import { useState, useEffect } from 'react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  RadialBarChart, RadialBar,
} from 'recharts'

const COLORS = {
  positive: '#00e5a0',
  negative: '#ff3d6e',
  neutral : '#f0b429',
}

const GLOW_COLORS = {
  positive: 'rgba(0, 229, 160, 0.25)',
  negative: 'rgba(255, 61, 110, 0.25)',
  neutral : 'rgba(240, 180, 41, 0.25)',
}

function buildChartData(distribution) {
  return [
    { name: 'Positive', value: distribution.positive, key: 'positive' },
    { name: 'Negative', value: distribution.negative, key: 'negative' },
    { name: 'Neutral',  value: distribution.neutral,  key: 'neutral'  },
  ]
}

// ── Custom Tooltip ──────────────────────────────────────────────────────────
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const item = payload[0]
  const key  = item.payload?.key || item.dataKey
  const color = COLORS[key] || '#6c63ff'

  return (
    <div
      className="glass-card px-4 py-3"
      style={{ border: `1px solid ${color}40`, minWidth: 140 }}
    >
      <div className="font-display font-bold text-sm" style={{ color }}>
        {item.name || key}
      </div>
      <div className="font-mono text-2xl font-bold text-white mt-1">
        {item.value?.toFixed(1)}%
      </div>
    </div>
  )
}

// ── Pie chart label ──────────────────────────────────────────────────────────
function PieLabel({ cx, cy, midAngle, outerRadius, value, name, key: k }) {
  const RADIAN = Math.PI / 180
  const r = outerRadius + 30
  const x = cx + r * Math.cos(-midAngle * RADIAN)
  const y = cy + r * Math.sin(-midAngle * RADIAN)
  if (value < 5) return null
  return (
    <text
      x={x} y={y}
      fill={COLORS[k] || '#fff'}
      textAnchor={x > cx ? 'start' : 'end'}
      dominantBaseline="central"
      fontSize={12}
      fontFamily="'JetBrains Mono', monospace"
      fontWeight="500"
    >
      {`${value?.toFixed(1)}%`}
    </text>
  )
}

// ── Toggle button ────────────────────────────────────────────────────────────
function ChartToggle({ type, onChange }) {
  return (
    <div
      className="flex p-1 rounded-xl"
      style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.08)' }}
    >
      {[
        { id: 'bar',     label: '▬ Bar',  icon: '▬' },
        { id: 'pie',     label: '◕ Pie',  icon: '◕' },
        { id: 'radial',  label: '◎ Radial', icon: '◎' },
      ].map(btn => (
        <button
          key={btn.id}
          onClick={() => onChange(btn.id)}
          className="px-4 py-2 rounded-lg text-xs font-display font-semibold transition-all duration-200"
          style={{
            background : type === btn.id ? 'rgba(108,99,255,0.3)' : 'transparent',
            color      : type === btn.id ? '#fff' : 'rgba(255,255,255,0.35)',
            border     : type === btn.id ? '1px solid rgba(108,99,255,0.5)' : '1px solid transparent',
          }}
        >
          {btn.label}
        </button>
      ))}
    </div>
  )
}

// ── Animated bar fill ────────────────────────────────────────────────────────
function AnimatedBar({ data }) {
  const [widths, setWidths] = useState({ positive: 0, negative: 0, neutral: 0 })

  useEffect(() => {
    const t = setTimeout(() => {
      setWidths({
        positive: data.find(d => d.key === 'positive')?.value || 0,
        negative: data.find(d => d.key === 'negative')?.value || 0,
        neutral : data.find(d => d.key === 'neutral')?.value  || 0,
      })
    }, 100)
    return () => clearTimeout(t)
  }, [data])

  return (
    <div className="space-y-5">
      {data.map(item => (
        <div key={item.key}>
          <div className="flex justify-between items-center mb-2">
            <div className="flex items-center gap-2">
              <div
                className="w-2.5 h-2.5 rounded-full"
                style={{ background: COLORS[item.key], boxShadow: `0 0 8px ${COLORS[item.key]}` }}
              />
              <span className="font-display font-semibold text-sm text-white/80">
                {item.name}
              </span>
            </div>
            <span
              className="font-mono font-bold text-lg"
              style={{ color: COLORS[item.key] }}
            >
              {item.value.toFixed(1)}%
            </span>
          </div>
          <div className="progress-bar-track">
            <div
              className="progress-bar-fill"
              style={{
                width     : `${widths[item.key]}%`,
                background: `linear-gradient(90deg, ${COLORS[item.key]}88, ${COLORS[item.key]})`,
                boxShadow : `0 0 10px ${GLOW_COLORS[item.key]}`,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}

export default function SentimentChart({ distribution }) {
  const [chartType, setChartType] = useState('bar')
  const data = buildChartData(distribution)

  return (
    <div className="glass-card p-6 w-full animate-slide-up" style={{ animationDelay: '0.2s' }}>
      {/* Header + toggle */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
        <div>
          <h3 className="font-display font-bold text-base text-white/90">
            Sentiment Distribution
          </h3>
          <p className="text-xs text-white/30 mt-0.5 font-mono">
            % of analyzed comments
          </p>
        </div>
        <ChartToggle type={chartType} onChange={setChartType} />
      </div>

      {/* Animated Bar */}
      {chartType === 'bar' && (
        <AnimatedBar data={data} />
      )}

      {/* Recharts BarChart */}
      {chartType === 'barchart' && (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} barCategoryGap="35%">
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis
              dataKey="name"
              tick={{ fill: 'rgba(255,255,255,0.5)', fontSize: 12, fontFamily: 'DM Sans' }}
              axisLine={false} tickLine={false}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 11, fontFamily: 'JetBrains Mono' }}
              axisLine={false} tickLine={false}
              tickFormatter={v => `${v}%`}
              domain={[0, 100]}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {data.map(entry => (
                <Cell key={entry.key} fill={COLORS[entry.key]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      {/* Pie chart */}
      {chartType === 'pie' && (
        <ResponsiveContainer width="100%" height={260}>
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
                  stroke={COLORS[entry.key]}
                  strokeWidth={2}
                  style={{ filter: `drop-shadow(0 0 8px ${GLOW_COLORS[entry.key]})` }}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip />} />
            <Legend
              iconType="circle"
              iconSize={8}
              formatter={(value, entry) => (
                <span style={{ color: COLORS[entry.payload.key], fontFamily: 'DM Sans', fontSize: 13 }}>
                  {value}
                </span>
              )}
            />
          </PieChart>
        </ResponsiveContainer>
      )}

      {/* Radial bar chart */}
      {chartType === 'radial' && (
        <ResponsiveContainer width="100%" height={240}>
          <RadialBarChart
            cx="50%" cy="50%"
            innerRadius={30}
            outerRadius={100}
            data={data}
            startAngle={180}
            endAngle={-180}
          >
            <RadialBar
              minAngle={5}
              dataKey="value"
              cornerRadius={6}
              background={{ fill: 'rgba(255,255,255,0.04)' }}
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
                <span style={{ color: COLORS[entry.payload?.key], fontFamily: 'DM Sans', fontSize: 13 }}>
                  {entry.payload?.name} — {entry.payload?.value?.toFixed(1)}%
                </span>
              )}
            />
          </RadialBarChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
