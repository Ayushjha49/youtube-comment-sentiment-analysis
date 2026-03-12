/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        display : ['Syne', 'sans-serif'],
        body    : ['DM Sans', 'sans-serif'],
        mono    : ['JetBrains Mono', 'monospace'],
      },
      colors: {
        void     : '#04040d',
        surface  : '#0b0b1a',
        card     : '#10102a',
        border   : '#1e1e42',
        positive : '#00e5a0',
        negative : '#ff3d6e',
        neutral  : '#f0b429',
        accent   : '#6c63ff',
        glow     : '#6c63ff',
      },
      animation: {
        'pulse-slow'   : 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float'        : 'float 6s ease-in-out infinite',
        'shimmer'      : 'shimmer 2s linear infinite',
        'scan'         : 'scan 2s linear infinite',
        'glow-pulse'   : 'glow-pulse 2s ease-in-out infinite',
        'slide-up'     : 'slide-up 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in'      : 'fade-in 0.4s ease-out',
        'bar-fill'     : 'bar-fill 1s cubic-bezier(0.16, 1, 0.3, 1)',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%'     : { transform: 'translateY(-10px)' },
        },
        shimmer: {
          '0%'  : { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        scan: {
          '0%'  : { top: '0%', opacity: '1' },
          '100%': { top: '100%', opacity: '0' },
        },
        'glow-pulse': {
          '0%, 100%': { opacity: '0.6' },
          '50%'     : { opacity: '1' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(30px)' },
          to  : { opacity: '1', transform: 'translateY(0)' },
        },
        'fade-in': {
          from: { opacity: '0' },
          to  : { opacity: '1' },
        },
        'bar-fill': {
          from: { width: '0%' },
          to  : { width: 'var(--target-width)' },
        },
      },
      backgroundImage: {
        'grid-pattern': `
          linear-gradient(rgba(108,99,255,0.05) 1px, transparent 1px),
          linear-gradient(90deg, rgba(108,99,255,0.05) 1px, transparent 1px)
        `,
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      },
      backgroundSize: {
        'grid': '40px 40px',
      },
      boxShadow: {
        'glow-sm'  : '0 0 20px rgba(108,99,255,0.15)',
        'glow-md'  : '0 0 40px rgba(108,99,255,0.2)',
        'glow-pos' : '0 0 30px rgba(0,229,160,0.15)',
        'glow-neg' : '0 0 30px rgba(255,61,110,0.15)',
        'inner-glow': 'inset 0 0 30px rgba(108,99,255,0.05)',
      },
    },
  },
  plugins: [],
}
