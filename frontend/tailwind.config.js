/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        serif : ['Cormorant Garamond', 'Garamond', 'Georgia', 'Times New Roman', 'serif'],
        sans  : ['Plus Jakarta Sans', 'system-ui', '-apple-system', 'sans-serif'],
        mono  : ['JetBrains Mono', 'Cascadia Code', 'Fira Code', 'monospace'],
      },
      colors: {
        cream    : '#F4F3ED',
        ink      : '#0F172A',
        positive : '#059669',
        negative : '#DC2626',
        neutral  : '#D97706',
      },
      animation: {
        'spin-smooth': 'sv-spin 0.85s linear infinite',
        'fadeup'     : 'sv-fadeup 0.5s cubic-bezier(0.16,1,0.3,1) both',
        'pulse-dot'  : 'sv-pulse-dot 1.8s ease-in-out infinite',
      },
      keyframes: {
        'sv-spin'    : { to: { transform: 'rotate(360deg)' } },
        'sv-fadeup'  : {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to  : { opacity: '1', transform: 'translateY(0)' },
        },
        'sv-pulse-dot': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%'      : { opacity: '0.4', transform: 'scale(1.45)' },
        },
      },
    },
  },
  plugins: [],
}
