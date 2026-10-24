/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        inter: ["var(--font-inter)", "Inter", "sans-serif"],
      },
      colors: {
        navy:     "#142849",
        brand:    "#3333FF",
        positive: "#2C8833",
        warning:  "#9D803E",
        danger:   "#C05852",
        purple:   "#6050B8",
        blueMid:  "#2B5CAE",
        muted:    "#75787D",
        sidebar:  "#A3AEC2",
        border:   "#DADADA",
        bg:       "#F9FAFD",
      },
    },
  },
  plugins: [],
};
