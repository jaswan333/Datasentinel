/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--border) / <alpha-value>)",
        background: "hsl(var(--background) / <alpha-value>)",
        foreground: "hsl(var(--foreground) / <alpha-value>)",
        navy: {
          950: "#070c18",
          900: "#0b1120",
          850: "#0f172a",
          800: "#1e293b",
          700: "#334155",
        },
        primary: {
          DEFAULT: "#2563eb",
          hover: "#1d4ed8",
          light: "#dbeafe",
          foreground: "#ffffff",
        },
        teal: {
          DEFAULT: "#0d9488",
          hover: "#0f766e",
          light: "#ccfbf1",
          foreground: "#ffffff",
        },
        amber: {
          DEFAULT: "#d97706",
          hover: "#b45309",
          light: "#fef3c7",
          foreground: "#78350f",
        },
        rose: {
          DEFAULT: "#dc2626",
          hover: "#b91c1c",
          light: "#fee2e2",
          foreground: "#991b1b",
        },
        muted: {
          DEFAULT: "#f1f5f9",
          foreground: "#64748b",
        },
        surface: {
          DEFAULT: "#ffffff",
          subtle: "#f8fafc",
          dark: "#0f172a",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "-apple-system",
          "BlinkMacSystemFont",
          "'Segoe UI'",
          "Roboto",
          "Oxygen",
          "Ubuntu",
          "Cantarell",
          "sans-serif",
        ],
        mono: [
          "'JetBrains Mono'",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};
