/**
 * Design tokens — single source of truth for colours used across the app.
 * Import these wherever you need colour values in TypeScript/TSX.
 */

export const colors = {
  // Brand / sidebar
  navy:        "#142849",
  brand:       "#3333FF",
  brandLight:  "#F1EFFD",

  // Neutrals
  white:       "#FFFFFF",
  bg:          "#F9FAFD",
  border:      "#DADADA",
  borderLight: "#D9D9D9",
  textPrimary: "#1A1A2E",
  textMuted:   "#75787D",
  textMuted60: "rgba(117,120,125,0.6)",
  iconMid:     "#A5AAB5",
  sidebarText: "#A3AEC2",

  // Semantic
  positive:    "#2C8833",
  positiveBg:  "#E9F5F1",
  positiveBg2: "#E9F9F4",
  positiveBorder: "#C4F0E3",

  warning:     "#9D803E",
  warningBg:   "#FEF4DD",

  danger:      "#C05852",
  dangerBg:    "#FEEBEA",

  purple:      "#6050B8",
  purpleBg:    "#F1EFFD",

  // Tints used in cards
  blueTint:    "#EAF3FE",
  blueTint2:   "#ECF4FE",
  blueMid:     "#2B5CAE",
} as const;

export type ColorKey = keyof typeof colors;
