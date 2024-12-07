import type { Config } from 'tailwindcss';
import daisyui from "daisyui";

export default {
  content: ['./src/**/*.{html,js,svelte,ts}'],
  darkMode: 'class',
  theme: {
    extend: {}

  },

  daisyui: {
    themes: ["light", "dark"],
  },
  plugins: [daisyui]
} satisfies Config;
