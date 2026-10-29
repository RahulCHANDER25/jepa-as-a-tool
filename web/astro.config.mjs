import mdx from "@astrojs/mdx";
import { defineConfig } from "astro/config";

// GitHub Pages serves the site under /<repo>/
export default defineConfig({
  site: "https://rahulchander25.github.io",
  base: "/jepa-as-a-tool",
  integrations: [mdx()],
});
