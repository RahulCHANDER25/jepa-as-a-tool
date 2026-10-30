import mdx from "@astrojs/mdx";
import { defineConfig } from "astro/config";

// The site lives at the root. GitHub Pages serves it under /<repo>/: the pages workflow passes
// that prefix with `astro build --base`.
export default defineConfig({
  site: "https://rahulchander25.github.io",
  integrations: [mdx()],
});
