import mdx from "@astrojs/mdx";
import { defineConfig } from "astro/config";

// The site lives at the root. GitHub Pages serves it under /<repo>/: the pages workflow passes
// that prefix with `astro build --base`.
export default defineConfig({
  site: "https://rahulchander25.github.io",
  integrations: [mdx()],
  // Bundled when the dev server starts: found later, Vite re-bundles it and the page's import of
  // the old bundle fails ("error loading dynamically imported module")
  vite: { optimizeDeps: { include: ["onnxruntime-web"] } },
});
