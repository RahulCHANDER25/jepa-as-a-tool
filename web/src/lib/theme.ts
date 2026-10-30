// Light/dark switch. The inline script in Article.astro sets data-theme before first paint.

const root = document.documentElement;

export function mountThemeToggle() {
  const button = document.querySelector<HTMLButtonElement>("[data-theme-toggle]");
  const label = button?.querySelector<HTMLElement>("[data-theme-label]");
  if (!button || !label) return;
  const sync = () => (label.textContent = root.dataset.theme === "dark" ? "Light" : "Dark");
  button.addEventListener("click", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      // private mode or blocked storage: the switch still works for this visit
    }
  });
  onThemeChange(sync);
  sync();
}

/** Canvases read colors when they draw, so they redraw when the theme flips. */
export function onThemeChange(callback: () => void) {
  new MutationObserver(callback).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
}
