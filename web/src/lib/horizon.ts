// Line chart drawn at the container's real width, so text stays readable on phones.

import { hideTip, showTip } from "./tooltip";

interface Row {
  horizon: number;
  within_tolerance: number;
  px_median: number;
  deg_median: number;
}
interface Model {
  name: string;
  rows: Row[];
}

const SVG = "http://www.w3.org/2000/svg";
const HEIGHT = 280;
const M = { top: 16, right: 44, bottom: 40, left: 44 };

function el<K extends keyof SVGElementTagNameMap>(name: K, attrs: Record<string, string | number>, parent: Element) {
  const node = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  parent.append(node);
  return node;
}

function draw(box: HTMLElement, models: Model[]) {
  const width = box.clientWidth;
  if (!width) return;
  const maxH = Math.max(...models.flatMap((m) => m.rows.map((r) => r.horizon)));
  const x = (h: number) => M.left + (h / maxH) * (width - M.left - M.right);
  const y = (v: number) => M.top + (1 - v) * (HEIGHT - M.top - M.bottom);
  const svg = document.createElementNS(SVG, "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(HEIGHT));

  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: M.left, x2: width - M.right, y1: y(v), y2: y(v), class: v === 0 ? "axis" : "grid" }, svg);
    el("text", { x: M.left - 8, y: y(v), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" }, svg).textContent = `${v * 100}%`;
  }
  for (let h = 0; h <= maxH; h++) {
    el("text", { x: x(h), y: HEIGHT - M.bottom + 18, class: "tick", "text-anchor": "middle" }, svg).textContent = String(h);
  }
  el("text", { x: (M.left + width - M.right) / 2, y: HEIGHT - 4, class: "tick", "text-anchor": "middle" }, svg).textContent =
    "imagined pushes in a row";

  const cross = el("line", { y1: M.top, y2: HEIGHT - M.bottom, class: "crosshair", visibility: "hidden" }, svg);
  models.forEach((m, i) => {
    const cls = i === models.length - 1 ? "s-main" : "s-other"; // the newest model is the story
    const d = m.rows.map((r, j) => `${j ? "L" : "M"}${x(r.horizon)},${y(r.within_tolerance)}`).join("");
    el("path", { d, class: `line ${cls}` }, svg);
    for (const r of m.rows) el("circle", { cx: x(r.horizon), cy: y(r.within_tolerance), r: 4, class: `dot ${cls}` }, svg);
    const last = m.rows[m.rows.length - 1];
    el("text", { x: x(last.horizon) + 8, y: y(last.within_tolerance), class: "end-label", "dominant-baseline": "middle" }, svg).textContent = m.name;
  });

  // hover: snap to the nearest horizon
  const hit = el("rect", { x: M.left, y: M.top, width: width - M.left - M.right, height: HEIGHT - M.top - M.bottom, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (e) => {
    const bounds = svg.getBoundingClientRect();
    const h = Math.round(((e.clientX - bounds.left - M.left) / (width - M.left - M.right)) * maxH);
    const clamped = Math.max(0, Math.min(maxH, h));
    cross.setAttribute("x1", String(x(clamped)));
    cross.setAttribute("x2", String(x(clamped)));
    cross.setAttribute("visibility", "visible");
    const lines = models.flatMap((m) => {
      const r = m.rows.find((row) => row.horizon === clamped);
      return r ? [`${m.name}: ${Math.round(r.within_tolerance * 100)}% accurate · median ${r.px_median.toFixed(1)} px, ${r.deg_median.toFixed(1)}°`] : [];
    });
    showTip([`${clamped} imagined push${clamped === 1 ? "" : "es"}`, ...lines].join("\n"), e.clientX, e.clientY);
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    hideTip();
  });
  box.replaceChildren(svg);
}

export function mountHorizonCharts() {
  for (const fig of document.querySelectorAll<HTMLElement>("[data-horizon]:not([data-mounted])")) {
    fig.dataset.mounted = "1";
    const models: Model[] = JSON.parse(fig.dataset.horizon!);
    const box = fig.querySelector<HTMLElement>(".horizon-chart")!;
    new ResizeObserver(() => draw(box, models)).observe(box);
  }
}
