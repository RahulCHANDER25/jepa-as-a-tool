// One shared tooltip for every element with a data-tip attribute (lines split on "\n").

let tip: HTMLDivElement | null = null;

export function showTip(text: string, x: number, y: number) {
  if (!tip) {
    tip = document.createElement("div");
    tip.className = "viz-tip";
    tip.setAttribute("role", "tooltip");
    document.body.append(tip);
  }
  tip.replaceChildren(
    ...text.split("\n").map((line, i) => {
      const div = document.createElement("div");
      div.textContent = line;
      if (i === 0) div.className = "viz-tip-title";
      return div;
    }),
  );
  tip.hidden = false;
  const pad = 12;
  const { width, height } = tip.getBoundingClientRect();
  const left = Math.min(x + pad, window.innerWidth - width - pad);
  const top = y - height - pad < 0 ? y + pad : y - height - pad;
  tip.style.left = `${Math.max(pad, left)}px`;
  tip.style.top = `${top}px`;
}

export function hideTip() {
  if (tip) tip.hidden = true;
}

export function mountTooltips() {
  for (const el of document.querySelectorAll<HTMLElement>("[data-tip]:not([data-tip-mounted])")) {
    el.dataset.tipMounted = "1";
    el.addEventListener("pointermove", (e) => showTip(el.dataset.tip!, e.clientX, e.clientY));
    el.addEventListener("pointerleave", hideTip);
    el.addEventListener("focus", () => {
      const r = el.getBoundingClientRect();
      showTip(el.dataset.tip!, r.left + r.width / 2, r.top);
    });
    el.addEventListener("blur", hideTip);
  }
}
