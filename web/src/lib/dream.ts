// Dream mode: the world model runs in the browser. V-JEPA 2 is too big for a page, so each
// start state comes with its latent, computed offline (`jat export-dream`). Every push the
// visitor draws is imagined by the predictor and read by the translator, both in ONNX.

import { drawScene, fitCanvas, type Ghost } from "./draw";
import type { Point, Pose } from "./episode";
import { onThemeChange } from "./theme";

const BASE = import.meta.env.BASE_URL.replace(/\/?$/, "/");
const ORT_VERSION = "1.31.1"; // must match web/package.json: the wasm files come from the CDN
const MIN_LENGTH = 10;
const MAX_LENGTH = 200;
const MARGIN = 20;

interface Manifest {
  world: string;
  readout: string;
  table: number;
  tokens: number;
  dim: number;
  starts: number[][];
}

interface Dreamer {
  manifest: Manifest;
  latents: Float32Array[]; // one per start
  step(z: Float32Array, push: number[]): Promise<Float32Array>;
  read(z: Float32Array): Promise<number[]>;
}

/** IEEE half floats (the latents ship as float16) to float32. */
function halfToFloat(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    const h = bits[i];
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    out[i] =
      exp === 0
        ? sign * 2 ** -14 * (frac / 1024)
        : exp === 31
          ? frac ? NaN : sign * Infinity
          : sign * 2 ** (exp - 15) * (1 + frac / 1024);
  }
  return out;
}

async function loadDreamer(world: string): Promise<Dreamer> {
  const ort = await import("onnxruntime-web");
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  const root = `${BASE}dream/${world}/`;
  const [manifest, bin, predictor, reader] = await Promise.all([
    fetch(`${root}dream.json`).then((r) => r.json() as Promise<Manifest>),
    fetch(`${root}starts.bin`).then((r) => r.arrayBuffer()),
    ort.InferenceSession.create(`${root}predictor.onnx`),
    ort.InferenceSession.create(`${root}reader.onnx`),
  ]);
  const size = manifest.tokens * manifest.dim;
  const all = halfToFloat(new Uint16Array(bin));
  const latents = manifest.starts.map((_, i) => all.slice(i * size, (i + 1) * size));
  const shape = [1, manifest.tokens, manifest.dim];
  return {
    manifest,
    latents,
    async step(z, push) {
      const scaled = push.map((v) => v / (manifest.table / 2) - 1);
      const out = await predictor.run({
        z: new ort.Tensor("float32", z, shape),
        push: new ort.Tensor("float32", Float32Array.from(scaled), [1, 4]),
      });
      return out.z_next.data as Float32Array;
    },
    async read(z) {
      const out = await reader.run({ z: new ort.Tensor("float32", z, shape) });
      return Array.from(out.state.data as Float32Array);
    },
  };
}

/** A push the world model was trained on: inside the table, 10 to 200 px long. */
function legal(start: Point, end: Point): [Point, Point] {
  const clamp = (v: number) => Math.min(512 - MARGIN, Math.max(MARGIN, v));
  const s: Point = [clamp(start[0]), clamp(start[1])];
  let [dx, dy] = [end[0] - s[0], end[1] - s[1]];
  const length = Math.hypot(dx, dy);
  if (length > MAX_LENGTH) [dx, dy] = [(dx * MAX_LENGTH) / length, (dy * MAX_LENGTH) / length];
  return [s, [clamp(s[0] + dx), clamp(s[1] + dy)]];
}

function drawBalls(canvas: HTMLCanvasElement, layers: { state: number[]; alpha: number }[], arrow?: [Point, Point]) {
  const ctx = canvas.getContext("2d")!;
  const style = getComputedStyle(canvas);
  const scale = canvas.width / 512;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.fillStyle = style.getPropertyValue("--table").trim();
  ctx.fillRect(0, 0, 512, 512);
  const colors = ["#e8e6dc", "#c62828"];
  for (const { state, alpha } of layers) {
    for (let b = 0; b < state.length / 2; b++) {
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(state[2 * b], state[2 * b + 1], 25, 0, Math.PI * 2);
      ctx.fillStyle = colors[b % colors.length];
      ctx.fill();
      ctx.strokeStyle = "#555";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  if (arrow) {
    ctx.strokeStyle = style.getPropertyValue("--pusher").trim();
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(...arrow[0]);
    ctx.lineTo(...arrow[1]);
    ctx.stroke();
  }
}

export function mountDreams() {
  for (const root of document.querySelectorAll<HTMLElement>("[data-dream]:not([data-mounted])")) {
    root.dataset.mounted = "1";
    mount(root);
  }
}

function mount(root: HTMLElement) {
  const $ = <T extends Element>(s: string) => root.querySelector<T>(s)!;
  const canvas = $<HTMLCanvasElement>("canvas");
  const status = $<HTMLElement>("[data-status]");
  const wake = $<HTMLButtonElement>("[data-wake]");
  const worlds = (root.dataset.worlds ?? "pusht").split(",");
  let dreamer: Dreamer | null = null;
  let world = worlds[0];
  let start = 0;
  let chain: { z: Float32Array; state: number[]; push?: [Point, Point] }[] = [];
  let drag: [Point, Point] | null = null;
  let busy = false;

  const isT = () => dreamer?.manifest.readout === "t_pose";

  function draw() {
    if (!dreamer) return;
    const states = chain.map((c) => c.state);
    const last = chain[chain.length - 1];
    if (isT()) {
      const ghosts: Ghost[] = chain.slice(1).map((c, i) => ({
        pose: c.state as Pose,
        label: `I${i + 1}`,
        strong: i === chain.length - 2,
        chosen: i === chain.length - 2,
        push: c.push && { start: c.push[0], end: c.push[1] },
      }));
      drawScene(canvas, { real: states[0] as Pose, ghosts, showTruth: false, realPush: drag ? { start: drag[0], end: drag[1] } : undefined });
    } else {
      const layers = chain.map((c, i) => ({ state: c.state, alpha: i === 0 ? 0.35 : i === chain.length - 1 ? 1 : 0.2 }));
      drawBalls(canvas, layers.length > 1 ? layers : [{ state: last.state, alpha: 1 }], drag ?? undefined);
    }
  }

  async function reset(i: number) {
    if (!dreamer) return;
    start = (i + dreamer.latents.length) % dreamer.latents.length;
    const z = dreamer.latents[start];
    chain = [{ z, state: await dreamer.read(z) }];
    status.textContent = `Start ${start + 1} of ${dreamer.latents.length}. Drag on the table to push.`;
    draw();
  }

  async function load(name: string) {
    world = name;
    status.textContent = `Loading the ${name} world model (about 20 MB)…`;
    wake.disabled = true;
    try {
      dreamer = await loadDreamer(name);
      root.classList.add("awake");
      await reset(0);
    } catch (error) {
      status.textContent = `Could not load the world model (${(error as Error).message}).`;
      wake.disabled = false;
    }
  }

  const point = (e: PointerEvent): Point => {
    const box = canvas.getBoundingClientRect();
    return [((e.clientX - box.left) / box.width) * 512, ((e.clientY - box.top) / box.height) * 512];
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (!dreamer || busy) return;
    canvas.setPointerCapture(e.pointerId);
    const p = point(e);
    drag = [p, p];
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    drag = legal(drag[0], point(e));
    draw();
  });
  canvas.addEventListener("pointerup", async () => {
    if (!drag || !dreamer) return;
    const push = drag;
    drag = null;
    if (Math.hypot(push[1][0] - push[0][0], push[1][1] - push[0][1]) < MIN_LENGTH) return draw();
    busy = true;
    const started = performance.now();
    const z = await dreamer.step(chain[chain.length - 1].z, [...push[0], ...push[1]]);
    const state = await dreamer.read(z);
    chain.push({ z, state, push });
    const ms = Math.round(performance.now() - started);
    status.textContent = `${chain.length - 1} imagined push${chain.length === 2 ? "" : "es"}, the last one in ${ms} ms in your browser. Drag again to continue the dream.`;
    busy = false;
    draw();
  });

  wake.addEventListener("click", () => load(world));
  $<HTMLButtonElement>("[data-back]").addEventListener("click", () => reset(start));
  $<HTMLButtonElement>("[data-next]").addEventListener("click", () => reset(start + 1));
  root.querySelectorAll<HTMLButtonElement>("[data-world]").forEach((b) =>
    b.addEventListener("click", () => load(b.dataset.world!)),
  );
  fitCanvas(canvas, draw);
  onThemeChange(draw);
}
