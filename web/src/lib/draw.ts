// The PushT table on a canvas. Same T geometry as src/jat/worlds/tblock/shape.py, so ghosts and real
// T's line up exactly with what the Python side computed.

import type { Point, Pose } from "./episode";

const TABLE = 512;
const OUTLINE: Point[] = [
  [-60, 0], [60, 0], [60, 30], [15, 30], [15, 120], [-15, 120], [-15, 30], [-60, 30],
];
const CENTER: Point = [0, 45];
const PUSHER_RADIUS = 15;

export function toWorld(pose: Pose, [lx, ly]: Point): Point {
  const [x, y, theta] = pose;
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [x + c * lx - s * ly, y + s * lx + c * ly];
}

export interface Ghost {
  pose: Pose;
  truth?: Pose; // where the real physics puts it
  label?: string;
  strong?: boolean; // the frame in focus; others are drawn fainter
  chosen?: boolean; // the imagined plan the agent picked: drawn green
  alpha?: number; // 0..1, for fading in and out
  push?: { start: Point; end: Point };
}

export interface Scene {
  goal?: Pose;
  real: Pose;
  ghosts: Ghost[];
  showTruth: boolean;
  showLabels?: boolean;
  pusher?: Point;
  realPush?: { start: Point; end: Point };
}

type Colors = Record<
  "table" | "grid" | "real" | "goal" | "ghost" | "chosen" | "truth" | "pusher" | "ink",
  string
>;

function colors(canvas: HTMLCanvasElement): Colors {
  const style = getComputedStyle(canvas);
  const v = (name: string) => style.getPropertyValue(name).trim();
  return {
    table: v("--table"),
    grid: v("--grid"),
    real: v("--t-real"),
    goal: v("--goal"),
    ghost: v("--ghost"),
    chosen: v("--chosen"),
    truth: v("--truth"),
    pusher: v("--pusher"),
    ink: v("--text-secondary"),
  };
}

function tracePolygon(ctx: CanvasRenderingContext2D, pose: Pose) {
  ctx.beginPath();
  OUTLINE.forEach((p, i) => {
    const [x, y] = toWorld(pose, p);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.closePath();
}

function arrow(ctx: CanvasRenderingContext2D, from: Point, to: Point, color: string, width: number) {
  const angle = Math.atan2(to[1] - from[1], to[0] - from[0]);
  const head = 9;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(...from);
  ctx.lineTo(to[0] - Math.cos(angle) * head * 0.8, to[1] - Math.sin(angle) * head * 0.8);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(...to);
  ctx.lineTo(to[0] - head * Math.cos(angle - 0.45), to[1] - head * Math.sin(angle - 0.45));
  ctx.lineTo(to[0] - head * Math.cos(angle + 0.45), to[1] - head * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
}

/** Keeps the canvas backing store at the element's CSS size x devicePixelRatio. */
export function fitCanvas(canvas: HTMLCanvasElement, onResize: () => void) {
  const resize = () => {
    const size = Math.round(canvas.clientWidth * window.devicePixelRatio);
    if (size > 0 && canvas.width !== size) {
      canvas.width = canvas.height = size;
      onResize();
    }
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
}

export function drawScene(canvas: HTMLCanvasElement, scene: Scene) {
  const ctx = canvas.getContext("2d")!;
  const c = colors(canvas);
  const scale = canvas.width / TABLE;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.fillStyle = c.table;
  ctx.fillRect(0, 0, TABLE, TABLE);

  ctx.strokeStyle = c.grid;
  ctx.lineWidth = 1 / scale;
  for (let v = 64; v < TABLE; v += 64) {
    ctx.beginPath();
    ctx.moveTo(v, 0);
    ctx.lineTo(v, TABLE);
    ctx.moveTo(0, v);
    ctx.lineTo(TABLE, v);
    ctx.stroke();
  }

  // goal: soft fill + outline
  if (scene.goal) {
    tracePolygon(ctx, scene.goal);
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = c.goal;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = c.goal;
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.stroke();
  }

  // real T under the ghosts, so ghosts that overlap it stay visible
  tracePolygon(ctx, scene.real);
  ctx.fillStyle = c.real;
  ctx.fill();

  // the chosen plan last, on top of the others
  const ghosts = [...scene.ghosts.filter((g) => !g.chosen), ...scene.ghosts.filter((g) => g.chosen)];
  for (const g of ghosts) {
    const color = g.chosen ? c.chosen : c.ghost;
    const fade = g.alpha ?? 1;
    ctx.globalAlpha = fade;
    if (g.push) arrow(ctx, g.push.start, g.push.end, color, g.chosen ? 3 : g.strong ? 2 : 1.25);
    tracePolygon(ctx, g.pose);
    ctx.globalAlpha = fade * (g.chosen ? 0.4 : g.strong ? 0.28 : 0.12);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.globalAlpha = fade * (g.strong || g.chosen ? 1 : 0.6);
    ctx.strokeStyle = color;
    ctx.lineWidth = g.chosen ? 3.5 : g.strong ? 2 : 1.25;
    if (g.chosen) {
      ctx.shadowColor = color;
      ctx.shadowBlur = 14;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    if (scene.showTruth && g.truth) {
      tracePolygon(ctx, g.truth);
      ctx.strokeStyle = c.truth;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  for (const g of scene.showLabels === false ? [] : scene.ghosts) {
    if (!g.label || (g.alpha ?? 1) < 0.5) continue;
    const [x, y] = toWorld(g.pose, CENTER);
    ctx.font = "600 13px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineWidth = 3;
    ctx.strokeStyle = c.table;
    ctx.strokeText(g.label, x, y);
    ctx.fillStyle = c.ink;
    ctx.fillText(g.label, x, y);
  }

  if (scene.realPush) arrow(ctx, scene.realPush.start, scene.realPush.end, c.pusher, 2);
  if (scene.pusher) {
    ctx.beginPath();
    ctx.arc(scene.pusher[0], scene.pusher[1], PUSHER_RADIUS, 0, Math.PI * 2);
    ctx.fillStyle = c.pusher;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = c.table;
    ctx.stroke();
  }
}
