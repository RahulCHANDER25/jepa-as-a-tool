// A world's table on a canvas: goal, real state, imagined ghosts, the pusher. The shapes come
// from world.ts, so one drawing serves every world.

import { drawScene3d, tablePoint3d } from "./draw3d";
import { anchor, goalParts, is3D, isBilliards, parts, TABLE, type Point, type State } from "./world";

const PUSHER_RADIUS = 15;

export interface Ghost {
  pose: State;
  truth?: State; // where the real physics puts it
  label?: string;
  strong?: boolean; // the frame in focus; others are drawn fainter
  chosen?: boolean; // the imagined plan the agent picked: drawn green
  alpha?: number; // 0..1, for fading in and out
  push?: { start: Point; end: Point };
}

export interface Scene {
  world: string;
  goal?: State;
  real: State;
  ghosts: Ghost[];
  showTruth: boolean;
  showLabels?: boolean;
  pusher?: Point;
  realPush?: { start: Point; end: Point };
  view?: { x: number; y: number; size: number }; // the square of the table to show (default: all)
}

type Colors = Record<
  "table" | "grid" | "real" | "white" | "red" | "goal" | "ghost" | "chosen" | "truth" | "pusher" | "ink",
  string
>;

function colors(canvas: HTMLCanvasElement): Colors {
  const style = getComputedStyle(canvas);
  const v = (name: string) => style.getPropertyValue(name).trim();
  return {
    table: v("--table"),
    grid: v("--grid"),
    real: v("--t-real"),
    white: v("--ball-white"),
    red: v("--ball-red"),
    goal: v("--goal"),
    ghost: v("--ghost"),
    chosen: v("--chosen"),
    truth: v("--truth"),
    pusher: v("--pusher"),
    ink: v("--text-secondary"),
  };
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

/** The table point under the pointer (3D worlds: where the ray meets the table). */
export function tablePoint(canvas: HTMLCanvasElement, world: string, clientX: number, clientY: number): Point | null {
  if (is3D(world)) return tablePoint3d(canvas, clientX, clientY);
  const box = canvas.getBoundingClientRect();
  return [((clientX - box.left) / box.width) * TABLE, ((clientY - box.top) / box.height) * TABLE];
}

/** A canvas draws one kind of world for its whole life: 2D, or 3D through WebGL. */
export function drawScene(canvas: HTMLCanvasElement, scene: Scene) {
  if (is3D(scene.world)) return drawScene3d(canvas, scene);
  const ctx = canvas.getContext("2d")!;
  const c = colors(canvas);
  const { world } = scene;
  const view = scene.view ?? { x: 0, y: 0, size: TABLE };
  const scale = canvas.width / view.size;
  ctx.setTransform(scale, 0, 0, scale, -view.x * scale, -view.y * scale);
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
    for (const path of goalParts(world, scene.goal)) {
      ctx.globalAlpha = 0.22;
      ctx.fillStyle = c.goal;
      ctx.fill(path);
      ctx.globalAlpha = 1;
      ctx.strokeStyle = c.goal;
      ctx.lineWidth = 2;
      ctx.stroke(path);
    }
  }

  // real objects under the ghosts, so ghosts that overlap them stay visible
  const fills = isBilliards(world) ? [c.white, c.red] : [c.real];
  parts(world, scene.real).forEach((path, i) => {
    ctx.fillStyle = fills[i];
    ctx.fill(path);
    if (isBilliards(world)) {
      ctx.strokeStyle = c.ink;
      ctx.lineWidth = 1.5;
      ctx.stroke(path);
    }
  });

  // the chosen plan last, on top of the others
  const ghosts = [...scene.ghosts.filter((g) => !g.chosen), ...scene.ghosts.filter((g) => g.chosen)];
  for (const g of ghosts) {
    const color = g.chosen ? c.chosen : c.ghost;
    const fade = g.alpha ?? 1;
    ctx.globalAlpha = fade;
    if (g.push) arrow(ctx, g.push.start, g.push.end, color, g.chosen ? 3 : g.strong ? 2 : 1.25);
    for (const path of parts(world, g.pose)) {
      ctx.globalAlpha = fade * (g.chosen ? 0.4 : g.strong ? 0.28 : 0.12);
      ctx.fillStyle = color;
      ctx.fill(path);
      ctx.globalAlpha = fade * (g.strong || g.chosen ? 1 : 0.6);
      ctx.strokeStyle = color;
      ctx.lineWidth = g.chosen ? 3.5 : g.strong ? 2 : 1.25;
      if (g.chosen) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 14;
      }
      ctx.stroke(path);
      ctx.shadowBlur = 0;
    }
    ctx.globalAlpha = 1;
    if (scene.showTruth && g.truth) {
      ctx.strokeStyle = c.truth;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      for (const path of parts(world, g.truth)) ctx.stroke(path);
      ctx.setLineDash([]);
    }
  }

  for (const g of scene.showLabels === false ? [] : scene.ghosts) {
    if (!g.label || (g.alpha ?? 1) < 0.5) continue;
    const [x, y] = anchor(world, g.pose);
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
