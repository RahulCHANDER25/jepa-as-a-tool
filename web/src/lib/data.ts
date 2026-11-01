// Build-time access to the exported data (web/public/data, written by `jat export-web`).
// Components call these in their frontmatter, so a missing run or episode fails the build
// instead of shipping a broken figure. Server-only: it reads the file system.

import { existsSync } from "node:fs";
import { join } from "node:path";
import results from "../../public/data/results.json";
import horizon from "../../public/data/horizon.json";

// Astro runs from web/; import.meta.url points into the build output, not src/
const exists = (path: string) => existsSync(join(process.cwd(), "public", path));

export type Row = (typeof results.runs)[number]["rows"][number];
export type PairedRow = (typeof results.runs)[number]["paired"][number];
export type Run = (typeof results.runs)[number];
export type HorizonRow = { horizon: number; n: number; px_median: number; deg_median: number; within_tolerance: number };

export const SETTINGS = [
  ["no_tool", "LLM alone"],
  ["wm_tool", "LLM + world model"],
  ["oracle_tool", "LLM + real physics"],
] as const;

export const WORLD_NAMES: Record<string, string> = {
  pusht: "PushT (2D)",
  tabletop: "Tabletop (3D)",
  billiards: "Billiards (3D)",
  "tabletop-new-look": "New room (never seen)",
};

export const FAMILY_NAMES: Record<string, string> = {
  all: "All tasks",
  translate: "Move",
  rotate: "Turn",
  pose: "Move and turn",
  shot: "Direct shot",
  bank: "Off a cushion",
  carom: "Through the white ball",
};

export const runs: Run[] = results.runs;

export function run(name: string): Run {
  const found = runs.find((r) => r.name === name);
  if (!found) throw new Error(`no run "${name}" in results.json (runs: ${runs.map((r) => r.name).join(", ")})`);
  return found;
}

export const hasRun = (name: string) => runs.some((r) => r.name === name);

export function row(runName: string, setting: string, family = "all"): Row | undefined {
  return run(runName).rows.find((r) => r.setting === setting && r.family === family);
}

export function pairedRow(runName: string, setting: string, family = "all"): PairedRow | undefined {
  return run(runName).paired.find((r) => r.setting === setting && r.family === family);
}

/** "<run>/<id>" for the player, after checking the episode was exported. */
export function episodeSrc(runName: string, id: string): string {
  run(runName);
  if (!exists(`data/episodes/${runName}/${id}.json`)) {
    throw new Error(`no episode ${runName}/${id} in web/public/data/episodes`);
  }
  return `${runName}/${id}`;
}

/** Every world-model version with test rows for `world`. */
export function horizonModels(world: string): { name: string; rows: HorizonRow[] }[] {
  return horizon.models.flatMap((m) => {
    const rows = (m.worlds as Record<string, HorizonRow[]>)[world];
    return rows ? [{ name: m.name, rows }] : [];
  });
}

/** Dream mode for a world exists once `jat export-dream` wrote its bundle. */
export const hasDream = (world: string) => exists(`dream/${world}/dream.json`);

/** A 3D clip (`jat clip`) of a world, if one was rendered into web/public/clips. */
export const hasClip = (world: string) => exists(`clips/${world}.mp4`);

export const pct = (v: number) => `${Math.round(v * 100)}%`;

/** A p-value the way the text prints it. */
export const pText = (p: number) => (p < 0.001 ? "p < 0.001" : `p = ${p < 0.01 ? p.toFixed(3) : p.toFixed(2)}`);
