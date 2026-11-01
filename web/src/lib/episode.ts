// Episode logs as written by `jat eval` (src/jat/agent/episode_log.py), and the steps the
// player walks through.

import { goalText, isBilliards, successText, type Point, type State } from "./world";

export interface Push {
  start: Point;
  end: Point;
}

export interface Frame {
  id: string;
  kind: "real" | "imagined";
  pose: State; // what the LLM was told
  true_pose: State; // real physics on the same branch
  parent: string | null;
  push: Push | null;
  trajectory: number[][]; // real frames: [pusher x, y, ...state] every 0.1 s
}

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  result: string;
}

export interface Turn {
  content: string;
  tool_calls: ToolCall[];
}

export interface Episode {
  episode_id: string;
  setting: string;
  // `world` is missing from logs written before there were several worlds: PushT
  task: { id: string; family: string; start: State; goal: State; max_pushes: number; world?: string };
  frames: Frame[];
  turns: Turn[];
  outcome: { success: boolean; position_error: number; angle_error: number };
  stop_reason: string;
}

export type Step =
  | { kind: "start"; real: string; caption: string }
  | { kind: "imagine"; real: string; from: string; frames: string[]; caption: string; detail: string }
  | { kind: "choose"; real: string; chosen: string; candidates: string[]; caption: string; detail: string }
  | { kind: "act"; real: string; previous: string; chosen?: string; caption: string; detail: string }
  | { kind: "note"; real: string; caption: string; detail: string }
  | { kind: "end"; real: string; caption: string; detail: string };

const BASE = import.meta.env.BASE_URL.replace(/\/?$/, "/");

/** `src` is "<run>/<episode id>", as exported by `jat export-web`. */
export async function loadEpisode(src: string): Promise<Episode> {
  const response = await fetch(`${BASE}data/episodes/${src}.json`);
  if (!response.ok) throw new Error(`episode ${src}: HTTP ${response.status}`);
  return response.json();
}

export const worldOf = (episode: Episode) => episode.task.world ?? "pusht";

export function frameMap(episode: Episode): Map<string, Frame> {
  return new Map(episode.frames.map((f) => [f.id, f]));
}

export function pushText(push: Push): string {
  const r = (v: number) => Math.round(v);
  return `(${r(push.start[0])}, ${r(push.start[1])}) → (${r(push.end[0])}, ${r(push.end[1])})`;
}

const samePush = (a: Push, b: Push) =>
  Math.hypot(a.start[0] - b.start[0], a.start[1] - b.start[1]) +
  Math.hypot(a.end[0] - b.end[0], a.end[1] - b.end[1]);

/** The imagined push from real frame `real` that the real push repeats: the plan the agent
 * picked. Candidates are every first push it imagined from that frame. */
export function chosenPlan(
  episode: Episode,
  real: string,
  push: Push,
): { chosen: string; candidates: string[] } | null {
  const candidates = episode.frames.filter((f) => f.kind === "imagined" && f.parent === real && f.push);
  let best: Frame | null = null;
  for (const f of candidates) {
    if (samePush(f.push!, push) < (best ? samePush(best.push!, push) : 20)) best = f;
  }
  return best ? { chosen: best.id, candidates: candidates.map((f) => f.id) } : null;
}

/** One step per tool call, in order. The LLM's message rides on its turn's first call. Before
 * each real push, a "choose" step shows which imagined plan the push repeats. */
export function buildSteps(episode: Episode): Step[] {
  const frames = frameMap(episode);
  let real = "R0";
  const world = worldOf(episode);
  const steps: Step[] = [{ kind: "start", real, caption: goalText(world, episode.task.goal) }];

  for (const turn of episode.turns) {
    let caption = turn.content.trim();
    if (turn.tool_calls.length === 0 && caption) {
      steps.push({ kind: "note", real, caption, detail: "" });
    }
    for (const call of turn.tool_calls) {
      if (call.name === "simulate") {
        // "I3 = I1 + push ..." lines name every frame this call created
        const created = [...call.result.matchAll(/(I\d+) = [RI]\d+ \+ push/g)].map((m) => m[1]);
        const plans = (call.args.plans as unknown[] | undefined)?.length ?? 0;
        const from = String(call.args.from_frame ?? real);
        steps.push({
          kind: "imagine",
          real,
          from,
          frames: created,
          caption,
          detail: `${plans} plan${plans === 1 ? "" : "s"}, ${created.length} imagined push${created.length === 1 ? "" : "es"} from ${from}`,
        });
      } else if (call.name === "act") {
        const next = call.result.match(/Current real frame: (R\d+)/)?.[1];
        if (next && frames.has(next)) {
          const push = frames.get(next)!.push!;
          const pick = chosenPlan(episode, real, push);
          if (pick) {
            const n = pick.candidates.length;
            steps.push({
              kind: "choose",
              real,
              ...pick,
              caption,
              detail: `picks ${pick.chosen}, one of ${n} push${n === 1 ? "" : "es"} it imagined from ${real}`,
            });
            caption = "";
          }
          steps.push({
            kind: "act",
            real: next,
            previous: real,
            chosen: pick?.chosen,
            caption,
            detail: `real push ${pushText(push)}`,
          });
          real = next;
        } else {
          steps.push({ kind: "note", real, caption, detail: "push rejected: nothing happened" });
        }
      } else if (call.name === "done") {
        steps.push({ kind: "note", real, caption, detail: "the agent says it is done" });
      }
      caption = "";
    }
  }

  const o = episode.outcome;
  steps.push({
    kind: "end",
    real,
    caption: o.success ? "Success." : "Not at the goal.",
    detail: `final error ${Math.round(o.position_error)} px${isBilliards(world) ? "" : `, ${Math.round(o.angle_error)}°`} (${successText(world)})`,
  });
  return steps;
}
