// Episode logs as written by `jat eval` (src/jat/agent/episode_log.py), and the steps the
// player walks through.

import { toWorld } from "./draw";

export type Pose = [number, number, number];
export type Point = [number, number];

export interface Push {
  start: Point;
  end: Point;
}

export interface Frame {
  id: string;
  kind: "real" | "imagined";
  pose: Pose; // what the LLM was told
  true_pose: Pose; // real physics on the same branch
  parent: string | null;
  push: Push | null;
  trajectory: number[][]; // real frames: [pusher x, y, T x, y, theta] every 0.1 s
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
  task: { id: string; family: string; start: Pose; goal: Pose; max_pushes: number };
  frames: Frame[];
  turns: Turn[];
  outcome: { success: boolean; position_error: number; angle_error: number };
  stop_reason: string;
}

export type Step =
  | { kind: "start"; real: string; caption: string }
  | { kind: "imagine"; real: string; from: string; frames: string[]; caption: string; detail: string }
  | { kind: "act"; real: string; previous: string; caption: string; detail: string }
  | { kind: "note"; real: string; caption: string; detail: string }
  | { kind: "end"; real: string; caption: string; detail: string };

const BASE = import.meta.env.BASE_URL.replace(/\/?$/, "/");

export async function loadEpisode(id: string): Promise<Episode> {
  const response = await fetch(`${BASE}data/episodes/${id}.json`);
  if (!response.ok) throw new Error(`episode ${id}: HTTP ${response.status}`);
  return response.json();
}

export function frameMap(episode: Episode): Map<string, Frame> {
  return new Map(episode.frames.map((f) => [f.id, f]));
}

export function pushText(push: Push): string {
  const r = (v: number) => Math.round(v);
  return `(${r(push.start[0])}, ${r(push.start[1])}) → (${r(push.end[0])}, ${r(push.end[1])})`;
}

/** The goal as the agent was told it (Task.text on the Python side, which the log does not store). */
export function goalText(goal: Pose): string {
  const [cx, cy] = toWorld(goal, [0, 45]);
  const degrees = ((Math.round((goal[2] * 180) / Math.PI) % 360) + 360) % 360;
  return `Move the T to the green outline: center (${Math.round(cx)}, ${Math.round(cy)}), angle ${degrees}°.`;
}

/** One step per tool call, in order. The LLM's message rides on its turn's first call. */
export function buildSteps(episode: Episode): Step[] {
  const frames = frameMap(episode);
  let real = "R0";
  const steps: Step[] = [{ kind: "start", real, caption: goalText(episode.task.goal) }];

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
          steps.push({
            kind: "act",
            real: next,
            previous: real,
            caption,
            detail: `real push ${pushText(frames.get(next)!.push!)}`,
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
    detail: `final error ${Math.round(o.position_error)} px, ${Math.round(o.angle_error)}° (success needs < 20 px and < 15°)`,
  });
  return steps;
}
