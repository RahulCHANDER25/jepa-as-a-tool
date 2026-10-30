// Imagination tree: every frame of an episode, real ones as the trunk, imagined ones as branches.
// Selecting a frame draws it with the path that led to it.

import { drawScene, fitCanvas, toWorld, type Ghost, type Scene } from "./draw";
import { onThemeChange } from "./theme";
import { frameMap, loadEpisode, pushText, type Frame, type Pose } from "./episode";

function error(a: Pose, b: Pose): string {
  const [ax, ay] = toWorld(a, [0, 45]);
  const [bx, by] = toWorld(b, [0, 45]);
  const px = Math.hypot(ax - bx, ay - by);
  const turn = Math.abs((((a[2] - b[2] + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI) - Math.PI);
  return `${Math.round(px)} px, ${Math.round((turn * 180) / Math.PI)}°`;
}

const samePush = (a: Frame["push"], b: Frame["push"]) =>
  !!a && !!b && [...a.start, ...a.end].every((v, i) => Math.abs(v - [...b.start, ...b.end][i]) < 0.5);

export function mountTrees() {
  for (const root of document.querySelectorAll<HTMLElement>("[data-tree]:not([data-mounted])")) {
    root.dataset.mounted = "1";
    mount(root).catch((e) => (root.querySelector("[data-caption]")!.textContent = `Could not load this episode (${e.message}).`));
  }
}

async function mount(root: HTMLElement) {
  const episode = await loadEpisode(root.dataset.episode!);
  const frames = frameMap(episode);
  const canvas = root.querySelector<HTMLCanvasElement>("canvas")!;
  const list = root.querySelector<HTMLElement>("[data-list]")!;
  const caption = root.querySelector<HTMLElement>("[data-caption]")!;
  const truth = root.querySelector<HTMLInputElement>("[data-truth]")!;

  const children = new Map<string, Frame[]>();
  for (const f of episode.frames) {
    if (f.kind === "imagined" && f.parent) children.set(f.parent, [...(children.get(f.parent) ?? []), f]);
  }
  // an imagined push that the agent then ran for real
  const taken = new Set<string>();
  for (const real of episode.frames.filter((f) => f.kind === "real" && f.parent)) {
    for (const f of children.get(real.parent!) ?? []) if (samePush(f.push, real.push)) taken.add(f.id);
  }

  let selected = "";
  const buttons = new Map<string, HTMLButtonElement>();

  function realAncestor(id: string): Frame {
    let f = frames.get(id)!;
    while (f.kind === "imagined") f = frames.get(f.parent!)!;
    return f;
  }

  function select(id: string) {
    selected = id;
    buttons.forEach((b, key) => b.setAttribute("aria-pressed", String(key === id)));
    const f = frames.get(id)!;
    const path: Ghost[] = [];
    for (let p = f; p.kind === "imagined"; p = frames.get(p.parent!)!) {
      path.unshift({ pose: p.pose, truth: p.true_pose, label: p.id, strong: p.id === id, push: p.push ?? undefined });
    }
    const s: Scene = { goal: episode.task.goal, real: realAncestor(id).true_pose, ghosts: path, showTruth: truth.checked };
    drawScene(canvas, s);
    caption.textContent = f.kind === "real"
      ? `${id}: the real table${f.push ? ` after the push ${pushText(f.push)}` : " at the start"}.`
      : `${id}: imagined from ${f.parent} with the push ${pushText(f.push!)}. The world model is off by ${error(f.pose, f.true_pose)}.`;
  }

  function node(f: Frame): HTMLLIElement {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = `node node-${f.kind}${taken.has(f.id) ? " taken" : ""}`;
    button.textContent = f.id;
    button.title = f.kind === "real" ? "real frame" : `off by ${error(f.pose, f.true_pose)}`;
    if (taken.has(f.id)) button.setAttribute("aria-description", "this push was then run for real");
    button.addEventListener("click", () => select(f.id));
    buttons.set(f.id, button);
    li.append(button);
    const kids = children.get(f.id);
    if (kids) {
      const ul = document.createElement("ul");
      ul.append(...kids.map(node));
      li.append(ul);
    }
    return li;
  }

  const trunk = document.createElement("ul");
  trunk.append(...episode.frames.filter((f) => f.kind === "real").map(node));
  list.append(trunk);

  select([...taken][0] ?? episode.frames.find((f) => f.kind === "imagined")?.id ?? "R0");
  truth.addEventListener("change", () => select(selected));
  fitCanvas(canvas, () => select(selected));
  onThemeChange(() => select(selected));
}
