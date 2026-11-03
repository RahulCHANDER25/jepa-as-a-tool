// Imagination fan: for each real push of an episode, every push the agent imagined from that
// table pops up as a small card; the one it keeps lights up; then the real push plays on the big
// table, next to what was imagined. Plays on its own while visible.

import { drawScene, fitCanvas, type Ghost, type Scene } from "./draw";
import { onThemeChange } from "./theme";
import { frameMap, loadEpisode, worldOf, type Frame, type Push } from "./episode";
import { anchor, gapText, successText, TABLE, type Point, type State } from "./world";

const CARDS = 8;
const MARGIN = 100; // px of table around the objects, so a whole T fits
const MIN_VIEW = 240;
const POP_MS = 160; // between two cards appearing
const PICK_MS = 1200;
const ROW_MS = 50; // one trajectory row = 0.1 s of physics, at 2x
const COMPARE_MS = 1800;
const END_MS = 2600;

const gap = (a: Push, b: Push) =>
  Math.hypot(a.start[0] - b.start[0], a.start[1] - b.start[1]) +
  Math.hypot(a.end[0] - b.end[0], a.end[1] - b.end[1]);

interface Round {
  before: Frame; // the real table it imagined from
  after: Frame; // the real table after the push
  cards: Frame[]; // imagined pushes from `before`
  chosen: Frame | null; // the one the real push repeats
}

/** One round per real push. A push that continues a plan imagined earlier (no new imagining
 * from this table) shows that plan's step alone. */
function rounds(frames: Frame[]): Round[] {
  const byId = new Map(frames.map((f) => [f.id, f]));
  const imagined = frames.filter((f) => f.kind === "imagined" && f.push);
  return frames
    .filter((f) => f.kind === "real" && f.parent && f.push)
    .map((after) => {
      const before = byId.get(after.parent!)!;
      const fresh = imagined.filter((f) => f.parent === before.id);
      const pool = fresh.length ? fresh : imagined;
      const best = pool.reduce<Frame | null>(
        (b, f) => (gap(f.push!, after.push!) < (b ? gap(b.push!, after.push!) : 20) ? f : b),
        null,
      );
      let cards = fresh.length ? fresh.slice(0, CARDS) : best ? [best] : [];
      if (best && !cards.includes(best)) cards = [...cards.slice(0, CARDS - 1), best];
      return { before, after, cards, chosen: best };
    });
}

/** The square of the table where a round happens: the real object before and after, the goal,
 * every imagined result and the pushes. Shared by the big table and the cards. */
function viewOf(world: string, goal: State, r: Round, pushes = true): Scene["view"] {
  const points: Point[] = [anchor(world, r.before.true_pose), anchor(world, goal)];
  points.push(...r.cards.map((f) => anchor(world, f.pose)));
  if (pushes) {
    points.push(anchor(world, r.after.true_pose));
    for (const f of [r.after, ...r.cards]) if (f.push) points.push(f.push.start, f.push.end);
  }
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const size = Math.min(TABLE, Math.max(MIN_VIEW, Math.max(...xs) - Math.min(...xs) + 2 * MARGIN, Math.max(...ys) - Math.min(...ys) + 2 * MARGIN));
  const clamp = (v: number) => Math.max(0, Math.min(TABLE - size, v));
  return {
    x: clamp((Math.max(...xs) + Math.min(...xs)) / 2 - size / 2),
    y: clamp((Math.max(...ys) + Math.min(...ys)) / 2 - size / 2),
    size,
  };
}

export function mountFans() {
  for (const root of document.querySelectorAll<HTMLElement>("[data-fan]:not([data-mounted])")) {
    root.dataset.mounted = "1";
    mount(root).catch((e) => (root.querySelector("[data-say]")!.textContent = `Could not load this episode (${e.message}).`));
  }
}

async function mount(root: HTMLElement) {
  const episode = await loadEpisode(root.dataset.episode!);
  const world = worldOf(episode);
  const goal = episode.task.goal;
  const all = rounds(episode.frames);
  const byId = frameMap(episode);
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const table = root.querySelector<HTMLCanvasElement>("[data-table]")!;
  const grid = root.querySelector<HTMLElement>("[data-cards]")!;
  const say = root.querySelector<HTMLElement>("[data-say]")!;
  const phase = root.querySelector<HTMLElement>("[data-phase]")!;
  const play = root.querySelector<HTMLButtonElement>("[data-play]")!;
  const counter = root.querySelector<HTMLElement>("[data-counter]")!;

  let playing = false;
  let paused = false;
  let timers: number[] = [];
  let frame = 0;
  let lastScene: Scene | null = null;
  const redraws: (() => void)[] = [];

  const later = (ms: number, f: () => void) => timers.push(window.setTimeout(f, ms));
  function stop() {
    timers.forEach(clearTimeout);
    timers = [];
    cancelAnimationFrame(frame);
  }

  let view: Scene["view"]; // the big table: the objects and the pushes
  let close: Scene["view"]; // the cards: the objects only, closer

  function drawTable(s: Scene) {
    s = { ...s, view };
    lastScene = s;
    drawScene(table, s);
  }

  function tell(name: "imagine" | "pick" | "act" | "end", text: string) {
    phase.className = `phase phase-${name}`;
    phase.textContent = { imagine: "1 · Imagine", pick: "2 · Pick", act: "3 · Push for real", end: "Result" }[name];
    say.textContent = text;
  }

  function card(f: Frame, real: State): HTMLElement {
    const box = document.createElement("div");
    box.className = "fan-card";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", `${f.id}: an imagined push and where the world model says the object ends up`);
    const label = document.createElement("p");
    label.innerHTML = `<b>${f.id}</b> <span>${gapText(world, f.pose, goal)}</span>`;
    box.append(canvas, label);
    const ghost: Ghost = { pose: f.pose, push: f.push ?? undefined, strong: true };
    const draw = () =>
      drawScene(canvas, { world, goal, real, ghosts: [ghost], showTruth: false, showLabels: false, view: close });
    redraws.push(draw);
    requestAnimationFrame(() => fitCanvas(canvas, draw));
    return box;
  }

  /** `still`: every card at once and no animation, for the picture before the figure plays. */
  function showRound(i: number, still = false) {
    stop();
    const r = all[i];
    counter.textContent = `push ${i + 1} of ${all.length}`;
    redraws.length = 0;
    grid.replaceChildren();
    const real = r.before.true_pose;
    view = viewOf(world, goal, r);
    close = viewOf(world, goal, r, false);
    drawTable({ world, goal, real, ghosts: [], showTruth: false });
    tell("imagine", `From the table on the left, the agent imagines ${r.cards.length} push${r.cards.length === 1 ? "" : "es"}. Each card is the world model's answer.`);

    const boxes = r.cards.map((f) => card(f, real));
    if (still) {
      boxes.forEach((b) => b.classList.add("in"));
      grid.append(...boxes);
      return;
    }
    boxes.forEach((b, k) => later(reduceMotion ? 0 : k * POP_MS, () => {
      grid.append(b);
      requestAnimationFrame(() => b.classList.add("in"));
    }));
    later(r.cards.length * POP_MS + 500, () => pick(i, boxes));
  }

  function pick(i: number, boxes: HTMLElement[]) {
    const r = all[i];
    const k = r.chosen ? r.cards.indexOf(r.chosen) : -1;
    boxes.forEach((b, j) => b.classList.toggle(j === k ? "chosen" : "dim", true));
    const chosenGhost: Ghost[] = r.chosen ? [{ pose: r.chosen.pose, push: r.chosen.push ?? undefined, chosen: true }] : [];
    drawTable({ world, goal, real: r.before.true_pose, ghosts: chosenGhost, showTruth: false });
    tell("pick", r.chosen ? `It keeps ${r.chosen.id} (green): the push it will make for real.` : "It makes a push it did not imagine.");
    later(PICK_MS, () => act(i, chosenGhost));
  }

  function act(i: number, keep: Ghost[]) {
    const r = all[i];
    const rows = r.after.trajectory.length ? r.after.trajectory : [[...(r.after.push!.end as Point), ...r.after.true_pose]];
    tell("act", "The real pusher makes that push. Green is what it imagined.");
    const started = performance.now();
    const tick = (now: number) => {
      const n = Math.max(0, Math.floor((now - started) / ROW_MS)); // a frame's time can predate `started`
      if (n >= rows.length || reduceMotion) {
        drawTable({ world, goal, real: r.after.true_pose, ghosts: keep, showTruth: false, realPush: r.after.push ?? undefined });
        const off = r.chosen ? `Imagined vs real: off by ${gapText(world, r.chosen.pose, r.after.true_pose)}.` : "";
        tell("act", `${off} Now ${gapText(world, r.after.true_pose, goal)} from the goal.`);
        later(COMPARE_MS, () => (i + 1 < all.length ? showRound(i + 1) : end()));
        return;
      }
      const row = rows[n];
      drawTable({ world, goal, real: row.slice(2), ghosts: keep, showTruth: false, pusher: [row[0], row[1]] });
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }

  function end() {
    const o = episode.outcome;
    const last = byId.get(all[all.length - 1].after.id)!;
    drawTable({ world, goal, real: last.true_pose, ghosts: [], showTruth: false });
    tell("end", `${o.success ? "Success" : "Not at the goal"} in ${all.length} real push${all.length === 1 ? "" : "es"}, after ${episode.frames.filter((f) => f.kind === "imagined").length} imagined ones (${successText(world)}).`);
    later(END_MS, () => playing && showRound(0));
  }

  function setPlaying(on: boolean) {
    if (on === playing) return;
    playing = on;
    play.textContent = on ? "❚❚ Pause" : "▶ Play";
    if (on) showRound(0);
    else stop();
  }

  play.addEventListener("click", () => {
    paused = playing;
    setPlaying(!playing);
  });
  fitCanvas(table, () => lastScene && drawTable(lastScene));
  onThemeChange(() => {
    if (lastScene) drawTable(lastScene);
    redraws.forEach((d) => d());
  });
  showRound(0, true); // until the figure is in view
  new IntersectionObserver(([e]) => !paused && setPlaying(e.isIntersecting), { threshold: 0.35 }).observe(root);
}
