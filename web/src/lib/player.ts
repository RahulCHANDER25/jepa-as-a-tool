// Episode player: walks an episode step by step (imagine, push, ...) on the table canvas.

import { drawScene, fitCanvas, type Ghost, type Scene } from "./draw";
import { onThemeChange } from "./theme";
import { buildSteps, frameMap, loadEpisode, type Point, type Pose, type Step } from "./episode";

const STEP_MS = 2400; // pause on a step while playing
const ROW_MS = 110; // one trajectory row = 0.1 s of physics, shown slightly slower
const LOOP_PAUSE_MS = 3500;
const FAN_MS = 1400; // imagined frames appear one by one over this time
const CHOOSE_MS = 700; // the other plans fade, the chosen one turns green
const TYPE_MS = 18; // per character of the agent's sentence
const BADGES: Record<Step["kind"], string> = {
  start: "Task",
  imagine: "Imagine",
  choose: "Choose",
  act: "Push",
  note: "Note",
  end: "Result",
};

export function mountPlayers() {
  for (const root of document.querySelectorAll<HTMLElement>("[data-player]:not([data-mounted])")) {
    root.dataset.mounted = "1";
    mount(root).catch((error) => {
      root.querySelector("[data-caption]")!.textContent = `Could not load this episode (${error.message}).`;
    });
  }
}

async function mount(root: HTMLElement) {
  const episode = await loadEpisode(root.dataset.episode!);
  const frames = frameMap(episode);
  const steps = buildSteps(episode);
  const compact = root.dataset.compact === "1";
  const loop = root.dataset.autoplay === "1";
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = <T extends Element>(selector: string) => root.querySelector<T>(selector);
  const canvas = $<HTMLCanvasElement>("canvas")!;
  const caption = $<HTMLElement>("[data-caption]")!;
  const transcript = $<HTMLOListElement>("[data-transcript]");
  const slider = $<HTMLInputElement>("[data-slider]");
  const counter = $<HTMLElement>("[data-counter]");
  const playButton = $<HTMLButtonElement>("[data-play]")!;
  const truth = $<HTMLInputElement>("[data-truth]");
  const thought = $<HTMLElement>("[data-thought]");

  let index = 0;
  let playing = false;
  let timer = 0;
  let animation = 0;
  let lastScene: Scene | null = null;
  let typer = 0;

  const truePose = (id: string) => frames.get(id)!.true_pose;

  function ghost(id: string, strong: boolean, extra: Partial<Ghost> = {}): Ghost {
    const f = frames.get(id)!;
    return { pose: f.pose, truth: f.true_pose, label: id, strong, push: f.push ?? undefined, ...extra };
  }

  /** `t` in 0..1: how far the step's entrance animation is (1 = still picture). */
  function scene(step: Step, real: Pose, pusher?: Point, t = 1): Scene {
    const base: Scene = {
      goal: episode.task.goal,
      real,
      ghosts: [],
      showTruth: truth?.checked ?? !compact, // small players stay readable: ghosts only
      showLabels: !compact,
    };
    if (step.kind === "imagine") {
      // an imagined starting frame is shown faint, under the new ones
      if (step.from.startsWith("I")) base.ghosts.push(ghost(step.from, false));
      // fan in: frame i appears at t = i / n
      const n = step.frames.length;
      step.frames.forEach((id, i) => {
        const alpha = Math.max(0, Math.min(1, t * n - i));
        if (alpha > 0) base.ghosts.push(ghost(id, true, { alpha }));
      });
    }
    if (step.kind === "choose") {
      const fade = 1 - 0.75 * t;
      base.ghosts.push(
        ...step.candidates.filter((id) => id !== step.chosen).map((id) => ghost(id, false, { alpha: fade })),
        ghost(step.chosen, true, { chosen: t > 0.3 }),
      );
    }
    if (step.kind === "act") {
      if (step.chosen) base.ghosts.push(ghost(step.chosen, false, { chosen: true, alpha: 0.7 }));
      base.realPush = frames.get(step.real)!.push ?? undefined;
      base.pusher = pusher;
    }
    return base;
  }

  function draw(s: Scene) {
    lastScene = s;
    drawScene(canvas, s);
  }

  /** The agent's sentence over the table, typed out when `typing`. */
  function think(step: Step, typing: boolean) {
    if (!thought) return;
    const text = step.kind === "start" || step.kind === "end" ? "" : step.caption;
    if (!text) return; // keep the last thought until the agent says something new
    thought.hidden = false;
    if (!typing || reduceMotion) {
      thought.textContent = text;
      return;
    }
    const started = performance.now();
    const tick = (now: number) => {
      const n = Math.floor((now - started) / TYPE_MS);
      thought.textContent = text.slice(0, n);
      if (n < text.length) typer = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(typer);
    typer = requestAnimationFrame(tick);
  }

  function render(final: boolean) {
    const step = steps[index];
    caption.textContent = compact || step.kind === "start" || step.kind === "end"
      ? [step.caption, step.detail].filter(Boolean).join(" ")
      : step.detail;
    if (slider) slider.value = String(index);
    if (counter) counter.textContent = `${index + 1} / ${steps.length}`;
    transcript?.querySelectorAll("li").forEach((li, i) => {
      li.classList.toggle("current", i === index);
      if (i === index) li.setAttribute("aria-current", "step");
      else li.removeAttribute("aria-current");
    });
    const current = transcript?.children[index] as HTMLElement | undefined;
    if (transcript && current) {
      // scroll the panel only, never the page
      const top = current.offsetTop - transcript.offsetTop;
      if (top < transcript.scrollTop || top + current.offsetHeight > transcript.scrollTop + transcript.clientHeight) {
        transcript.scrollTo({ top: top - 8, behavior: reduceMotion ? "auto" : "smooth" });
      }
    }
    if (!final) return; // an animation draws this step
    draw(scene(step, truePose(step.real)));
  }

  function animateAct(step: Extract<Step, { kind: "act" }>, done: () => void) {
    const rows = frames.get(step.real)!.trajectory;
    const push = frames.get(step.real)!.push!;
    const started = performance.now();
    const tick = (now: number) => {
      const i = Math.floor((now - started) / ROW_MS);
      if (i >= rows.length) {
        draw(scene(step, truePose(step.real)));
        done();
        return;
      }
      const row = rows[Math.max(i, 0)];
      draw(scene(step, [row[2], row[3], row[4]], [row[0], row[1]]));
      animation = requestAnimationFrame(tick);
    };
    draw(scene(step, truePose(step.previous), push.start));
    animation = requestAnimationFrame(tick);
  }

  /** The imagine fan-in and the choose fade, over `ms`. */
  function animateEntrance(step: Step, ms: number, done: () => void) {
    const started = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - started) / ms);
      draw(scene(step, truePose(step.real), undefined, t));
      if (t < 1) animation = requestAnimationFrame(tick);
      else done();
    };
    animation = requestAnimationFrame(tick);
  }

  function stopMotion() {
    cancelAnimationFrame(animation);
    cancelAnimationFrame(typer);
    clearTimeout(timer);
  }

  function go(i: number, animate: boolean) {
    stopMotion();
    index = Math.max(0, Math.min(steps.length - 1, i));
    const step = steps[index];
    const motion = animate && !reduceMotion;
    think(step, motion);
    if (motion && step.kind === "act") {
      render(false);
      animateAct(step, scheduleNext);
    } else if (motion && (step.kind === "imagine" || step.kind === "choose")) {
      render(false);
      animateEntrance(step, step.kind === "imagine" ? FAN_MS : CHOOSE_MS, scheduleNext);
    } else {
      render(true);
      scheduleNext();
    }
  }

  function scheduleNext() {
    if (!playing) return;
    if (index < steps.length - 1) {
      timer = window.setTimeout(() => go(index + 1, true), STEP_MS);
    } else if (loop) {
      timer = window.setTimeout(() => go(0, true), LOOP_PAUSE_MS);
    } else {
      setPlaying(false);
    }
  }

  function setPlaying(on: boolean) {
    playing = on;
    playButton.textContent = on ? "❚❚" : "▶";
    playButton.setAttribute("aria-label", on ? "Pause" : "Play");
    if (on) {
      if (index === steps.length - 1 && !loop) go(0, true);
      else scheduleNext();
    } else {
      stopMotion();
      render(true);
    }
  }

  if (transcript) {
    steps.forEach((step, i) => {
      const li = document.createElement("li");
      li.className = `step-${step.kind}`;
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = BADGES[step.kind];
      li.append(badge);
      for (const [text, cls] of [[step.caption, "say"], [step.detail, "detail"]]) {
        if (!text) continue;
        const p = document.createElement("p");
        p.className = cls;
        p.textContent = text;
        li.append(p);
      }
      li.tabIndex = 0;
      li.addEventListener("click", () => go(i, true));
      li.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          go(i, true);
        }
      });
      transcript.append(li);
    });
  }
  if (slider) {
    slider.max = String(steps.length - 1);
    slider.addEventListener("input", () => {
      setPlaying(false);
      go(Number(slider.value), false);
    });
  }
  $("[data-prev]")?.addEventListener("click", () => {
    setPlaying(false);
    go(index - 1, false);
  });
  $("[data-next]")?.addEventListener("click", () => {
    setPlaying(false);
    go(index + 1, true);
  });
  playButton.addEventListener("click", () => setPlaying(!playing));
  truth?.addEventListener("change", () => lastScene && draw({ ...lastScene, showTruth: truth.checked }));

  fitCanvas(canvas, () => (lastScene ? draw(lastScene) : render(true)));
  onThemeChange(() => lastScene && draw(lastScene));
  render(true);

  if (loop && !reduceMotion) {
    // only play while visible
    new IntersectionObserver(([entry]) => setPlaying(entry.isIntersecting), { threshold: 0.4 }).observe(root);
  }
}
