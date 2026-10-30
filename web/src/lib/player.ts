// Episode player: walks an episode step by step (imagine, push, ...) on the table canvas.

import { drawScene, fitCanvas, type Ghost, type Scene } from "./draw";
import { onThemeChange } from "./theme";
import { buildSteps, frameMap, loadEpisode, type Point, type Pose, type Step } from "./episode";

const STEP_MS = 2400; // pause on a step while playing
const ROW_MS = 110; // one trajectory row = 0.1 s of physics, shown slightly slower
const LOOP_PAUSE_MS = 3500;
const BADGES: Record<Step["kind"], string> = {
  start: "Task",
  imagine: "Imagine",
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

  let index = 0;
  let playing = false;
  let timer = 0;
  let animation = 0;
  let lastScene: Scene | null = null;

  const truePose = (id: string) => frames.get(id)!.true_pose;

  function ghost(id: string, strong: boolean): Ghost {
    const f = frames.get(id)!;
    return { pose: f.pose, truth: f.true_pose, label: id, strong, push: f.push ?? undefined };
  }

  function scene(step: Step, real: Pose, pusher?: Point): Scene {
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
      base.ghosts.push(...step.frames.map((id) => ghost(id, true)));
    }
    if (step.kind === "act") {
      base.realPush = frames.get(step.real)!.push ?? undefined;
      base.pusher = pusher;
    }
    return base;
  }

  function draw(s: Scene) {
    lastScene = s;
    drawScene(canvas, s);
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
    if (step.kind === "act" && !final) return;
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

  function stopMotion() {
    cancelAnimationFrame(animation);
    clearTimeout(timer);
  }

  function go(i: number, animate: boolean) {
    stopMotion();
    index = Math.max(0, Math.min(steps.length - 1, i));
    const step = steps[index];
    const moving = animate && step.kind === "act" && !reduceMotion;
    render(!moving);
    if (moving) animateAct(step as Extract<Step, { kind: "act" }>, scheduleNext);
    else scheduleNext();
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
