// What differs between worlds on the site: how a state is drawn, described and compared. Every
// world is a 512 x 512 px table seen from above (src/jat/worlds/base.py). The T worlds (PushT,
// the tabletop, its new look) store x, y, angle; billiards stores the white and the red ball.

export type State = number[];
export type Point = [number, number];

export const TABLE = 512;

// Same T geometry as src/jat/worlds/tblock/shape.py, so ghosts and real T's line up exactly
// with what the Python side computed.
export const OUTLINE: Point[] = [
  [-60, 0], [60, 0], [60, 30], [15, 30], [15, 120], [-15, 120], [-15, 30], [-60, 30],
];
const T_CENTER: Point = [0, 45];
const BALL_RADIUS = 25; // src/jat/worlds/billiards/world.py
const RED = 1; // the ball the goal is about

export const isBilliards = (world: string) => world === "billiards";
/** Worlds simulated in 3D (MuJoCo): the site draws them in 3D too. */
export const is3D = (world: string) => world !== "pusht";

export function toWorld([x, y, theta]: State, [lx, ly]: Point): Point {
  const c = Math.cos(theta);
  const s = Math.sin(theta);
  return [x + c * lx - s * ly, y + s * lx + c * ly];
}

const ball = (state: State, i: number): Point => [state[2 * i], state[2 * i + 1]];

/** One path per object: the T, or the white then the red ball. */
export function parts(world: string, state: State): Path2D[] {
  if (isBilliards(world)) {
    return [0, 1].map((i) => {
      const path = new Path2D();
      path.arc(...ball(state, i), BALL_RADIUS, 0, Math.PI * 2);
      return path;
    });
  }
  const path = new Path2D();
  OUTLINE.forEach((p, i) => (i ? path.lineTo(...toWorld(state, p)) : path.moveTo(...toWorld(state, p))));
  path.closePath();
  return [path];
}

/** The objects a goal is about: in billiards only the red ball's position counts. */
export function goalParts(world: string, goal: State): Path2D[] {
  const all = parts(world, goal);
  return isBilliards(world) ? [all[RED]] : all;
}

/** Where a frame's label goes. */
export function anchor(world: string, state: State): Point {
  return isBilliards(world) ? ball(state, RED) : toWorld(state, T_CENTER);
}

/** The goal as the agent was told it (goal_text on the Python side; the log does not store it). */
export function goalText(world: string, goal: State): string {
  const [cx, cy] = anchor(world, goal).map(Math.round);
  if (isBilliards(world)) return `Put the red ball at (${cx}, ${cy}). The white ball may end anywhere.`;
  const degrees = ((Math.round((goal[2] * 180) / Math.PI) % 360) + 360) % 360;
  return `Move the T to the green outline: center (${cx}, ${cy}), angle ${degrees}°.`;
}

/** How far apart two states are, in words. */
export function gapText(world: string, a: State, b: State): string {
  if (isBilliards(world)) {
    const [w, r] = [0, 1].map((i) => Math.round(Math.hypot(...ball(a, i).map((v, k) => v - ball(b, i)[k]))));
    return `red ${r} px, white ${w} px`;
  }
  const [ax, ay] = anchor(world, a);
  const [bx, by] = anchor(world, b);
  const turn = Math.abs(((((a[2] - b[2] + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI);
  return `${Math.round(Math.hypot(ax - bx, ay - by))} px, ${Math.round((turn * 180) / Math.PI)}°`;
}

/** The success rule, for captions. */
export function successText(world: string): string {
  return isBilliards(world) ? "success needs the red ball < 20 px from the goal" : "success needs < 20 px and < 15°";
}
