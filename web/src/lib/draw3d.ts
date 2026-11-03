// The 3D worlds (tabletop, billiards, the new room) drawn in 3D, like the MuJoCo clips: the same
// Scene as the flat drawing in draw.ts, on a lit table seen at an angle. Units are table pixels
// (1 px = 1 mm in MuJoCo): x to the right, y up, z toward the camera (the picture's y).

import * as THREE from "three";
import type { Ghost, Scene } from "./draw";
import { isBilliards, OUTLINE, TABLE, type Point, type State } from "./world";

// Sizes from src/jat/worlds: table3d/scene.py, tabletop/world.py, billiards/world.py
const T_HEIGHT = 20;
const BALL_RADIUS = 25;
const PUSHER_RADIUS = 15;
const PUSHER_HEIGHT = 30;
const RIM_INSET = 5;
const RIM_THICK = 20;
const FOVY = 50; // a little wider than the world model's 45°, so the whole table fits

interface Look {
  table: [string, string];
  rim: string;
  floor: string;
  light: [number, number, number]; // MuJoCo's light direction (z up)
  elevation: number; // camera, degrees above the table
  distance: number;
  shift: number;
}

// table3d/scene.py: Look and NEW_LOOK
const LOOKS: Record<string, Look> = {
  default: {
    table: ["#9e8061", "#947559"],
    rim: "#4d3829",
    floor: "#333840",
    light: [0.3, 0.5, -1],
    elevation: 62,
    distance: 770,
    shift: 0,
  },
  "tabletop-new-look": {
    table: ["#c7cccc", "#b3babd"],
    rim: "#26292e",
    floor: "#73594d",
    light: [-0.5, 0.3, -1],
    elevation: 57,
    distance: 800,
    shift: 30,
  },
};

interface View {
  world: string;
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  dynamic: THREE.Group; // rebuilt at every draw
  shapes: { t: THREE.BufferGeometry; tEdges: THREE.BufferGeometry; ball: THREE.SphereGeometry };
}

const views = new WeakMap<HTMLCanvasElement, View>();

function checker([a, b]: [string, string]): THREE.Texture {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 512;
  const ctx = canvas.getContext("2d")!;
  for (let i = 0; i < 16; i++) {
    for (let j = 0; j < 16; j++) {
      ctx.fillStyle = (i + j) % 2 ? a : b;
      ctx.fillRect(i * 32, j * 32, 32, 32);
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  return texture;
}

function makeView(canvas: HTMLCanvasElement, world: string): View {
  const look = LOOKS[world] ?? LOOKS.default;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(look.floor);
  const c = TABLE / 2;

  const e = (look.elevation * Math.PI) / 180;
  const camera = new THREE.PerspectiveCamera(FOVY, 1, 10, 5000);
  camera.position.set(c + look.shift, look.distance * Math.sin(e), c + look.distance * Math.cos(e));
  camera.lookAt(c, 0, c);

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2);
  const [lx, ly, lz] = look.light; // MuJoCo (x, y, z up) to ours (x, z up, -y)
  const dir = new THREE.Vector3(lx, lz, -ly).normalize();
  sun.position.set(c, 0, c).addScaledVector(dir, -900);
  sun.target.position.set(c, 0, c);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -400, right: 400, top: 400, bottom: -400, near: 100, far: 2000 });
  sun.shadow.bias = -0.0005;
  scene.add(sun, sun.target);

  const table = new THREE.Mesh(
    new THREE.BoxGeometry(TABLE, 10, TABLE),
    [0, 0, 0, 0, 0, 0].map((_, i) =>
      i === 2
        ? new THREE.MeshStandardMaterial({ map: checker(look.table), roughness: 0.9 })
        : new THREE.MeshStandardMaterial({ color: look.table[1], roughness: 0.9 }),
    ),
  );
  table.position.set(c, -5, c);
  table.receiveShadow = true;
  scene.add(table);

  const rimHeight = isBilliards(world) ? 35 : 20;
  const rimMaterial = new THREE.MeshStandardMaterial({ color: look.rim, roughness: 0.7 });
  const long = TABLE + 2 * (RIM_THICK - RIM_INSET);
  for (const [x, z, w, d] of [
    [RIM_INSET - RIM_THICK / 2, c, RIM_THICK, long],
    [TABLE - RIM_INSET + RIM_THICK / 2, c, RIM_THICK, long],
    [c, RIM_INSET - RIM_THICK / 2, long, RIM_THICK],
    [c, TABLE - RIM_INSET + RIM_THICK / 2, long, RIM_THICK],
  ]) {
    const rim = new THREE.Mesh(new THREE.BoxGeometry(w, rimHeight, d), rimMaterial);
    rim.position.set(x, rimHeight / 2, z);
    rim.castShadow = rim.receiveShadow = true;
    scene.add(rim);
  }

  // the T: its outline in table pixels, lifted from the table plane into a slab
  const shape = new THREE.Shape(OUTLINE.map(([x, y]) => new THREE.Vector2(x, y)));
  const t = new THREE.ExtrudeGeometry(shape, { depth: T_HEIGHT, bevelEnabled: false });
  t.rotateX(Math.PI / 2); // outline y -> z, the slab hangs below
  t.translate(0, T_HEIGHT, 0);

  const dynamic = new THREE.Group();
  scene.add(dynamic);
  return {
    world,
    renderer,
    scene,
    camera,
    dynamic,
    shapes: { t, tEdges: new THREE.EdgesGeometry(t), ball: new THREE.SphereGeometry(BALL_RADIUS, 40, 24) },
  };
}

function colors(canvas: HTMLCanvasElement) {
  const style = getComputedStyle(canvas);
  const v = (name: string) => style.getPropertyValue(name).trim() || "#888";
  return {
    real: v("--t-real"),
    white: v("--ball-white"),
    red: v("--ball-red"),
    goal: v("--goal"),
    ghost: v("--ghost"),
    chosen: v("--chosen"),
    truth: v("--truth"),
    pusher: v("--pusher"),
  };
}

/** The pieces of a state: one T, or the white and the red ball. */
function place(view: View, state: State, material: THREE.Material, lift: number, only?: number): THREE.Object3D[] {
  if (isBilliards(view.world)) {
    return [0, 1]
      .filter((i) => only === undefined || i === only)
      .map((i) => {
        const ball = new THREE.Mesh(view.shapes.ball, Array.isArray(material) ? material[i] : material);
        ball.position.set(state[2 * i], BALL_RADIUS + lift, state[2 * i + 1]);
        return ball;
      });
  }
  const t = new THREE.Mesh(view.shapes.t, material);
  t.position.set(state[0], lift, state[1]);
  t.rotation.y = -state[2];
  return [t];
}

function outline(view: View, state: State, color: string, lift: number, only?: number): THREE.Object3D[] {
  const material = new THREE.LineBasicMaterial({ color, transparent: true });
  if (isBilliards(view.world)) {
    return [0, 1]
      .filter((i) => only === undefined || i === only)
      .map((i) => {
        const ring = new THREE.LineLoop(
          new THREE.BufferGeometry().setFromPoints(
            Array.from({ length: 48 }, (_, k) => {
              const a = (k / 48) * Math.PI * 2;
              return new THREE.Vector3(Math.cos(a) * BALL_RADIUS, 0, Math.sin(a) * BALL_RADIUS);
            }),
          ),
          material,
        );
        ring.position.set(state[2 * i], 1 + lift, state[2 * i + 1]);
        return ring;
      });
  }
  const edges = new THREE.LineSegments(view.shapes.tEdges, material);
  edges.position.set(state[0], lift, state[1]);
  edges.rotation.y = -state[2];
  return [edges];
}

/** A flat arrow lying on the table. */
function arrow(from: Point, to: Point, color: string, width: number, opacity = 1): THREE.Object3D {
  const length = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const group = new THREE.Group();
  const material = new THREE.MeshBasicMaterial({ color, transparent: opacity < 1, opacity, side: THREE.DoubleSide });
  const head = Math.min(18, length / 2);
  const shaft = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(0.1, length - head), width), material);
  shaft.rotation.x = -Math.PI / 2;
  shaft.position.x = (length - head) / 2;
  const tip = new THREE.Mesh(
    new THREE.ShapeGeometry(new THREE.Shape([new THREE.Vector2(0, -head / 2), new THREE.Vector2(head, 0), new THREE.Vector2(0, head / 2)])),
    material,
  );
  tip.rotation.x = -Math.PI / 2;
  tip.position.x = length - head;
  group.add(shaft, tip);
  group.position.set(from[0], 1.5, from[1]);
  group.rotation.y = -Math.atan2(to[1] - from[1], to[0] - from[0]);
  return group;
}

/** A bright ring on the table around a ball. */
function ring(x: number, z: number, color: string, opacity: number): THREE.Object3D {
  const mesh = new THREE.Mesh(
    new THREE.RingGeometry(BALL_RADIUS + 1, BALL_RADIUS + 8, 48),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity, side: THREE.DoubleSide, depthWrite: false }),
  );
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.set(x, 1.2, z);
  return mesh;
}

/** A dashed path on the table from a real ball to where it is imagined. */
function trail(from: Point, to: Point, color: string, opacity: number): THREE.Object3D {
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(from[0], 1.3, from[1]), new THREE.Vector3(to[0], 1.3, to[1])]),
    new THREE.LineDashedMaterial({ color, dashSize: 10, gapSize: 7, transparent: true, opacity: 0.9 * opacity }),
  );
  line.computeLineDistances();
  return line;
}

function clear(group: THREE.Group) {
  group.traverse((o) => {
    if (o instanceof THREE.Mesh || o instanceof THREE.Line) {
      const shared = o.geometry instanceof THREE.ExtrudeGeometry || o.geometry instanceof THREE.SphereGeometry || o.geometry instanceof THREE.EdgesGeometry;
      if (!shared) o.geometry.dispose();
      (Array.isArray(o.material) ? o.material : [o.material]).forEach((m: THREE.Material) => m.dispose());
    }
  });
  group.clear();
}

export function drawScene3d(canvas: HTMLCanvasElement, scene: Scene) {
  if (canvas.width === 0) return;
  let view = views.get(canvas);
  if (!view || view.world !== scene.world) {
    view?.renderer.dispose();
    view = makeView(canvas, scene.world);
    views.set(canvas, view);
  }
  const c = colors(canvas);
  const balls = isBilliards(scene.world);
  const add = (...objects: THREE.Object3D[]) => view!.dynamic.add(...objects);
  clear(view.dynamic);

  // the real objects, solid and casting shadows
  const solid = balls
    ? [c.white, c.red].map((color) => new THREE.MeshStandardMaterial({ color, roughness: 0.3 }))
    : new THREE.MeshStandardMaterial({ color: c.real, roughness: 0.6 });
  for (const o of place(view, scene.real, solid as THREE.Material, 0)) {
    o.castShadow = true;
    add(o);
  }

  // see-through layers, each a little higher so overlapping ones do not flicker
  let lift = 0.4;
  const layer = (state: State, color: string, opacity: number, edge: number, only?: number) => {
    const material = new THREE.MeshStandardMaterial({ color, transparent: true, opacity, depthWrite: false, roughness: 0.8 });
    add(...place(view!, state, material, lift, only));
    add(...outline(view!, state, color, lift + 0.2, only).map((o) => {
      ((o as THREE.Line).material as THREE.LineBasicMaterial).opacity = edge;
      return o;
    }));
    lift += 0.4;
  };

  if (scene.goal) layer(scene.goal, c.goal, 0.3, 1, balls ? 1 : undefined);
  const ghosts: Ghost[] = [...scene.ghosts.filter((g) => !g.chosen), ...scene.ghosts.filter((g) => g.chosen)];
  for (const g of ghosts) {
    const fade = g.alpha ?? 1;
    const color = g.chosen ? c.chosen : c.ghost;
    if (g.push) add(arrow(g.push.start, g.push.end, color, g.chosen ? 5 : 3, fade * (g.chosen || g.strong ? 0.9 : 0.5)));
    if (balls) {
      // each imagined ball keeps its own color, so white and red stay apart; a ring on the
      // table says imagined (or kept), and a trail joins it to where the real ball is
      const opacity = fade * (g.chosen || g.strong ? 0.75 : 0.35);
      [c.white, c.red].forEach((own, i) => {
        const from: Point = [scene.real[2 * i], scene.real[2 * i + 1]];
        const to: Point = [g.pose[2 * i], g.pose[2 * i + 1]];
        const moved = Math.hypot(to[0] - from[0], to[1] - from[1]);
        if (moved < 6) return; // a ball that stays put is the real one: no ghost on top of it
        const material = new THREE.MeshStandardMaterial({ color: own, transparent: true, opacity, depthWrite: false, roughness: 0.4 });
        add(...place(view!, g.pose, material, lift, i), ring(g.pose[2 * i], g.pose[2 * i + 1], color, fade));
        if (moved > BALL_RADIUS) add(trail(from, to, own === c.white ? "#f4f2ea" : own, fade));
      });
      lift += 0.4;
      continue;
    }
    layer(g.pose, color, fade * (g.chosen ? 0.5 : g.strong ? 0.35 : 0.15), fade * (g.strong || g.chosen ? 1 : 0.5));
    if (scene.showTruth && g.truth) add(...outline(view, g.truth, c.truth, lift));
  }

  if (scene.realPush) add(arrow(scene.realPush.start, scene.realPush.end, c.pusher, 4));
  if (scene.pusher) {
    const pusher = new THREE.Mesh(
      new THREE.CylinderGeometry(PUSHER_RADIUS, PUSHER_RADIUS, PUSHER_HEIGHT, 32),
      new THREE.MeshStandardMaterial({ color: c.pusher, roughness: 0.4 }),
    );
    pusher.position.set(scene.pusher[0], PUSHER_HEIGHT / 2 + 1, scene.pusher[1]);
    pusher.castShadow = true;
    add(pusher);
  }

  view.renderer.setSize(canvas.width, canvas.height, false);
  view.renderer.render(view.scene, view.camera);
}

/** The table point under the pointer, or null off the table. */
export function tablePoint3d(canvas: HTMLCanvasElement, clientX: number, clientY: number): Point | null {
  const view = views.get(canvas);
  if (!view) return null;
  const box = canvas.getBoundingClientRect();
  const ndc = new THREE.Vector2(((clientX - box.left) / box.width) * 2 - 1, -((clientY - box.top) / box.height) * 2 + 1);
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, view.camera);
  const hit = ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), new THREE.Vector3());
  return hit ? [hit.x, hit.z] : null;
}
