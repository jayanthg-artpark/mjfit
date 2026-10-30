"""A Franka Panda placing a 96-well plate into an Opentrons Flex deck slot.

    python examples/flex_slot/flex_slot.py --view                  # watch one grid search
    python examples/flex_slot/flex_slot.py --episodes 50 --error 3 --chamfer 1

The slot's locators come from lab-sentry's Flex model (deck_insert.obj). Every rib,
wall and corner pad there is a box or a convex prism, so each becomes one convex geom
with no decomposition. lab-sentry's ribs leave a 140 x 93 mm pocket, estimated from
photos; ``--pocket opentrons`` (the default) moves each corner in to the 128 x 86 mm slot
of Opentrons' deck definition, ``--pocket labsentry`` keeps the mesh as drawn.
``--chamfer`` bevels the top inner edge of every rib: the real lead-in is unmeasured.

The stock Franka Hand opens to 80 mm, narrower than the plate's 85.5 mm, so both fingers
sit FINGER_OFFSET further out, like printed lab fingers. The arm runs a Cartesian
impedance controller with gravity compensation, as SERL does. The robot believes the
slot is off by a random ``--error`` (mm) and ``--yaw`` (degrees), then either drops the
plate straight down or grid-searches around its belief.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np

DECK_INSERT = 'device-designer/models/opentrons_flex/meshes/deck_insert.obj'
OPENTRONS_POCKET = (0.128, 0.086)    # slot boundingBox in Opentrons' ot3_standard.json

# Corning 96 Well Plate 360 uL Flat: SBS footprint, skirt and inset upper body.
PLATE_SIZE = (0.12776, 0.08547, 0.01422)
PLATE_SKIRT = 0.0061                  # ANSI/SLAS flange height
PLATE_INSET = 0.0016                  # upper body inset per side
PLATE_RADIUS = 0.00318                # footprint corner radius
PLATE_MASS = 0.05
GRIP_HEIGHT = 0.0102                  # tool point above the plate's bottom

FINGER_OFFSET = 0.006                 # each finger moved out, m
SLOT_POS = np.array([0.5, 0.0, 0.0])  # slot center on the deck surface, world frame
CONTACT = dict(solref=[0.002, 1], solimp=[0.99, 0.999, 0.001, 0.5, 2], condim=4)
LOCATOR_RGBA = {'locator_clear': [0.75, 0.88, 1.0, 0.6], 'clip_blue': [0.15, 0.45, 0.95, 1.0]}

# Cartesian impedance, SERL-style: translation, rotation, posture.
KP, KD, KR, DR, KQ, DQ = 1500.0, 80.0, 40.0, 3.0, 10.0, 1.0
MAX_PUSH = 0.008                      # m of position error, so at most KP * MAX_PUSH N
GRASP_FORCE = 80.0                    # N on the finger tendon


def read_obj_parts(path):
    """{object name: (material, unique vertices)} of an OBJ file."""
    verts, mats, name = defaultdict(list), {}, None
    for line in open(path):
        if line.startswith('o '):
            name = line.split()[1]
        elif line.startswith('usemtl '):
            mats[name] = line.split()[1]
        elif line.startswith('v '):
            verts[name].append([float(t) for t in line.split()[1:4]])
    return {k: (mats.get(k), np.unique(np.round(v, 7), axis=0)) for k, v in verts.items()}


def locators(lab_sentry, pocket, chamfer):
    """Convex parts of the four corner locators, each as (kind, material, vertices)."""
    parts = [(m, v) for m, v in read_obj_parts(Path(lab_sentry) / DECK_INSERT).values()
             if m in LOCATOR_RGBA]
    ribs = [(m, v) for m, v in parts if len(v) == 8 and np.isclose(v[:, 2].max(), 0.013)]
    # A rib faces the plate along the axis it sticks out on, its longer side.
    def facing(v):
        return int(np.ptp(v[:, 1]) > np.ptp(v[:, 0]))
    drawn = [min(np.abs(v[:, a]).min() for _, v in ribs if facing(v) == a) for a in (0, 1)]
    half = drawn if pocket is None else [pocket[0] / 2, pocket[1] / 2]
    out = []
    for m, v in parts:
        sign = np.sign(v[:, :2].mean(0))
        v = v + np.r_[(np.array(half) - drawn) * sign, 0.0]
        if len(v) == 8 and np.isclose(v[:, 2].max(), 0.013) and chamfer > 0:
            v = chamfered(v, facing(v), chamfer)
        out.append((m, v))
    return out, half


def chamfered(box, axis, c):
    """The box with the top edge of its face nearest the slot center bevelled by c."""
    lo, hi = box.min(0), box.max(0)
    inner = lo[axis] if np.mean(box[:, axis]) > 0 else hi[axis]
    out = 1.0 if inner == lo[axis] else -1.0
    c = min(c, hi[axis] - lo[axis], hi[2] - lo[2])
    keep = box[~((box[:, axis] == inner) & (box[:, 2] == hi[2]))]
    cut = box[(box[:, axis] == inner) & (box[:, 2] == hi[2])]
    low, back = cut.copy(), cut.copy()
    low[:, 2] -= c
    back[:, axis] += out * c
    return np.vstack([keep, low, back])


def rounded_rect(w, h, r, n=6):
    """Counterclockwise rectangle with corner arcs of radius r."""
    pts = []
    for k, (cx, cy) in enumerate([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        a = np.linspace(k * np.pi / 2, (k + 1) * np.pi / 2, n)
        pts += list(np.c_[cx * (w / 2 - r) + r * np.cos(a), cy * (h / 2 - r) + r * np.sin(a)])
    return np.array(pts)


def prism(poly, z0, z1):
    return np.vstack([np.c_[poly, np.full(len(poly), z0)], np.c_[poly, np.full(len(poly), z1)]])


def build(lab_sentry, pocket=OPENTRONS_POCKET, chamfer=0.0):
    """The compiled model and the slot's pocket half-sizes."""
    from robot_descriptions import panda_mj_description

    spec = mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)
    spec.option.timestep = 0.001
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = 3
    world = spec.worldbody
    world.add_light(pos=[0.5, 0, 1.5], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, -0.002],
                   rgba=[0.3, 0.3, 0.32, 1])

    # Torque control for the arm joints; the gripper squeezes with a fixed force.
    for act in spec.actuators:
        act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_NONE
        act.gainprm[:3] = [1, 0, 0]
        act.biasprm[:3] = [0, 0, 0]
        act.ctrlrange = act.forcerange
    for body, side in (('left_finger', 1), ('right_finger', -1)):
        spec.body(body).pos[1] += side * FINGER_OFFSET
    spec.body('hand').add_site(name='tcp', pos=[0, 0, 0.0584 + 0.0445])

    slot = world.add_body(name='slot', pos=SLOT_POS)
    slot.add_geom(name='deck', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.08075, 0.0525, 0.001],
                  pos=[0, 0, -0.001], rgba=[0.55, 0.57, 0.6, 1], friction=[0.3, 0.005, 0.0001])
    parts, half = locators(lab_sentry, pocket, chamfer)
    for k, (mat, verts) in enumerate(parts):
        spec.add_mesh(name=f'locator_{k}', uservert=verts.ravel().tolist())
        slot.add_geom(name=f'locator_{k}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'locator_{k}',
                      rgba=LOCATOR_RGBA[mat], friction=[0.3, 0.005, 0.0001], **CONTACT)

    length, width, height = PLATE_SIZE
    skirt = rounded_rect(length, width, PLATE_RADIUS)
    body = rounded_rect(length - 2 * PLATE_INSET, width - 2 * PLATE_INSET, PLATE_RADIUS)
    spec.add_mesh(name='plate_skirt', uservert=prism(skirt, 0, PLATE_SKIRT).ravel().tolist())
    spec.add_mesh(name='plate_body', uservert=prism(body, PLATE_SKIRT, height).ravel().tolist())
    plate = world.add_body(name='plate', pos=SLOT_POS + [0, 0, 0.05])
    plate.add_freejoint(name='plate_joint')
    v_skirt, v_body = PLATE_SKIRT * length * width, (height - PLATE_SKIRT) * length * width
    for name, vol in (('plate_skirt', v_skirt), ('plate_body', v_body)):
        plate.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                       mass=PLATE_MASS * vol / (v_skirt + v_body), rgba=[0.92, 0.93, 0.95, 1],
                       friction=[0.3, 0.005, 0.0001], **CONTACT)
    return spec.compile(), np.array(half)


def yaw_mat(yaw):
    """Hand pointing down, fingers closing along the world y axis turned by yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def rot_error(r_des, r):
    """World-frame rotation vector taking r to r_des."""
    q_des, q, err = np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q_des, r_des.ravel())
    mujoco.mju_mat2Quat(q, r.ravel())
    mujoco.mju_subQuat(err, q_des, q)
    return r @ err


class Arm:
    """Cartesian impedance control of the Panda's tool point, at every physics step."""

    def __init__(self, model, data):
        self.m, self.d = model, data
        self.tcp = model.site('tcp').id
        self.q0 = model.key('home').qpos[:7].copy()
        self.jp, self.jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        self.target, self.r_target = None, None
        self.kp, self.kr = np.full(3, KP), np.full(3, KR)   # world axes

    def stiffness(self, xy=KP, yaw=KR):
        """Lateral and yaw stiffness; damping follows the square root."""
        self.kp[:2], self.kr[2] = xy, yaw

    def pose(self):
        return self.d.site_xpos[self.tcp].copy(), self.d.site_xmat[self.tcp].reshape(3, 3).copy()

    def ik(self, pos, rot, iters=300):
        """Joint positions putting the tool point at pos, rot (damped least squares)."""
        m, d = self.m, self.d
        for _ in range(iters):
            mujoco.mj_fwdPosition(m, d)
            x, r = self.pose()
            err = np.r_[pos - x, rot_error(rot, r)]
            if np.linalg.norm(err) < 1e-7:
                break
            mujoco.mj_jacSite(m, d, self.jp, self.jr, self.tcp)
            j = np.r_[self.jp, self.jr][:, :7]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), err)
            dq *= min(1.0, 0.1 / np.abs(dq).max())
            d.qpos[:7] = np.clip(d.qpos[:7] + dq, *m.jnt_range[:7].T)
        return d.qpos[:7].copy()

    def control(self):
        m, d = self.m, self.d
        x, r = self.pose()
        mujoco.mj_jacSite(m, d, self.jp, self.jr, self.tcp)
        jp, jr = self.jp[:, :7], self.jr[:, :7]
        dq = d.qvel[:7]
        dx = np.clip(self.target - x, -MAX_PUSH, MAX_PUSH)
        force = self.kp * dx - KD * np.sqrt(self.kp / KP) * (jp @ dq)
        torque = self.kr * rot_error(self.r_target, r) - DR * np.sqrt(self.kr / KR) * (jr @ dq)
        j = np.r_[jp, jr]
        null = np.eye(7) - j.T @ np.linalg.pinv(j.T)
        tau = jp.T @ force + jr.T @ torque + null @ (KQ * (self.q0 - d.qpos[:7]) - DQ * dq)
        d.ctrl[:7] = tau + d.qfrc_bias[:7]
        d.ctrl[7] = -GRASP_FORCE


class Episode:
    """One placement: the robot believes the slot is at SLOT_POS + error, turned by yaw."""

    def __init__(self, model, half, error_xy, error_yaw, compliance=(KP, KR), viewer=None,
                 time_limit=60.0):
        self.m, self.half, self.viewer, self.time_limit = model, half, viewer, time_limit
        self.compliance = compliance  # xy and yaw stiffness while descending
        self.d = mujoco.MjData(model)
        self.arm = Arm(model, self.d)
        self.belief = SLOT_POS + np.r_[error_xy, 0.0]
        self.rot = yaw_mat(error_yaw)
        self.seat_z = SLOT_POS[2] + 0.0022 + GRIP_HEIGHT  # tool point with the plate seated
        self.hover_z = SLOT_POS[2] + 0.013 + 0.003 + GRIP_HEIGHT
        self.probes = 0

        d = self.d
        d.qpos[:9] = model.key('home').qpos[:9]
        start = np.r_[self.belief[:2], self.hover_z]
        d.qpos[:7] = self.arm.ik(start, self.rot)
        d.qpos[7:9] = PLATE_SIZE[1] / 2 - FINGER_OFFSET + 0.0003  # inner pads on the skirt
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (self.rot @ yaw_mat(0).T).ravel())
        adr = model.joint('plate_joint').qposadr[0]
        d.qpos[adr:adr + 7] = np.r_[start - [0, 0, GRIP_HEIGHT], quat]
        mujoco.mj_forward(model, d)
        self.arm.target, self.arm.r_target = start.copy(), self.rot
        self.wait(0.3)

    def step(self):
        self.arm.control()
        mujoco.mj_step(self.m, self.d)
        if self.viewer is not None and self.d.time % 0.02 < self.m.opt.timestep:
            self.viewer.sync()
            time.sleep(0.02)
        if self.d.time > self.time_limit:
            raise TimeoutError

    def wait(self, seconds):
        end = self.d.time + seconds
        while self.d.time < end:
            self.step()

    def move(self, goal, speed):
        start = self.arm.target.copy()
        n = max(1, int(np.linalg.norm(goal - start) / speed / self.m.opt.timestep))
        for k in range(1, n + 1):
            self.arm.target = start + (goal - start) * k / n
            self.step()

    def probe(self, xy):
        """Descend at xy (belief frame); True if the plate went all the way down."""
        self.probes += 1
        self.move(np.r_[xy, self.hover_z], 0.05)
        self.arm.stiffness(*self.compliance)
        self.move(np.r_[xy, self.seat_z - 0.004], 0.03)  # pushes at most KP * 4 mm = 6 N
        self.wait(0.2)
        seated = self.arm.pose()[0][2] < self.seat_z + 0.0015
        if not seated:
            self.move(np.r_[xy, self.hover_z], 0.05)
        self.arm.stiffness()
        return seated

    def seated(self):
        """Ground truth: the plate rests on the corner pads inside the locators."""
        p = self.d.body('plate')
        down = p.xmat.reshape(3, 3)[:, 2] @ [0, 0, 1]
        return (p.xpos[2] - SLOT_POS[2] < 0.0022 + 0.0008 and down > np.cos(np.radians(1))
                and np.all(np.abs(p.xpos[:2] - SLOT_POS[:2]) < self.half - np.array(PLATE_SIZE[:2]) / 2 + 5e-4))

    def run(self, points):
        try:
            for xy in points:
                if self.probe(self.belief[:2] + xy):
                    break
        except TimeoutError:
            pass
        return self.seated()


def grid(step, radius):
    """Square grid offsets within radius, nearest first, like a search around a belief."""
    n = int(radius / step)
    pts = [np.array([i, j]) * step for i in range(-n, n + 1) for j in range(-n, n + 1)
           if np.hypot(i, j) * step <= radius + 1e-12]
    return sorted(pts, key=lambda p: (np.hypot(*p), np.arctan2(p[1], p[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lab-sentry', default='D:/lab-sentry', help='lab-sentry checkout')
    parser.add_argument('--pocket', choices=['opentrons', 'labsentry'], default='opentrons')
    parser.add_argument('--chamfer', type=float, default=0.0, help='rib lead-in, mm')
    parser.add_argument('--error', type=float, default=2.0, help='largest xy belief error, mm')
    parser.add_argument('--yaw', type=float, default=0.3, help='largest yaw belief error, deg')
    parser.add_argument('--strategy', choices=['straight', 'grid', 'both'], default='both')
    parser.add_argument('--xy-stiffness', type=float, default=KP, help='while descending, N/m')
    parser.add_argument('--yaw-stiffness', type=float, default=KR, help='while descending, N m/rad')
    parser.add_argument('--step', type=float, default=1.0, help='grid step, mm')
    parser.add_argument('--radius', type=float, default=4.0, help='grid radius, mm')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--view', action='store_true', help='watch one episode')
    args = parser.parse_args()

    pocket = OPENTRONS_POCKET if args.pocket == 'opentrons' else None
    model, half = build(args.lab_sentry, pocket, args.chamfer * 1e-3)
    gap = (half - np.array(PLATE_SIZE[:2]) / 2) * 1e3
    print(f'pocket {2 * half[0] * 1e3:.2f} x {2 * half[1] * 1e3:.2f} mm, clearance per side '
          f'{gap[0]:.2f} mm in x and {gap[1]:.2f} mm in y, chamfer {args.chamfer} mm')

    compliance = (args.xy_stiffness, args.yaw_stiffness)
    strategies = {'straight': [np.zeros(2)], 'grid': grid(args.step * 1e-3, args.radius * 1e-3)}
    if args.strategy != 'both':
        strategies = {args.strategy: strategies[args.strategy]}
    rng = np.random.default_rng(args.seed)
    errors = []
    for _ in range(1 if args.view else args.episodes):
        r, a = args.error * 1e-3 * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
        errors.append((r * np.array([np.cos(a), np.sin(a)]), np.radians(rng.uniform(-1, 1) * args.yaw)))

    if args.view:
        import mujoco.viewer
        name, points = next(iter(strategies.items())) if args.strategy != 'both' else ('grid', strategies['grid'])
        ep = Episode(model, half, *errors[0], compliance)
        with mujoco.viewer.launch_passive(model, ep.d) as viewer:
            ep.viewer = viewer
            ok = ep.run(points)
            print(f'{name}: {"seated" if ok else "not seated"} after {ep.probes} probes, {ep.d.time:.1f} s')
            ep.time_limit = np.inf
            while viewer.is_running():
                ep.wait(0.02)
        return

    for name, points in strategies.items():
        results = []
        for xy, yaw in errors:
            ep = Episode(model, half, xy, yaw, compliance)
            results.append((ep.run(points), ep.probes, ep.d.time))
        ok = np.array([r[0] for r in results])
        probes = np.array([r[1] for r in results])
        times = np.array([r[2] for r in results])
        print(f'{name:9s} seated {ok.sum()}/{len(ok)}'
              + (f', mean {probes[ok].mean():.1f} probes and {times[ok].mean():.1f} s when seated'
                 if ok.any() else ''))


if __name__ == '__main__':
    main()
