"""Build a task MJCF from the unmodified official Panda description."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from physstudio.paths import data_path

ASSETS = data_path("robot_pouring", "franka_emika_panda")
HOME = np.array([0.0, -0.40, 0.0, -2.10, 0.0, 1.70, 0.7854, 0.025, 0.025])


def vessel(body, prefix, radius, height, wall, color, mass=0.0):
    """A floor plus 32 wall boxes; fluid uses the matching circular SDF."""
    common = dict(contype="1" if prefix == "cup" else "2", conaffinity="3" if prefix == "cup" else "1", rgba=color, friction="0.5 0.01 0.001")
    ET.SubElement(body, "geom", name=prefix + "_base", type="cylinder",
                  size=f"{radius + wall} {wall / 2}", pos=f"0 0 {-wall / 2}", mass=str(mass / 3), **common)
    for i in range(32):
        theta = 2 * math.pi * i / 32
        r = radius + wall / 2
        ET.SubElement(body, "geom", name=f"{prefix}_wall_{i}", type="box",
                      pos=f"{r * math.cos(theta)} {r * math.sin(theta)} {height / 2}",
                      size=f"{wall / 2} {(radius + wall) * math.tan(math.pi / 32)} {height / 2}",
                      euler=f"0 0 {theta}", mass=str(mass / 48), **common)


def make_model(cfg):
    path = ASSETS / "mjx_panda.xml"
    if not path.exists():
        raise FileNotFoundError("Run python -m physstudio fetch robot_pouring first")
    root = ET.parse(path).getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "assets"))
    option = ET.SubElement(root, "option", timestep=str(cfg.dt), integrator="implicitfast",
                           solver="Newton", iterations="12", ls_iterations="6", gravity="0 0 -9.81")
    # Contact is enabled for the original collision meshes, disabled for visuals.
    for node in root.findall(".//default[@class='collision']/geom"):
        node.set("contype", "1"); node.set("conaffinity", "3")
    for node in root.findall(".//default[@class='primitive_collision']/geom"):
        node.set("contype", "1"); node.set("conaffinity", "3")
    world = root.find("worldbody")
    for body in world.iter("body"):
        body.set("gravcomp", "1")
    ET.SubElement(world, "geom", name="table", type="plane", size="1 1 .02",
                  rgba=".13 .18 .21 1", contype="2", conaffinity="1")
    hand = root.find(".//body[@name='hand']")
    cup = ET.SubElement(hand, "body", name="cup", pos="0.105 0 0.17", quat="0 1 0 0", gravcomp="1")
    vessel(cup, "cup", cfg.cup_radius, cfg.cup_height, cfg.cup_wall, ".12 .55 .65 .45", mass=0.12)
    ET.SubElement(cup, "site", name="cup_origin", size=".003", pos="0 0 0")
    ET.SubElement(cup, "site", name="pour_lip", size=".003", pos=f"{cfg.cup_radius} 0 {cfg.cup_height}")
    target = ET.SubElement(world, "body", name="target", mocap="true", pos=f"{cfg.target_x} {cfg.target_y} .012")
    vessel(target, "target", cfg.target_radius, cfg.target_height, .008, ".72 .33 .14 1" if cfg.task == "flower" else ".76 .85 .87 1")
    if cfg.task == "flower":
        ET.SubElement(target, "geom", name="soil", type="cylinder", size=f"{cfg.target_radius} .025",
                      pos="0 0 .025", rgba=".13 .065 .025 1", contype="2", conaffinity="1")
        ET.SubElement(target, "geom", name="stem", type="cylinder", size=".007 .11",
                      pos="0 0 .16", rgba=".15 .25 .04 1", contype="2", conaffinity="1")
    # Explicitly exclude the grasped vessel from its wrist/fingers; the cup is a
    # fixed tool for the pouring task, not a separately learned grasp.
    contact = root.find("contact")
    for name in ("hand", "left_finger", "right_finger", "link7"):
        ET.SubElement(contact, "exclude", body1="cup", body2=name)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    return model, xml


def inverse_kinematics(model, position, rotation, q0=HOME, iterations=150):
    """Damped least-squares IK for reset/demo control, not physics stepping."""
    d = mujoco.MjData(model)
    d.qpos[:] = q0
    bid = model.body("cup").id
    lower, upper = model.jnt_range[:7].T
    for _ in range(iterations):
        mujoco.mj_forward(model, d)
        current = d.xmat[bid].reshape(3, 3)
        ep = np.asarray(position) - d.xpos[bid]
        er = Rotation.from_matrix(np.asarray(rotation) @ current.T).as_rotvec()
        if np.linalg.norm(ep) < 2e-5 and np.linalg.norm(er) < 2e-4:
            break
        jp = np.zeros((3, model.nv)); jr = jp.copy()
        mujoco.mj_jacBody(model, d, jp, jr, bid)
        j = np.concatenate([jp[:, :7], 0.35 * jr[:, :7]])
        e = np.r_[ep, 0.35 * er]
        dq = j.T @ np.linalg.solve(j @ j.T + 0.0002 * np.eye(6), e)
        d.qpos[:7] = np.clip(d.qpos[:7] + np.clip(dq, -.12, .12), lower + .015, upper - .015)
    mujoco.mj_forward(model, d)
    error = float(np.linalg.norm(np.asarray(position) - d.xpos[bid]))
    orientation_error = float(np.linalg.norm(Rotation.from_matrix(np.asarray(rotation) @ d.xmat[bid].reshape(3, 3).T).as_rotvec()))
    return d.qpos.copy(), error, orientation_error
