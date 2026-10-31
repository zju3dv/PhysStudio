"""Newton's official VBD rod solver with hydraulic material updates."""
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
import newton
from newton.solvers import SolverVBD

from .hydraulics import modulus


@wp.kernel
def damp_velocities(v: wp.array(dtype=wp.spatial_vector), damping: float):
    i = wp.tid()
    v[i] = v[i] * damping



class PlantSolver:
    def __init__(self, rig, scene, water, iterations=160, substeps=12, dt=.001, device='cuda:0', gravity=-9.81, contacts_enabled=True):
        self.rig, self.scene, self.device = rig, scene, device
        self.substeps, self.dt = substeps, dt
        self.contacts_enabled = contacts_enabled
        if substeps % 2:
            raise ValueError('Graph replay requires an even substep count.')
        self.centers = (rig['head'] + rig['tail']) * .5
        self.body_count = len(self.centers)
        builder = newton.ModelBuilder(gravity=(0., 0., gravity))
        nodes = np.vstack([rig['head'][0], rig['tail']])
        edges = [(0, 1)] + [(int(rig['parent'][i]) + 1, i + 1) for i in range(1, self.body_count)]
        rod = newton.Rod(nodes, edges=edges, radius=.004)
        cfg = newton.ModelBuilder.ShapeConfig(density=600., ke=3e5, kd=10., mu=.5)
        bodies, joints = builder.add_rod(rod=rod, cfg=cfg,
            stretch_stiffness=1e6, shear_stiffness=1e6, stretch_damping=10.,
            bend_stiffness=10., bend_damping=.002, twist_stiffness=20., twist_damping=.002,
            label=scene, body_frame_origin='com')
        assert bodies == list(range(self.body_count))
        shape_ids = list(range(self.body_count))
        self.mass, self.inertia = [], []
        for i in bodies:
            length, radius = float(rig['length'][i]), float(rig['radius'][i])
            dry_mass = np.pi * radius**2 * length * (900 if rig['leaf'][i] else 600)
            if rig['leaf'][i]:
                dry_mass += .08 * rig['leaf_area'][i]
            inertia = np.diag([length**2 / 12 + radius**2 / 4] * 2 + [radius**2 / 2])
            self.mass.append(dry_mass)
            self.inertia.append(inertia)
        builder.body_mass[0] = 0.
        builder.body_inv_mass[0] = 0.
        builder.body_inertia[0] = wp.mat33(0.)
        builder.body_inv_inertia[0] = wp.mat33(0.)
        # The pot and soil are fixed obstacles; short roots emerge through the soil.
        pot_height = float(rig['root_height'])
        pot = builder.add_shape_cylinder(body=-1,
            xform=wp.transform([0., 0., pot_height * .5], wp.quat_identity()),
            radius=dict(potted_plant=.23, ficus=.20, alocasia=.28)[scene],
            half_height=pot_height * .5, cfg=cfg)
        for i, shape in enumerate(shape_ids):
            if rig['head'][i, 2] < pot_height + .025:
                builder.shape_collision_filter_pairs.append((min(shape, pot), max(shape, pot)))
        # These capsules are an internal venation cage, not the leaf surface.
        # Contacts between overlapping internal support rods are not physical.
        for i in range(self.body_count):
            for j in range(i):
                builder.shape_collision_filter_pairs.append((shape_ids[j], shape_ids[i]))
        builder.add_ground_plane(cfg=cfg)
        builder.color()
        self.mass, self.inertia = np.array(self.mass), np.array(self.inertia)
        self.model = builder.finalize(device=device)
        self.rest_rotation = Rotation.from_quat(self.model.body_q.numpy()[:, 3:])
        self.joint_starts = self.model.joint_qd_start.numpy()
        self.joints = [(j, builder.joint_parent[j], builder.joint_child[j]) for j in joints]
        self.solver = SolverVBD(self.model, iterations=iterations,
            rigid_joint_linear_ke=1e7, rigid_joint_angular_ke=1e7,
            rigid_body_contact_buffer_size=128)
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        self.state0, self.state1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.set_water(water, water)
        self.advance(capture=False)
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            self.advance(capture=False)
        self.graph = capture.graph

    def set_water(self, water, initial):
        e = modulus(water / initial, self.rig['leaf'], self.scene)
        k = self.model.joint_target_ke.numpy()
        for j, parent, i in self.joints:
            radius = .5 * (self.rig['radius'][i] + self.rig['radius'][parent])
            length = .5 * (self.rig['length'][i] + self.rig['length'][parent])
            young = .5 * (e[i] + e[parent])
            bending = young * np.pi * radius**4 / (4 * length)
            start = self.joint_starts[j]
            k[start:start + 6] = [1e6, 1e6, 1e6, bending, bending, 2 * bending]
        self.model.joint_target_ke.assign(k)
        mass = self.mass + water * .001
        inv_mass = 1 / mass; inv_mass[0] = 0
        inertia = self.inertia * mass[:, None, None]
        inverse = np.linalg.inv(inertia); inverse[0] = 0; inertia[0] = 0; mass[0] = 0
        self.model.body_mass.assign(mass.astype(np.float32))
        self.model.body_inv_mass.assign(inv_mass.astype(np.float32))
        self.model.body_inertia.assign(inertia.astype(np.float32))
        self.model.body_inv_inertia.assign(inverse.astype(np.float32))
        self.solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def advance(self, capture=True):
        if capture:
            wp.capture_launch(self.graph)
            return
        for _ in range(self.substeps):
            self.state0.clear_forces()
            if self.contacts_enabled:
                self.pipeline.collide(self.state0, self.contacts)
            self.solver.step(self.state0, self.state1, self.control, self.contacts if self.contacts_enabled else None, self.dt)
            wp.launch(damp_velocities, self.body_count,
                      inputs=[self.state1.body_qd, float(np.exp(-8 * self.dt))], device=self.device)
            self.state0, self.state1 = self.state1, self.state0

    def poses(self):
        pose = self.state0.body_q.numpy()
        pose[:, 3:] = (Rotation.from_quat(pose[:, 3:]) * self.rest_rotation.inv()).as_quat()
        return pose

    def joint_error(self, poses):
        rotation = Rotation.from_quat(poses[:, 3:]).as_matrix()
        head = poses[:, :3] + np.einsum('nij,nj->ni', rotation, self.rig['head'] - self.centers)
        tail = poses[:, :3] + np.einsum('nij,nj->ni', rotation, self.rig['tail'] - self.centers)
        return np.linalg.norm(head[1:] - tail[self.rig['parent'][1:]], axis=1)
