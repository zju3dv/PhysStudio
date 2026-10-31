"""Warp Poisson solver and bipolar, resistance-aware dielectric breakdown.

Implements the discharge equations (9--12) of Thunderstruck. Quantities are
nondimensional. Scene demos prescribe cloud charge reservoirs; the atmospheric
six-species microphysics is independent of this discharge solver.
"""

from dataclasses import dataclass, asdict
import itertools
import time
import numpy as np
import warp as wp


@wp.kernel
def relax(
    phi: wp.array3d(dtype=float),
    rho: wp.array3d(dtype=float),
    fixed: wp.array3d(dtype=int),
    values: wp.array3d(dtype=float),
    dx: float,
    parity: int,
    omega: float,
):
    i, j, k = wp.tid()
    if fixed[i, j, k] == 0 and (i + j + k) % 2 == parity:
        total = float(0.0)
        count = float(0.0)
        # Missing neighbors encode homogeneous Neumann side/top boundaries.
        for a in range(3):
            for d in range(2):
                ii, jj, kk = i, j, k
                offset = 2 * d - 1
                if a == 0:
                    ii += offset
                elif a == 1:
                    jj += offset
                else:
                    kk += offset
                if (
                    ii >= 0
                    and jj >= 0
                    and kk >= 0
                    and ii < phi.shape[0]
                    and jj < phi.shape[1]
                    and kk < phi.shape[2]
                ):
                    total += phi[ii, jj, kk]
                    count += 1.0
        target = (total + rho[i, j, k] * dx * dx) / count
        phi[i, j, k] += omega * (target - phi[i, j, k])


@wp.kernel
def set_boundary(
    phi: wp.array3d(dtype=float),
    fixed: wp.array3d(dtype=int),
    values: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    if fixed[i, j, k] != 0:
        phi[i, j, k] = values[i, j, k]


@dataclass
class Settings:
    dx: float = 0.12
    eta: float = 2.2
    critical: float = 0.06
    internal: float = 0.018
    breakdown: float = 0.12
    iterations: int = 48
    max_nodes: int = 1800
    seed: int = 71
    balance_tolerance: float = 0.03


def laplacian(phi, dx):
    p = np.pad(phi, 1, mode="edge")
    return (
        p[2:, 1:-1, 1:-1]
        + p[:-2, 1:-1, 1:-1]
        + p[1:-1, 2:, 1:-1]
        + p[1:-1, :-2, 1:-1]
        + p[1:-1, 1:-1, 2:]
        + p[1:-1, 1:-1, :-2]
        - 6 * phi
    ) / (dx * dx)


def growth_weights(parent_phi, neighbor_phi, distance, polarity, critical, eta):
    field = polarity * (parent_phi - neighbor_phi) / distance
    return np.maximum(field - critical, 0.0) ** eta


class Poisson:
    def __init__(self, rho, fixed, values, dx, device="cuda:0", iterations=48):
        self.shape, self.dx, self.device = rho.shape, dx, device
        self.rho = wp.array(rho.astype("f"), dtype=float, device=device)
        self.fixed = wp.array(fixed.astype("i"), dtype=int, device=device)
        self.values = wp.array(values.astype("f"), dtype=float, device=device)
        self.phi = wp.array(
            np.where(fixed, values, 0).astype("f"), dtype=float, device=device
        )
        self.iterations = iterations
        self.sweep()
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            for _ in range(iterations):
                self.sweep()
        self.graph = capture.graph

    def sweep(self):
        for parity in (0, 1):
            wp.launch(
                relax,
                self.shape,
                [self.phi, self.rho, self.fixed, self.values, self.dx, parity, 1.86],
                device=self.device,
            )

    def solve(self, blocks=1):
        for _ in range(blocks):
            wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
        return self.phi.numpy()

    def boundaries(self, fixed, values):
        self.fixed.assign(np.asarray(fixed, dtype="i"))
        self.values.assign(np.asarray(values, dtype="f"))
        wp.launch(
            set_boundary,
            self.shape,
            [self.phi, self.fixed, self.values],
            device=self.device,
        )


def discharge(rho, solid, origin, settings, seed_mask=None, device="cuda:0"):
    """Grow both channel polarities and export the complete parent tree."""
    s = settings
    rng = np.random.default_rng(s.seed)
    start = time.monotonic()
    shape = rho.shape
    solid = np.asarray(solid, dtype=bool).copy()
    solid[:, :, 0] = True
    fixed = solid.astype("i")
    values = np.zeros(shape, dtype="f")
    pde = Poisson(rho, fixed, values, s.dx, device, s.iterations)
    phi = pde.solve(80)
    initial = phi.copy()
    grad = np.stack(np.gradient(phi, s.dx), -1)
    magnitude = np.linalg.norm(grad, axis=-1)
    eligible = ~solid if seed_mask is None else (~solid & seed_mask)
    eligible[:2] = False
    eligible[-2:] = False
    eligible[:, :2] = False
    eligible[:, -2:] = False
    eligible[:, :, -2:] = False
    if not eligible.any():
        raise ValueError("No air cells in initiation volume")
    first = np.array(
        np.unravel_index(np.where(eligible, magnitude, -1).argmax(), shape)
    )
    if magnitude[tuple(first)] < s.breakdown:
        raise ValueError("Atmospheric electric field is below breakdown threshold")
    offsets = np.array(list(itertools.product([-1, 0, 1], repeat=3)), dtype=int)
    offsets = offsets[np.any(offsets, axis=1)]
    direction = -grad[tuple(first)]
    direction /= np.linalg.norm(direction)
    off = offsets[np.argmax(offsets @ direction / np.linalg.norm(offsets, axis=1))]
    second = first + off
    seed_potential = float((phi[tuple(first)] + phi[tuple(second)]) * 0.5)
    nodes = [first, second]
    parents = [-1, 0]
    signs = [-1, 1]
    born = [0, 0]
    occupied = np.zeros(shape, dtype="i") - 1
    fronts = {-1: [], 1: []}
    grounded = {-1: False, 1: False}
    contacts = []
    charge_log = []
    stop = "node_budget"

    def append_front(node_id):
        cell = nodes[node_id]
        for offset in offsets:
            n = cell + offset
            if (
                (n >= 1).all()
                and (n < np.array(shape) - 1).all()
                and occupied[tuple(n)] < 0
            ):
                fronts[signs[node_id]].append(
                    (node_id, tuple(n), float(np.linalg.norm(offset) * s.dx))
                )

    for idx in range(2):
        cell = tuple(nodes[idx])
        occupied[cell] = idx
        fixed[cell] = 2
        values[cell] = seed_potential
    append_front(0)
    append_front(1)
    pde.boundaries(fixed, values)
    phi = pde.solve(6)
    response_values = (fixed == 2).astype("f")
    response = Poisson(
        np.zeros_like(rho), fixed, response_values, s.dx, device, s.iterations
    )
    unit = response.solve(30)
    floating_corrections = []
    for step in range(s.max_nodes - 2):
        # -L(phi)-rho measures induced charge relative to the ambient source.
        induced = -laplacian(phi, s.dx) - rho
        charges = {
            sg: float(
                sum(induced[tuple(n)] for n, sign in zip(nodes, signs) if sign == sg)
            )
            * s.dx**3
            for sg in (-1, 1)
        }
        imbalance = abs(charges[-1] + charges[1]) / max(
            abs(charges[-1]) + abs(charges[1]), 1e-12
        )
        order = [-1, 1] if abs(charges[-1]) <= abs(charges[1]) else [1, -1]
        if imbalance < s.balance_tolerance:
            order = [-1, 1] if rng.random() < 0.5 else [1, -1]
        if grounded[order[0]]:
            order.reverse()
        selection = None
        for sg in order:
            if grounded[sg]:
                continue
            candidates = [v for v in fronts[sg] if occupied[v[1]] < 0]
            fronts[sg] = candidates
            if not candidates:
                continue
            par = np.array([v[0] for v in candidates])
            idx = np.array([v[1] for v in candidates])
            pp = np.array([values[tuple(nodes[x])] for x in par])
            dist = np.array([v[2] for v in candidates])
            weights = growth_weights(pp, phi[tuple(idx.T)], dist, sg, s.critical, s.eta)
            if weights.sum() > 0:
                pick = int(rng.choice(len(weights), p=weights / weights.sum()))
                selection = (sg, *candidates[pick])
                break
        if selection is None:
            stop = "below_critical_field"
            break
        sg, parent, cell, distance = selection
        cell = tuple(cell)
        node_id = len(nodes)
        nodes.append(np.array(cell))
        parents.append(parent)
        signs.append(sg)
        born.append(step + 1)
        occupied[cell] = node_id
        if solid[cell]:
            grounded[sg] = True
            contacts.append(node_id)
            root_voltage = float(values[tuple(nodes[0])])
            for n, sign in zip(nodes[:-1], signs[:-1]):
                if sign != sg:
                    values[tuple(n)] -= root_voltage
            # A connected conductive channel becomes part of the grounded set.
            for n, sign in zip(nodes, signs):
                if sign == sg:
                    values[tuple(n)] = 0.0
                    fixed[tuple(n)] = 3
        else:
            fixed[cell] = 2
            values[cell] = values[tuple(nodes[parent])] - sg * distance * s.internal
            append_front(node_id)
        pde.boundaries(fixed, values)
        phi = pde.solve()
        if not contacts:
            # Floating-conductor correction: solve its unit-capacitance field,
            # then shift the entire bipolar tree to preserve zero net charge.
            # This preserves every Eq. (12) voltage drop while allowing either
            # tip to grow in a strongly nonuniform scene-scale field.
            response_values = (fixed == 2).astype("f")
            response.boundaries(fixed, response_values)
            unit = response.solve()
            channel = fixed == 2
            qnet = float((-laplacian(phi, s.dx) - rho)[channel].sum()) * s.dx**3
            capacitance = float((-laplacian(unit, s.dx))[channel].sum()) * s.dx**3
            delta = -qnet / max(capacitance, 1e-10)
            values[channel] += delta
            phi += delta * unit
            pde.phi.assign(phi.astype("f"))
            pde.boundaries(fixed, values)
            if step % 100 == 0:
                floating_corrections.append(
                    dict(step=step, offset=delta, capacitance=capacitance)
                )
        if step % 100 == 0:
            relative = abs(charges[-1] + charges[1]) / max(
                abs(charges[-1]) + abs(charges[1]), 1e-12
            )
            record = dict(
                step=step,
                nodes=len(nodes),
                charge=charges,
                relative_imbalance=relative,
                grounded=dict(grounded),
            )
            charge_log.append(record)
            print(record, flush=True)
        if all(grounded.values()):
            stop = "grounded"
            break
        # Keep the opposite cloud channel growing after ground attachment.
        if contacts and node_id - contacts[0] >= 260:
            stop = "post_attachment_budget"
            break
    phi = pde.solve(40)
    induced = -laplacian(phi, s.dx) - rho
    residual = laplacian(phi, s.dx) + rho
    free = fixed == 0
    cells = np.array(nodes)
    positions = np.asarray(origin) + (cells + 0.5) * s.dx
    current = np.abs(induced[tuple(cells.T)]) * s.dx**3
    propagated = current.copy()
    for i in range(len(nodes) - 1, 0, -1):
        if parents[i] >= 0:
            propagated[parents[i]] += propagated[i]
    path = np.zeros(len(nodes), dtype=bool)
    for index in contacts:
        while index >= 0:
            path[index] = True
            index = parents[index]
    elapsed = time.monotonic() - start
    report = dict(
        settings=asdict(s),
        nodes=len(nodes),
        contacts=contacts,
        stop=stop,
        seconds=elapsed,
        initial_max_field=float(magnitude[tuple(first)]),
        seed=positions[0].tolist(),
        poisson_relative_rms=float(
            np.sqrt(np.mean(residual[free] ** 2))
            / max(np.sqrt(np.mean(rho[free] ** 2)), 1e-10)
        ),
        charge_log=charge_log,
        floating_corrections=floating_corrections,
    )
    return dict(
        positions=positions.astype("f"),
        parents=np.array(parents, dtype="i"),
        polarity=np.array(signs, dtype="i"),
        born=np.array(born, dtype="i"),
        charge=current.astype("f"),
        flow=propagated.astype("f"),
        main_path=path,
        contact=np.array(contacts, dtype="i"),
        phi=phi,
        initial_phi=initial,
        rho=rho,
        solid=solid,
        origin=np.asarray(origin),
        dx=np.array(s.dx),
        report=report,
    )
