"""Physical, isolation, reset, action and Gymnasium contract checks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
import torch
import warp as wp
import mujoco
import mujoco_warp as mjw
from .config import PouringConfig
from .env import PouringEnv, GymPouringEnv
from .fluid import density, project


def check(output):
    results = {}
    env = PouringEnv(PouringConfig(num_envs=2, spacing=.011, randomize=False))
    zeros = torch.zeros((2,7), device=env.device)
    # Compare one robot dynamics substep against the independent CPU backend.
    cpu = mujoco.MjData(env.model); cpu.qpos[:] = env.initial_q
    offset=np.array([.10,-.05,.05,-.04,.02,.10,-.05])
    cpu.ctrl[:7] = env.initial_q[:7]+offset; cpu.ctrl[7] = .025
    env.ctrl[:,:7] += torch.as_tensor(offset,dtype=torch.float32,device=env.device)
    cpu.mocap_pos[0] = env.target[0].cpu().numpy()
    mujoco.mj_step(env.model, cpu)
    env._substep(); wp.synchronize_device(env.cfg.device)
    qerr = float(np.max(np.abs(env.q[0].cpu().numpy()-cpu.qpos)))
    verr = float(np.max(np.abs(env.qvel[0].cpu().numpy()-cpu.qvel)))
    assert np.linalg.norm(cpu.qvel[:7])>.02
    assert qerr < 2e-5 and verr < .005, (qerr, verr)
    results["cpu_mujoco_parity"] = dict(qpos_max_abs=qerr, qvel_max_abs=verr,
        control_offset_rad=offset.tolist(),cpu_joint_velocity_norm=float(np.linalg.norm(cpu.qvel[:7])))
    env.reset(seed=42)
    force = []
    for i in range(60):
        o, r, term, trunc, info = env.step(zeros)
        assert torch.isfinite(o).all() and torch.isfinite(r).all()
        assert not term.any() and not trunc.any()
        assert torch.allclose(info["mass_fractions"].sum(1), torch.ones(2, device=env.device), atol=1e-6)
        assert info["spill_fraction"].max() == 0
        if i >= 25: force.append(env.g.force.numpy().copy())
    s = env.snapshot(); points = s["x"][s["status"] == 0]
    nn = cKDTree(points).query(points,k=2)[0][:,1]
    assert nn.min() > .35 * env.cfg.spacing, nn.min()
    expected = env.count.cpu().numpy() * env.g.mass * 9.81
    measured = -np.mean(np.asarray(force),axis=0)[:,2]
    error = np.abs(measured/expected-1)
    assert error.max() < .20, (expected, measured)
    results["static_water"] = dict(seconds=60*env.cfg.control_dt, spill_fraction=0,
        min_neighbor_distance=float(nn.min()), expected_weight_N=expected.tolist(),
        mean_downward_cup_load_N=measured.tolist(), relative_load_error=error.tolist())
    # Reset one world while preserving every observed state in the other world.
    before = env.snapshot(1)
    env.reset(env_ids=[0])
    after = env.snapshot(1)
    for key in ("qpos","qvel","ctrl","x","v","status","target","cup_pos","cup_rot","mass_fractions"):
        assert np.array_equal(before[key], after[key]), key
    results["selective_reset"] = "other world state bit-identical"
    env.reset(seed=42)
    baseline = env.q.clone()
    action = zeros.clone(); action[0,6] = .5
    for _ in range(10): env.step(action)
    delta = env.q-baseline
    assert abs(float(delta[0,6])) > .03
    assert abs(float(delta[1,6])) < .01
    results["joint_action_response"] = dict(commanded_delta=float(delta[0,6]), other_world_delta=float(delta[1,6]))
    # Matching RNG states produce identical reset states, including water fill.
    env.cfg.randomize = True
    env.reset(seed=900); first = env.snapshot()
    env.step(zeros); env.reset(seed=900); second = env.snapshot()
    for key in ("qpos","x","status","target"): assert np.array_equal(first[key],second[key]), key
    results["seeded_reset"] = "exact reset state match"
    # Verify masking independently of spatial separation: overlap all particles
    # from world 1 with world 0, then remove world 1 from the neighbor solve.
    env.cfg.randomize = False; env.reset(seed=42)
    n = env.g.n
    pos = wp.to_torch(env.g.x); status = wp.to_torch(env.g.status)
    pos[n:2*n] = pos[:n]
    env.water.grid.build(env.g.x,env.g.h)
    wp.launch(density,env.water.count,[env.g],device=env.cfg.device)
    overlap = env.g.rho.numpy()[:n].copy()
    status[n:2*n] = 3
    wp.launch(density,env.water.count,[env.g],device=env.cfg.device)
    single = env.g.rho.numpy()[:n]
    diff = float(np.max(np.abs(overlap-single)))
    assert diff < 2e-6
    results["neighbor_world_isolation"] = dict(overlapping_world_density_error=diff)
    # Independently check the opposite boundary impulse and its moment arm.
    status[:] = 3; status[0] = 0
    center = np.array([.3,0.,.3],np.float32)
    wp.to_torch(env.g.cup_pos)[0] = torch.as_tensor(center,device=env.device)
    wp.to_torch(env.g.cup_rot)[0] = torch.eye(3,device=env.device)
    point = center + np.array([env.cfg.cup_radius-.001,0.,.06],np.float32)
    pos[0] = torch.as_tensor(point,device=env.device)
    env.g.force.zero_(); env.g.torque.zero_()
    wp.launch(project,env.water.count,[env.g,0],device=env.cfg.device)
    end = env.g.x.numpy()[0]
    force_expected = -env.g.mass*(end-point)/env.cfg.dt**2
    force_actual = env.g.force.numpy()[0]
    torque_expected = np.cross(end-center,force_expected)
    assert np.allclose(force_actual,force_expected,rtol=1e-5,atol=1e-5)
    assert np.allclose(env.g.torque.numpy()[0],torque_expected,rtol=1e-5,atol=1e-5)
    results["opposite_boundary_impulse"] = dict(force_N=force_actual.tolist(), moment_Nm=torque_expected.tolist())
    env.reset(seed=42)
    # Classification is an exhaustive, idempotent particle ledger.
    env.water.measure(); counts=env.metrics[:,:5].clone()
    for _ in range(3): env.water.measure()
    assert torch.equal(counts,env.metrics[:,:5])
    assert torch.equal(env.metrics[:,:5].sum(1),env.count.float())
    results["mass_accounting"] = "exhaustive and idempotent"
    # Test soil capacity and one-time absorption independently of a rollout.
    env.g.absorb = 1
    env.g.absorption_attempts.zero_()
    status[:] = 0
    for e in range(env.num_envs):
        point = env.target[e] + torch.tensor([-.04,0.,.049],device=env.device)
        point = point + torch.tensor([3.*(e%16),3.*(e//16),0.],device=env.device)
        pos[e*n:(e+1)*n] = point
    wp.launch(project,env.water.count,[env.g,0],device=env.cfg.device)
    env.water.measure()
    absorbed = env.metrics[:,4].clone()
    assert torch.equal(absorbed,(.75*env.count).int().float())
    wp.launch(project,env.water.count,[env.g,0],device=env.cfg.device)
    env.water.measure()
    assert torch.equal(absorbed,env.metrics[:,4])
    assert torch.equal(env.metrics[:,:5].sum(1),env.count.float())
    results["soil_capacity"] = dict(absorbed_particles=absorbed.cpu().tolist(), capacity_fraction=.75,
                                     repeated_projection_adds_no_water=True)
    env.g.absorb = 0; env.reset(seed=42)
    # A fixed milliliter dose must follow actual randomized fill volume.
    env.cfg.goal_ml = 200.;env.cfg.randomize=True
    env.reset(seed=11)
    assert torch.allclose(env.info()["goal_ml"],torch.full((2,),200.,device=env.device),atol=1e-4)
    results["fixed_volume_goal"] = dict(requested_ml=200.,actual_goal_ml=env.info()["goal_ml"].cpu().tolist())
    env.cfg.goal_ml=None;env.cfg.randomize=False;env.goal[:]=env.cfg.goal_fraction;env.reset(seed=42)
    for invalid in [torch.zeros((2,6)), torch.full((2,7),float('nan'))]:
        try: env.step(invalid)
        except ValueError: pass
        else: raise AssertionError("Invalid action accepted")
    results["action_validation"] = "shape and finiteness checked"
    policy_action = zeros.clone().requires_grad_(True)
    obs,rew,_,_,_=env.step(policy_action)
    assert not obs.requires_grad and not rew.requires_grad and not env.ctrl.requires_grad
    results["policy_autograd_boundary"] = "policy action graph detached at environment boundary"
    env.reset(seed=42)
    env.steps[:] = env.cfg.max_steps-1
    o,r,t,tr,info=env.step(zeros)
    assert tr.all() and not t.any()
    try: env.step(zeros)
    except RuntimeError: pass
    else: raise AssertionError("Finished worlds stepped without reset")
    results["episode_semantics"] = "time limit truncates; explicit reset required"
    env.close()
    # Native integration check supplied by Gymnasium, not a local API mock.
    from gymnasium.utils.env_checker import check_env
    gym_env=GymPouringEnv(spacing=.011,randomize=False)
    check_env(gym_env,skip_render_check=True)
    gym_env.close();results["gymnasium_checker"] = "passed"
    report=dict(passed=True,checks=results,versions=dict(warp=wp.__version__,mujoco=mujoco.__version__,mujoco_warp=mjw.__version__),
                sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


def benchmark(output, num_envs=16, steps=80):
    env=PouringEnv(PouringConfig(num_envs=num_envs,spacing=.011,randomize=False))
    actions=torch.zeros((num_envs,7),device=env.device)
    for _ in range(10):env.step(actions)
    wp.synchronize_device(env.cfg.device);start=time.perf_counter()
    for _ in range(steps):env.step(actions)
    wp.synchronize_device(env.cfg.device);elapsed=time.perf_counter()-start
    report=dict(num_envs=num_envs,particles_per_env=env.g.n,steps=steps,seconds=elapsed,
        aggregate_transitions_per_second=num_envs*steps/elapsed,
        wall_seconds_per_vector_step=elapsed/steps,
        total_particle_substeps=num_envs*env.g.n*steps*env.cfg.substeps,
        config=env.cfg.to_dict(),rendering=False,includes_observation_reward_and_action_checks=True)
    output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));env.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--benchmark',type=int)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError(a.output)
    benchmark(a.output,a.benchmark) if a.benchmark else check(a.output)
