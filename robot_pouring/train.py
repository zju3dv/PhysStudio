"""Minimal PPO trainer exercising the same public tensor reset/step interface."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from .config import PouringConfig
from .env import PouringEnv


class ActorCritic(nn.Module):
    def __init__(self, obs_size=51):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 7))
        self.critic = nn.Sequential(nn.Linear(obs_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1))
        self.log_std = nn.Parameter(torch.full((7,), -1.0))
        nn.init.orthogonal_(self.actor[-1].weight, .01)
        nn.init.zeros_(self.actor[-1].bias)

    def evaluate(self, obs, latent=None):
        distribution = torch.distributions.Normal(self.actor(obs), self.log_std.exp())
        z = distribution.sample() if latent is None else latent
        action = torch.tanh(z)
        log_prob = (distribution.log_prob(z) - torch.log(1. - action.square() + 1e-6)).sum(-1)
        return action, z, log_prob, distribution.entropy().sum(-1), self.critic(obs).squeeze(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["basin", "flower", "precision"], default="basin")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--horizon", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--spacing", type=float, default=.011)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(4)
    cfg = PouringConfig(task=args.task, num_envs=args.num_envs, spacing=args.spacing, seed=args.seed)
    env = PouringEnv(cfg)
    policy = ActorCritic().to(env.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    before = torch.cat([p.detach().flatten() for p in policy.parameters()]).clone()
    obs, _ = env.reset(seed=args.seed)
    metrics = []; started = time.perf_counter(); successes = episodes = 0
    for update in range(args.updates):
        observations, latents, logs, values, rewards, next_values, dones = [], [], [], [], [], [], []
        for _ in range(args.horizon):
            with torch.no_grad():
                a, z, logp, _, value = policy.evaluate(obs)
            observations.append(obs.clone()); latents.append(z); logs.append(logp); values.append(value)
            next_obs, reward, term, trunc, info = env.step(a)
            with torch.no_grad():
                bootstrap = policy.critic(next_obs).squeeze(-1) * (~term)
            rewards.append(reward); next_values.append(bootstrap); dones.append(term | trunc)
            obs = next_obs
            if (term | trunc).any():
                successes += int(info["success"].sum()); episodes += int((term | trunc).sum())
                obs, _ = env.reset(env_ids=torch.where(term | trunc)[0])
        v, rew, nv, done = map(torch.stack, [values, rewards, next_values, dones])
        advantage = torch.zeros_like(rew); running = torch.zeros(args.num_envs, device=env.device)
        for step in reversed(range(args.horizon)):
            delta = rew[step] + .99 * nv[step] - v[step]
            running = delta + .99 * .95 * (~done[step]) * running
            advantage[step] = running
        returns = (advantage + v).flatten()
        adv = advantage.flatten(); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch_obs, batch_z, old_logp = torch.cat(observations), torch.cat(latents), torch.cat(logs)
        losses = []
        for _ in range(args.epochs):
            order = torch.randperm(len(adv), device=env.device)
            for indices in order.split(min(256, len(adv))):
                _, _, logp, entropy, value = policy.evaluate(batch_obs[indices], batch_z[indices])
                ratio = (logp - old_logp[indices]).exp()
                actor_loss = -torch.minimum(ratio * adv[indices], ratio.clamp(.8, 1.2) * adv[indices]).mean()
                critic_loss = .5 * (value - returns[indices]).square().mean()
                loss = actor_loss + critic_loss - .001 * entropy.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite PPO loss")
                optimizer.zero_grad(); loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), .5)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError("Non-finite PPO gradient")
                optimizer.step(); losses.append(float(loss))
        row = dict(update=update+1, transitions=(update+1)*args.horizon*args.num_envs,
                   mean_reward=float(rew.mean()), loss=float(np.mean(losses)),
                   episodes=episodes, successes=successes, elapsed=time.perf_counter()-started)
        print(json.dumps(row), flush=True); metrics.append(row)
    torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "config": cfg.to_dict(), "arguments": vars(args) | {"output": str(args.output)},
                "observation_size": 51, "action_size": 7}, args.output / "checkpoint.pt")
    restored = ActorCritic().to(env.device)
    restored.load_state_dict(torch.load(args.output / "checkpoint.pt", weights_only=False, map_location=env.device)["policy"])
    with torch.no_grad():
        assert torch.equal(policy.actor(obs), restored.actor(obs))
        after = torch.cat([p.detach().flatten() for p in policy.parameters()])
        delta = float(torch.linalg.vector_norm(after - before))
        assert delta > 0
    report = dict(passed=True, purpose="PPO integration run; this does not establish a solved task",
                  parameter_change_l2=delta, checkpoint_reload_exact=True, records=metrics)
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
