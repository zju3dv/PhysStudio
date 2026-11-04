"""Evaluate a saved PPO checkpoint on independently seeded episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .config import PouringConfig
from .env import PouringEnv
from .train import ActorCritic


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=10)
    p.add_argument('--seed',type=int,default=10000)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.episodes<1:raise ValueError('episodes must be positive')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=PouringConfig(**(checkpoint['config']|{'num_envs':1,'seed':args.seed}))
    env=PouringEnv(cfg);policy=ActorCritic(checkpoint['observation_size']).to(env.device)
    policy.load_state_dict(checkpoint['policy']);policy.eval();records=[]
    for episode in range(args.episodes):
        obs,info=env.reset(seed=args.seed+episode);total=0.
        for step in range(cfg.max_steps):
            with torch.no_grad():action=torch.tanh(policy.actor(obs))
            obs,reward,terminated,truncated,info=env.step(action);total+=float(reward[0])
            if bool(terminated[0]|truncated[0]):break
        record=dict(episode=episode,seed=args.seed+episode,steps=step+1,total_reward=total,
                    success=bool(info['success'][0]),failure=bool(info['failure'][0]),
                    truncated=bool(truncated[0]),captured_fraction=float(info['captured_fraction'][0]),
                    spilled_fraction=float(info['spill_fraction'][0]))
        records.append(record);print(json.dumps(record),flush=True)
    result=dict(checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        policy='deterministic mean of saved PPO actor',episodes=args.episodes,
        success_rate=sum(r['success'] for r in records)/len(records),records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    env.close()


if __name__=='__main__':main()
