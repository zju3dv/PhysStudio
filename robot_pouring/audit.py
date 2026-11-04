"""Audit complete recorded episodes independently of online reward logic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import PouringConfig
from .model import make_model


def sdf(p, radius, height, wall):
    r=np.linalg.norm(p[:,:2],axis=1)
    a=np.stack([r-radius-wall,np.abs(p[:,2]+wall/2)-wall/2],axis=1)
    b=np.stack([np.abs(r-radius-wall/2)-wall/2,np.abs(p[:,2]-height/2)-height/2],axis=1)
    def box(q):return np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(1),0)
    return np.minimum(box(a),box(b))


def audit(root, allow_incomplete=False, source_dir=None):
    cfg=PouringConfig(**json.loads((root/'config.json').read_text()))
    report=json.loads((root/'report.json').read_text())
    sources=json.loads((root/'sources.json').read_text())
    for name in ['config.py','env.py','fluid.py','model.py','demo.py']:
        assert hashlib.sha256(((source_dir or Path(__file__).parent)/name).read_bytes()).hexdigest()==sources[name],name
    files=sorted((root/'states').glob('*.npz'))
    assert len(files)==report['frames']
    assert [int(f.stem) for f in files]==list(range(len(files)))
    model,_=make_model(cfg)
    minimum_gap=float('inf');maximum_speed=0.;mass_error=0.;joint_margin=float('inf')
    prev_spill=0.;prev_absorb=0.
    n=None
    for frame,f in enumerate(files):
        with np.load(f) as d:
            for key in ['x','v','qpos','qvel','cup_pos','cup_rot','mass_fractions','target','cup_force']:
                assert np.isfinite(d[key]).all(),(frame,key)
            active=d['status']<2
            assert np.all(np.isin(d['status'],[0,1,2,3]))
            count=np.count_nonzero(d['status']!=3)
            if n is None:n=count
            assert n==count
            error=float(abs(d['mass_fractions'].sum()-1));mass_error=max(mass_error,error)
            assert error<2e-6
            assert abs(d['mass_fractions'][3]-np.count_nonzero(d['status']==1)/n)<1e-6
            assert abs(d['mass_fractions'][4]-np.count_nonzero(d['status']==2)/n)<1e-6
            assert d['mass_fractions'][3]>=prev_spill-1e-7 and d['mass_fractions'][4]>=prev_absorb-1e-7
            prev_spill=float(d['mass_fractions'][3]);prev_absorb=float(d['mass_fractions'][4])
            if np.any(active):
                x=d['x'][active];v=d['v'][active]
                local=(x-d['cup_pos'])@d['cup_rot']
                cupgap=sdf(local,cfg.cup_radius,cfg.cup_height,cfg.cup_wall)
                targetgap=sdf(x-d['target'],cfg.target_radius,cfg.target_height,.008)
                gap=min(cupgap.min(),targetgap.min(),x[:,2].min())
                minimum_gap=min(minimum_gap,float(gap))
                maximum_speed=max(maximum_speed,float(np.linalg.norm(v,axis=1).max()))
            q=d['qpos'][:7]
            joint_margin=min(joint_margin,float(np.minimum(q-model.jnt_range[:7,0],model.jnt_range[:7,1]-q).min()))
    assert minimum_gap>=.46*cfg.spacing-2e-4,minimum_gap
    assert maximum_speed<15 and joint_margin>-.01
    assert not report['final']['failure']
    if not allow_incomplete: assert report['final']['success']
    final=report['final']['mass'];water_ml=n*cfg.density*cfg.spacing**3*1000
    result=dict(passed=True,frames=len(files),particles=n,water_ml=water_ml,
        final_captured_fraction=final[1]+final[4],final_spilled_fraction=final[3],
        final_captured_ml=(final[1]+final[4])*water_ml,
        requested_ml=cfg.goal_ml or cfg.goal_fraction*water_ml,
        maximum_mass_partition_error=mass_error,minimum_solid_gap_m=minimum_gap,
        particle_radius_m=.46*cfg.spacing,maximum_particle_speed_m_s=maximum_speed,
        minimum_joint_limit_margin_rad=joint_margin,physics_sources_match=True,
        success=bool(report['final']['success']),controller='reference controller, not trained RL')
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('inputs',type=Path,nargs='+');p.add_argument('--allow-incomplete',action='store_true',help='Audit a short prefix without requiring task success');p.add_argument('--source-dir',type=Path,help='Verify recording hashes against a retained source snapshot');args=p.parse_args()
    for root in args.inputs:audit(root,args.allow_incomplete,args.source_dir)
