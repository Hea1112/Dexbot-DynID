#!/usr/bin/env python3
"""Optional paper Eq. (29) acceleration computation and signal diagnostics.

Exports arrays without silently modifying original CSV or precomputed Y.
To use filtered q/v/a for dynamics, RECOMPUTE Y on those states.
"""
import argparse
from pathlib import Path
import numpy as np
from dexbot_id.data import load_id_csv
from dexbot_id.preprocessing import fourth_order_acceleration, zero_phase_filter


def main():
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True)
    p.add_argument('--out',default='runs/preprocess.npz');p.add_argument('--cutoff-hz',type=float,default=None)
    args=p.parse_args();d=load_id_csv(args.csv);dt=float(np.median(np.diff(d.time)))
    acc=fourth_order_acceleration(d.v,dt)
    vfilter=zero_phase_filter(d.v,1/dt,args.cutoff_hz) if args.cutoff_hz else d.v
    afilter=fourth_order_acceleration(vfilter,dt)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,time=d.time,q=d.q,v=d.v,vd_csv=d.a,
                        acceleration_fd4=acc,velocity_zero_phase=vfilter,
                        filtered_velocity_fd4_acceleration=afilter,valid_fd4=np.isfinite(acc).all(axis=1))
    print('Derivative diagnostic written:',out,'Do not reuse Y calculated from different unfiltered states.')
if __name__=='__main__':main()
