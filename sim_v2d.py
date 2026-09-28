# -*- coding: utf-8 -*-
"""
sim_v2d —— Part D：单节点逃逸 MFPT 与 Kramers 预言的对照（快速版）

sim_v2 中 Part D 取 D 从 10^-2 起、T_max=4000、dt=0.002、n_traj=3000，
而 Kramers 时间 T = (2π/√2) e^{ΔU/D} 在 D=0.01 时为 e^25 ≈ 7×10^10，
轨迹永远跑不到，导致每点耗时 700 s 以上（全程需 6 小时）。

本脚本把 D 限制在 MFPT 可测的区间 [0.04, 0.6]（对应 T 从 ~5 到 ~2300），
用更大的 dt（单节点一维问题，dt=0.005 稳定）与并行计算。
"""
import os, sys, json, math, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_v2 as S

DT_MF = 0.005
T_MAX = 3000.0
N_TRAJ = 3000
D_MIN, D_MAX, N_D = 0.04, 0.6, 14
X0 = -S.X_M          # 从势阱底出发
BARRIER = 0.0        # 首次到达势垒顶


def mfpt(alpha, D, seed):
    rng = np.random.default_rng(seed)
    n = N_TRAJ
    x = np.full(n, X0, dtype=np.float64)
    t = np.zeros(n)
    done = np.zeros(n, dtype=bool)
    first = np.full(n, np.nan)
    n_steps = int(T_MAX / DT_MF)
    for _ in range(n_steps):
        alive = ~done
        if not alive.any():
            break
        xa = x[alive]
        F = S.C.a * xa - S.C.b * xa ** 3
        x[alive] = xa + F * DT_MF + S.levy_inc(alpha, D, DT_MF, int(alive.sum()), rng)
        t[alive] += DT_MF
        idx = np.where(alive)[0][x[alive] >= BARRIER]
        first[idx] = t[idx]
        done[idx] = True
    if np.isnan(first).all():
        return float(T_MAX)
    return float(np.nanmean(first))


def job(args):
    a, di, D = args
    return a, di, mfpt(a, D, seed=S.C.random_seed + int(D * 1e6) + int(a * 100))


def main():
    t0 = time.time()
    D_mf = np.logspace(math.log10(D_MIN), math.log10(D_MAX), N_D).tolist()
    alphas = S.C.alpha_values
    resD = {str(a): dict(mfpt=[0.0] * N_D, rate=[0.0] * N_D) for a in alphas}
    jobs = [(a, di, D) for a in alphas for di, D in enumerate(D_mf)]
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        for a, di, Tm in ex.map(job, jobs):
            resD[str(a)]['mfpt'][di] = Tm
            resD[str(a)]['rate'][di] = 1.0 / max(Tm, 1e-12)
            done += 1
            print(f"\r  {done}/{len(jobs)}", end='', flush=True)
    print()
    resD['D_mf'] = D_mf
    resD['kramers'] = [S.kramers_rate(D) for D in D_mf]
    resD['kramers_mfpt'] = [1.0 / S.kramers_rate(D) for D in D_mf]
    with open(os.path.join(S.OUT, 'partD_mfpt.json'), 'w') as f:
        json.dump(resD, f, indent=1)

    Dm = np.array(D_mf)
    for a in alphas:
        k = str(a)
        T = np.array(resD[k]['mfpt'])
        sl = np.polyfit(1.0 / Dm, np.log(T), 1)[0]
        print('  alpha=%s  拟合斜率 dlnT/d(1/D) = %.4f  (Kramers 理论 %.4f)'
              % (a, sl, S.DELTA_U))
    print('  完成，用时 %.1f s' % (time.time() - t0))


if __name__ == '__main__':
    main()
