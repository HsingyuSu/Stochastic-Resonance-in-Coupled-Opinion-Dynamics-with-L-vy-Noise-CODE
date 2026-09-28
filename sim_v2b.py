# -*- coding: utf-8 -*-
"""
sim_v2b —— Part A / Part C 的修正重跑（多进程并行）

修正内容（相对 sim_v2）：
  1. sim_v2 中 T_transient 定义后从未使用，谱分析包含了初始生长瞬态，
     导致小 D 处 eta 与 SNR 被虚假抬高（瞬态在低频 bin 的能量）。
  2. T_total=100 只含 1.6 个驱动周期（P=2π/0.1≈62.8），频率分辨率不足，
     驱动频率无法与 DC/瞬态分离。

本脚本：
  * T_total = T_transient + N_PERIOD * P，其中 N_PERIOD=4，
    使稳态窗口恰好含整数个驱动周期 → 信号精确落在 FFT 第 N_PERIOD 个 bin；
  * 丢弃 t < T_transient 的瞬态；
  * eta / SNR / f_mis 相干幅值均在稳态整数周期窗口上用 lock-in（或 FFT）计算。
"""
import os, sys, json, math, time, pickle, io
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_v2 as S

# ---------------------------------------------------------------- 参数
C = S.C
N, Lg, DT, REC = S.N, S.Lg, S.DT, S.REC
P = 2.0 * math.pi / C.omega          # 驱动周期 ≈ 62.83
N_PERIOD = 4
T_TRANSIENT = 60.0
T_TOTAL = T_TRANSIENT + N_PERIOD * P
N_WIN = int(round(N_PERIOD * P / (DT * REC)))   # 稳态窗口样本数
N_TR = int(round(T_TRANSIENT / (DT * REC)))
DT_REC = DT * REC

print(f"[config] P={P:.4f}  T_total={T_TOTAL:.3f}  transient={T_TRANSIENT}  "
      f"window={N_PERIOD} periods ({N_WIN} samples)")

_G = {}


def _init():
    J, positions = S.build_coupling()
    _G['J'] = J
    _G['Jrow'] = np.asarray(J.sum(axis=1)).ravel()
    _G['pos'] = positions


# ---------------------------------------------------------------- 分析
def analyse(m, fmis, A):
    """稳态整数周期窗口上的 SNR / eta / f_mis 相干幅值。"""
    mw = m[-N_WIN:]
    fw = fmis[-N_WIN:]
    t = np.arange(N_WIN) * DT_REC
    z = np.exp(-1j * C.omega * t)
    M1 = 2.0 * np.mean(mw * z)              # 平均意见的一阶谐波
    eta = 2.0 * abs(M1) / A
    F1 = 2.0 * np.mean(fw * z)              # 信念摆动的一阶谐波
    famp = 2.0 * abs(F1)
    # FFT：整数周期 -> 信号精确落 bin N_PERIOD
    psd = np.abs(np.fft.rfft(mw - mw.mean())) ** 2 / N_WIN
    k = N_PERIOD
    sig = psd[k]
    mask = np.ones(len(psd), dtype=bool)
    mask[max(0, k - 2):k + 3] = False
    mask[0] = False
    noise = np.mean(psd[mask]) + 1e-30
    snr = float(10.0 * np.log10(sig / noise))
    return snr, float(eta), float(famp)


# ---------------------------------------------------------------- 仿真
def run_lattice2(D, alpha, seed, C_bias=0.0, snapshot_times=None):
    J, Jrow, positions = _G['J'], _G['Jrow'], _G['pos']
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01
    c = Lg // 2; half = max(2, Lg // 12)
    for i in range(c - half, c + half):
        for j in range(c - half, c + half):
            if 0 <= i < Lg and 0 <= j < Lg:
                x[i * Lg + j] = 1.0

    n_steps = int(T_TOTAL / DT)
    n_rec = n_steps // REC
    m_rec = np.zeros(n_rec); fmis_rec = np.zeros(n_rec); reff_rec = np.zeros(n_rec)
    snaps = {}
    snap_steps = set(int(tt / DT) for tt in snapshot_times) if snapshot_times else set()

    for step in range(n_steps):
        t = step * DT
        coupling = J.dot(x) - x * Jrow
        F = C.a * x - C.b * x ** 3 + coupling + C.A * math.cos(C.omega * t) + C_bias
        x += F * DT + S.levy_inc(alpha, D, DT, N, rng)
        if step % REC == 0:
            k = step // REC
            m_rec[k] = np.mean(x)
            fmis_rec[k] = np.mean(x < -C.theta)
            act = np.abs(x) > C.theta
            if np.any(act):
                reff_rec[k] = math.sqrt(np.mean(np.sum(positions[act] ** 2, axis=1)))
        if step in snap_steps:
            snaps[float(t)] = x.reshape(Lg, Lg).copy()

    snr, eta, famp = analyse(m_rec, fmis_rec, C.A)
    return dict(snr=snr, eta=eta, famp=famp,
                reff=float(reff_rec[-1]),
                vel=S.wavefront_velocity(reff_rec, DT_REC),
                fmis_mean=float(np.mean(fmis_rec[N_TR:])),
                m=m_rec, fmis=fmis_rec, snaps=snaps)


def run_aperiodic2(D, alpha, seed, tau_flip=30.0, A_ap=None):
    J, Jrow, positions = _G['J'], _G['Jrow'], _G['pos']
    if A_ap is None:
        A_ap = C.A
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01
    c = Lg // 2; half = max(2, Lg // 12)
    for i in range(c - half, c + half):
        for j in range(c - half, c + half):
            if 0 <= i < Lg and 0 <= j < Lg:
                x[i * Lg + j] = 1.0
    n_steps = int(T_TOTAL / DT)
    n_rec = n_steps // REC
    S_arr = np.ones(n_steps, dtype=np.int8)
    s = 1; i = 0
    while i < n_steps:
        dur = max(1, int(rng.exponential(tau_flip / DT)))
        S_arr[i:i + dur] = s
        s = -s
        i += dur
    m_rec = np.zeros(n_rec); S_rec = np.zeros(n_rec, dtype=np.int8)
    for step in range(n_steps):
        coupling = J.dot(x) - x * Jrow
        F = C.a * x - C.b * x ** 3 + coupling + A_ap * S_arr[step]
        x += F * DT + S.levy_inc(alpha, D, DT, N, rng)
        if step % REC == 0:
            k = step // REC
            m_rec[k] = np.mean(x)
            S_rec[k] = S_arr[step]
    # 丢弃瞬态
    Y = (m_rec[N_TR:] > 0).astype(np.int8)
    return float(S.mutual_information(S_rec[N_TR:], Y))


# ---------------------------------------------------------------- 任务
def task_A(args):
    a, di, D, ens = args
    seed = C.random_seed + int(D * 1e6) + int(a * 100) + ens * 9973
    snap_t = [10.0, 80.0, 260.0] if ens == 0 else None
    out = run_lattice2(D, a, seed, snapshot_times=snap_t)
    keep = dict(snr=out['snr'], eta=out['eta'], famp=out['famp'],
                reff=out['reff'], vel=out['vel'], fmis_mean=out['fmis_mean'])
    if ens == 0:
        keep['fmis_series'] = out['fmis'][N_TR:].tolist()
        keep['snaps'] = {str(k): v for k, v in out['snaps'].items()}
        keep['m_series'] = out['m'][N_TR:].tolist()
    return a, di, ens, keep


def task_C(args):
    a, di, D, ens = args
    seed = C.random_seed + int(D * 1e6) + int(a * 100) + 555 + ens * 9973
    return a, di, ens, run_aperiodic2(D, a, seed)


def main():
    S.set_style()
    t0 = time.time()
    OUT = S.OUT
    D_vals = np.logspace(C.D_min, C.D_max, C.n_D).tolist()
    alphas = C.alpha_values
    n_ens = C.n_ensemble

    # ---------------- Part A ----------------
    print("\n[Part A] 核心 SR 扫描（瞬态剔除 + 整数周期谱窗口，并行 %d 进程）" % 15)
    jobs = [(a, di, D, ens) for a in alphas for di, D in enumerate(D_vals)
            for ens in range(n_ens)]
    resA = {str(a): {k: [[] for _ in D_vals] for k in
                     ['snr', 'eta', 'famp', 'reff', 'vel', 'fmis_mean']}
            for a in alphas}
    series = {str(a): dict(fmis=[None] * len(D_vals), m=[None] * len(D_vals))
              for a in alphas}
    snaps = {}
    done = 0
    with ProcessPoolExecutor(max_workers=15, initializer=_init) as ex:
        futs = [ex.submit(task_A, j) for j in jobs]
        for f in as_completed(futs):
            a, di, ens, k = f.result()
            key = str(a)
            for fld in ['snr', 'eta', 'reff', 'vel', 'fmis_mean']:
                resA[key][fld][di].append(k[fld])
            resA[key]['famp'][di].append(k['famp'])
            if ens == 0:
                series[key]['fmis'][di] = k['fmis_series']
                series[key]['m'][di] = k['m_series']
                snaps[(a, di)] = {float(t): np.array(v) for t, v in k['snaps'].items()}
            done += 1
            print(f"\r    {done}/{len(jobs)}", end='', flush=True)
    print()

    metaA = dict(D_values=D_vals, alpha_values=alphas,
                 res={str(a): {k: [float(np.mean(v)) for v in resA[str(a)][k]]
                               for k in resA[str(a)]} for a in alphas})
    metaA['res_std'] = {str(a): {k: [float(np.std(v)) for v in resA[str(a)][k]]
                                 for k in resA[str(a)]} for a in alphas}
    # 兼容 sim_v2.make_figures 的字段命名
    for a in alphas:
        k = str(a)
        metaA['res'][k]['snr_s'] = metaA['res_std'][k]['snr']
        metaA['res'][k]['eta_s'] = metaA['res_std'][k]['eta']
        metaA['res'][k]['reff_s'] = metaA['res_std'][k]['reff']
        metaA['res'][k]['vel_s'] = metaA['res_std'][k]['vel']
        metaA['res'][k]['fmis_amp'] = metaA['res'][k]['famp']
        metaA['res'][k]['fmis_mean'] = metaA['res'][k]['fmis_mean']
    with open(os.path.join(OUT, 'partA_core.json'), 'w') as f:
        json.dump(metaA, f, indent=1)

    dstar = {str(a): int(np.argmax(metaA['res'][str(a)]['snr'])) for a in alphas}
    print('  D*:', {a: D_vals[dstar[a]] for a in dstar})
    print('  eta 峰:', {a: D_vals[int(np.argmax(metaA["res"][str(a)]["eta"]))]
                        for a in alphas})

    # ---------------- Part C ----------------
    print("\n[Part C] 非周期 SR 互信息（瞬态剔除）")
    D_asr = np.logspace(-3, 0, 9).tolist()
    jobsC = [(a, di, D, ens) for a in alphas for di, D in enumerate(D_asr)
             for ens in range(n_ens)]
    resC = {str(a): dict(mi=[[] for _ in D_asr], mi_s=[[] for _ in D_asr])
            for a in alphas}
    done = 0
    with ProcessPoolExecutor(max_workers=15, initializer=_init) as ex:
        futs = [ex.submit(task_C, j) for j in jobsC]
        for f in as_completed(futs):
            a, di, ens, mi = f.result()
            resC[str(a)]['mi'][di].append(mi)
            done += 1
            print(f"\r    {done}/{len(jobsC)}", end='', flush=True)
    print()
    for a in alphas:
        k = str(a)
        resC[k]['mi'] = [float(np.mean(v)) for v in resC[k]['mi']]
        resC[k]['mi_s'] = [float(np.std(v)) for v in resC[k]['mi']]
    metaC = dict(D_asr=D_asr, res=resC)
    with open(os.path.join(OUT, 'partC_asr.json'), 'w') as f:
        json.dump(metaC, f, indent=1)
    print('  MI 峰:', {a: D_asr[int(np.argmax(resC[str(a)]['mi']))] for a in alphas})

    # ---------------- 复用 Part B / D ----------------
    with open(os.path.join(OUT, 'partB_heat.json')) as f:
        metaB = json.load(f)
    pd_path = os.path.join(OUT, 'partD_mfpt.json')
    waited = 0
    while not os.path.exists(pd_path) and waited < 3600:
        time.sleep(30); waited += 30
    with open(pd_path) as f:
        resD = json.load(f)

    # ---------------- 绘图 ----------------
    print("\n[绘图] 出版级图形 ...")
    C.n_D = len(D_vals)
    S.make_figures(metaA, metaB, metaC, resD, series, snaps, dstar, D_vals, alphas)
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
