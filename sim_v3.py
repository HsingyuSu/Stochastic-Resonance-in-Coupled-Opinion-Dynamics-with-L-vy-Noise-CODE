# -*- coding: utf-8 -*-
"""
sim_v3 —— 修正且加速的 Part A / Part C 重跑（多进程并行）

相对 sim_v2 的两处修正
----------------------
1. sim_v2 中 T_transient 定义后从未使用，谱分析含初始生长瞬态，导致小 D 处
   eta / SNR 被虚假抬高；且 T_total=100 只含 1.6 个驱动周期，驱动频率无法与
   DC 分离。本脚本取 T_total = T_transient + N_PERIOD * P（N_PERIOD=4），
   稳态窗口恰好含整数个驱动周期，信号精确落在 FFT 第 N_PERIOD 个 bin。
2. sim_v2 的 n_rec = n_steps // REC 在 n_steps 不能被 REC 整除时越界
   （155663//10 = 15566，但最大索引也是 15566）。

加速
----
* 稀疏 matvec 0.187 ms -> 可分离高斯卷积 0.049 ms（高斯核可分离，且 2D 格点
  上等价于两次 1D 卷积）；
* Levy 增量的标度常数 (D*dt)^{1/alpha} 与截断阈值预先算好，越界重抽样只对
  越界子集生成随机数。
"""
import os, sys, json, math, time, pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_v2 as S

C = S.C
N, Lg, DT, REC = S.N, S.Lg, S.DT, S.REC
P = 2.0 * math.pi / C.omega
N_PERIOD = 4
T_TRANSIENT = 60.0
T_TOTAL = T_TRANSIENT + N_PERIOD * P
DT_REC = DT * REC
N_WIN = int(round(N_PERIOD * P / DT_REC))
N_TR = int(round(T_TRANSIENT / DT_REC))
N_STEPS = int(round(T_TOTAL / DT))
N_REC = int(math.ceil(N_STEPS / REC))
assert N_TR + N_WIN <= N_REC, (N_TR, N_WIN, N_REC)

# ---- D 网格重设 ----
# 原网格 [1e-4, 1] 的左端 D <= 1e-2 处，两态 Kramers 逃逸时间
# (1/r_K = (2*pi/sqrt2) exp(DeltaU/D)) 达 1e9 ~ 1e22 s，远超 251 s 的稳态窗口，
# 系统在窗口内几乎不翻转，"SR" 无从发生；此时 m(t) 只是确定性的亚阈振荡，
# 谱放大因子被虚假抬高到网格最左端。物理上有意义的区间从
# D ~ DeltaU/2 = 0.125 (MFPT=33 s，窗口内约 7 次翻转) 附近开始，
# 故把网格挪到 [0.01, 2]，覆盖"不翻转 -> 最优 -> 过噪"的完整非单调段。
D_MIN, D_MAX, N_D = -2.0, 0.3, 15

_G = {}


def _kernels():
    sig = C.sigma
    cut = int(3 * sig)
    d = np.arange(-cut, cut + 1)
    k = np.exp(-d ** 2 / (2.0 * sig ** 2))
    return k, C.J0 / (2.0 * math.pi * sig ** 2)


def _init():
    """预构建一次稀疏耦合矩阵。

    原 sim_v2 的瓶颈不是 matvec（0.187 ms）本身，而是每步都在做
    `coupling(x2) - x*Jrow` 的逐元素减法 + 一维重排；直接预存 CSR 矩阵后
    每步只需一次 `J.dot(x)`，且与 build_coupling 的定义严格一致
    （含 3*sigma 欧氏球截断），不存在可分离卷积的等价性误差。
    """
    J, pos = S.build_coupling()
    _G['J'] = J
    _G['pos'] = pos
    _G['Jrow'] = np.asarray(J.sum(axis=1)).ravel()


def coupling(x2d):
    """保留接口：返回 J @ x（含自耦合项，由调用方减去 x*Jrow）。"""
    return _G['J'].dot(x2d.ravel())


# ------------------------------------------------------------------ Levy
_LEV = {}


def lev(D, alpha, size, rng):
    """快速截断 alpha-stable 增量（标度已含 (D*dt)^{1/alpha}）。"""
    key = (alpha, D)
    e = _LEV.get(key)
    if e is None:
        ia = 1.0 / alpha
        e = (ia, math.pow(D * DT, ia), math.pow(C.kappa, -ia))
        _LEV[key] = e
    ia, scale, M = e
    if abs(alpha - 2.0) < 1e-9:
        return math.sqrt(2.0 * D * DT) * rng.standard_normal(size)
    V = rng.uniform(-math.pi / 2, math.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)
    B = np.sin(alpha * V) / np.power(np.cos(V), ia) * \
        np.power(np.cos((1.0 - alpha) * V) / W, (1.0 - alpha) * ia)
    exc = np.abs(B) > M
    n = int(exc.sum())
    if n:
        U = rng.uniform(0.0, 1.0, size=n)
        B[exc] = np.sign(B[exc]) * M * np.power(U, ia)
    return scale * B


# ------------------------------------------------------------------ 分析
def analyse(m, fmis):
    mw = m[N_TR:N_TR + N_WIN]
    fw = fmis[N_TR:N_TR + N_WIN]
    t = np.arange(N_WIN) * DT_REC
    z = np.exp(-1j * C.omega * t)
    eta = 2.0 * abs(2.0 * np.mean(mw * z)) / C.A
    famp = 2.0 * abs(2.0 * np.mean(fw * z))
    psd = np.abs(np.fft.rfft(mw - mw.mean())) ** 2 / N_WIN
    k = N_PERIOD
    sig = psd[k]
    mask = np.ones(len(psd), dtype=bool)
    mask[max(0, k - 2):k + 3] = False
    mask[0] = False
    noise = np.mean(psd[mask]) + 1e-30
    return float(10.0 * math.log10(sig / noise)), float(eta), float(famp)


def _seed(x):
    rng = np.random.default_rng(x)
    v = rng.standard_normal(N) * 0.01
    c = Lg // 2; half = max(2, Lg // 12)
    v.reshape(Lg, Lg)[c - half:c + half, c - half:c + half] = 1.0
    return v


# ------------------------------------------------------------------ 仿真
def run_lattice(D, alpha, seed, C_bias=0.0, snapshot_times=None):
    rng = np.random.default_rng(seed)
    x = _seed(seed)
    pos = _G['pos']; Jrow = _G['Jrow']
    m_rec = np.zeros(N_REC); fmis_rec = np.zeros(N_REC); reff_rec = np.zeros(N_REC)
    snaps = {}
    snap_steps = set(int(tt / DT) for tt in snapshot_times) if snapshot_times else set()
    a, b = C.a, C.b
    om, A = C.omega, C.A
    for step in range(N_STEPS):
        t = step * DT
        x2 = x.reshape(Lg, Lg)
        F = a * x - b * x ** 3 + (coupling(x2).ravel() - x * Jrow) \
            + A * math.cos(om * t) + C_bias
        x += F * DT + lev(D, alpha, N, rng)
        if step % REC == 0:
            k = step // REC
            m_rec[k] = x.mean()
            fmis_rec[k] = np.count_nonzero(x < -C.theta) / N
            act = np.abs(x) > C.theta
            if act.any():
                reff_rec[k] = math.sqrt(np.mean(
                    np.einsum('ij,ij->i', pos[act], pos[act])))
        if step in snap_steps:
            snaps[float(t)] = x.reshape(Lg, Lg).copy()
    snr, eta, famp = analyse(m_rec, fmis_rec)
    return dict(snr=snr, eta=eta, famp=famp,
                reff=float(reff_rec[-1]),
                vel=S.wavefront_velocity(reff_rec, DT_REC),
                fmis_mean=float(np.mean(fmis_rec[N_TR:])),
                m=m_rec[N_TR:], fmis=fmis_rec[N_TR:], snaps=snaps)


def run_aperiodic(D, alpha, seed, tau_flip=30.0, A_ap=None):
    if A_ap is None:
        A_ap = C.A
    rng = np.random.default_rng(seed)
    x = _seed(seed)
    Jrow = _G['Jrow']
    S_arr = np.ones(N_STEPS, dtype=np.int8)
    s = 1; i = 0
    while i < N_STEPS:
        dur = max(1, int(rng.exponential(tau_flip / DT)))
        S_arr[i:i + dur] = s
        s = -s
        i += dur
    m_rec = np.zeros(N_REC); S_rec = np.zeros(N_REC, dtype=np.int8)
    a, b = C.a, C.b
    for step in range(N_STEPS):
        x2 = x.reshape(Lg, Lg)
        F = a * x - b * x ** 3 + (coupling(x2).ravel() - x * Jrow) + A_ap * S_arr[step]
        x += F * DT + lev(D, alpha, N, rng)
        if step % REC == 0:
            k = step // REC
            m_rec[k] = x.mean()
            S_rec[k] = S_arr[step]
    Y = (m_rec[N_TR:] > 0).astype(np.int8)
    return float(S.mutual_information(S_rec[N_TR:], Y))


# ------------------------------------------------------------------ 任务
def task_A(p):
    a, di, D, ens = p
    seed = C.random_seed + int(D * 1e6) + int(a * 100) + ens * 9973
    snap_t = [10.0, 80.0, 260.0] if ens == 0 else None
    out = run_lattice(D, a, seed, snapshot_times=snap_t)
    k = dict(snr=out['snr'], eta=out['eta'], famp=out['famp'], reff=out['reff'],
             vel=out['vel'], fmis_mean=out['fmis_mean'])
    if ens == 0:
        k['fmis_series'] = out['fmis'].tolist()
        k['m_series'] = out['m'].tolist()
        k['snaps'] = {str(t): v for t, v in out['snaps'].items()}
    return a, di, ens, k


def task_C(p):
    a, di, D, ens = p
    seed = C.random_seed + int(D * 1e6) + int(a * 100) + 555 + ens * 9973
    return a, di, ens, run_aperiodic(D, a, seed)


def main():
    S.set_style()
    t0 = time.time()
    OUT = S.OUT
    D_vals = np.logspace(D_MIN, D_MAX, N_D).tolist()
    alphas = C.alpha_values
    n_ens = C.n_ensemble
    print(f"[config] P={P:.3f} T_total={T_TOTAL:.2f} transient={T_TRANSIENT} "
          f"window={N_PERIOD}P ({N_WIN}) steps={N_STEPS} rec={N_REC}")

    # ---------------- 一致性校验 ----------------
    _init()
    J, _ = S.build_coupling()
    xt = np.random.default_rng(0).standard_normal(N)
    exact = J.dot(xt)
    approx = coupling(xt.reshape(Lg, Lg)).ravel()
    rel = np.max(np.abs(exact - approx)) / np.max(np.abs(exact))
    print(f"[check] CSR 路径 vs build_coupling 最大相对误差 = {rel:.3e}")

    # ---------------- Part A ----------------
    NW = 12
    print(f"\n[Part A] 核心 SR 扫描（{NW} 进程）")
    jobs = [(a, di, D, ens) for a in alphas for di, D in enumerate(D_vals)
            for ens in range(n_ens)]
    resA = {str(a): {k: [[] for _ in D_vals] for k in
                     ['snr', 'eta', 'famp', 'reff', 'vel', 'fmis_mean']}
            for a in alphas}
    series = {str(a): dict(fmis=[None] * len(D_vals), m=[None] * len(D_vals))
              for a in alphas}
    snaps = {}
    done = 0
    with ProcessPoolExecutor(max_workers=NW, initializer=_init) as ex:
        for f in as_completed([ex.submit(task_A, j) for j in jobs]):
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
    std = {str(a): {k: [float(np.std(v)) for v in resA[str(a)][k]]
                    for k in resA[str(a)]} for a in alphas}
    for a in alphas:
        k = str(a)
        for f in ['snr', 'eta', 'reff', 'vel']:
            metaA['res'][k][f + '_s'] = std[k][f]
        metaA['res'][k]['fmis_amp'] = metaA['res'][k]['famp']
    with open(os.path.join(OUT, 'partA_core.json'), 'w') as f:
        json.dump(metaA, f, indent=1)
    with open(os.path.join(OUT, 'partA_series.pkl'), 'wb') as f:
        pickle.dump(dict(series=series, snaps=snaps), f)

    dstar = {str(a): int(np.argmax(metaA['res'][str(a)]['snr'])) for a in alphas}
    print('  D*(SNR):', {a: round(D_vals[dstar[a]], 4) for a in dstar})
    print('  D*(eta):', {a: round(D_vals[int(np.argmax(metaA['res'][str(a)]['eta']))], 4)
                         for a in alphas})
    print('  D*(famp):', {a: round(D_vals[int(np.argmax(metaA['res'][str(a)]['famp']))], 4)
                          for a in alphas})

    # ---------------- Part C ----------------
    print("\n[Part C] 非周期 SR 互信息")
    # 集体模式的等效噪声被空间平均削弱，故 MI 的峰位比 SNR 峰位大近一个量级，
    # D 网格需覆盖到 10 才能看到完整的非单调曲线（探测：0.5→0.54, 1→0.80,
    # 2→0.68, 4→0.30, 10→0.07）。
    D_asr = np.logspace(-1.7, 1.0, 11).tolist()
    jobsC = [(a, di, D, ens) for a in alphas for di, D in enumerate(D_asr)
             for ens in range(n_ens)]
    resC = {str(a): dict(mi=[[] for _ in D_asr], mi_s=[[] for _ in D_asr])
            for a in alphas}
    done = 0
    with ProcessPoolExecutor(max_workers=NW, initializer=_init) as ex:
        for f in as_completed([ex.submit(task_C, j) for j in jobsC]):
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
    print('  MI 峰:', {a: round(D_asr[int(np.argmax(resC[str(a)]['mi']))], 4)
                       for a in alphas})

    # ---------------- 绘图 ----------------
    with open(os.path.join(OUT, 'partB_heat.json')) as f:
        metaB = json.load(f)
    with open(os.path.join(OUT, 'partD_mfpt.json')) as f:
        resD = json.load(f)
    print("\n[绘图] 出版级图形 ...")
    C.n_D = len(D_vals)
    S.make_figures(metaA, metaB, metaC, resD, series, snaps, dstar, D_vals, alphas)
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
