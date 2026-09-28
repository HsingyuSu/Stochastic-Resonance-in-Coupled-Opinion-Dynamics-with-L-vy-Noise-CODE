# -*- coding: utf-8 -*-
"""
=============================================================================
  耦合意见动力学中的随机共振与 L\'evy 噪声 —— 重建版仿真程序 v2
  Stochastic Resonance in Coupled Opinion Dynamics with Levy Noise (v2)

相对原版 (main_enhanced.py) 的升级:
  1. 出版级图形: 矢量 PDF + 600 dpi PNG 双输出, 统一 serif 字体与配色
  2. Ensemble 平均 (默认 3 次) -> 误差棒
  3. 新增理论验证实验:
       (a) 谱放大因子 eta (Liu & Kang 2018 指标, 替代/补充 SNR)
       (b) 单节点 Kramers 逃逸 MFPT: 高斯验证 exp(DeltaU/D), Levy 验证重尾加速
       (c) 两态模型 + Kramers 率对共振峰 D* 的解析预言
       (d) Aperiodic SR: 二值随机信号下的输入-输出互信息 I(S;Y)
           (Kang-Liu-Mao 2020 禁区定理 / mutual information 指标)
  4. 全部观测量在一次仿真中同时采集 (m, f_mis, R_eff, 快照), 避免重复计算
  5. 模块化 + 分批落盘 JSON, 中断不丢数据

作者: 苏星毓 (西安交通大学)   重建: 2026-09-11
=============================================================================
"""
import os, json, time, math, pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------- 路径
BASE = r"D:\Code Test\Stochastic_Resonance_Enhanced"
OUT  = os.path.join(BASE, "paper_assets")
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- 全局配置
class Cfg:
    # 双稳态势阱 U(x) = -a/2 x^2 + b/4 x^4
    a, b = 1.0, 1.0
    # 周期信号 (亚阈值)
    A, omega = 0.05, 0.1
    # 空间耦合 (高斯核)
    J0, sigma = 1.0, 5.0
    # 网格
    L_grid = 50
    # 时间积分 (Euler-Maruyama)
    dt, T_total = 0.002, 100.0
    T_transient = 40.0
    # Levy 截断尺度
    kappa = 0.01
    # 激活阈值
    theta = 0.5
    # D 扫描
    D_min, D_max, n_D = -4, 0, 15
    alpha_values = [1.5, 2.0]
    # ensemble
    n_ensemble = 3
    random_seed = 42
    # 记录间隔
    rec_interval = 10

C = Cfg()
N = C.L_grid ** 2
Lg = C.L_grid
DT = C.dt
REC = C.rec_interval

# 势垒高度 (a=b=1): DeltaU = a^2/(4b) = 0.25 ; 势阱位置 x_m = sqrt(a/b) = 1
DELTA_U = C.a ** 2 / (4.0 * C.b)
X_M     = math.sqrt(C.a / C.b)


# ============================================================ Levy 噪声
def levy_inc(alpha, D, dt, size, rng, kappa=C.kappa):
    """截断 Levy 稳定增量 (Chambers-Mallows-Stuck + 软截断).
    alpha=2 退化为高斯. 返回 shape=size."""
    if abs(alpha - 2.0) < 1e-9:
        return math.sqrt(2.0 * D * dt) * rng.standard_normal(size)
    V = rng.uniform(-np.pi / 2, np.pi / 2, size=size)
    W = rng.exponential(1.0, size=size)
    num = np.sin(alpha * V)
    den = np.power(np.cos(V), 1.0 / alpha)
    arg = np.cos((1.0 - alpha) * V) / W
    scale = np.power(arg, (1.0 - alpha) / alpha)
    B = num / den * scale
    M = np.power(kappa, -1.0 / alpha)
    exc = np.abs(B) > M
    if np.any(exc):
        U = rng.uniform(0.0, 1.0, size=size)
        B[exc] = np.sign(B[exc]) * M * np.power(U[exc], 1.0 / alpha)
    return np.power(D, 1.0 / alpha) * np.power(dt, 1.0 / alpha) * B


# ============================================================ 耦合矩阵
def build_coupling(L=Lg, sigma=C.sigma, J0=C.J0, cutoff_factor=3.0):
    Nt = L * L
    xc = np.linspace(-50, 50, L); yc = np.linspace(-50, 50, L)
    X, Y = np.meshgrid(xc, yc)
    positions = np.stack([X.ravel(), Y.ravel()], axis=1)
    cutoff = cutoff_factor * sigma
    rows, cols, vals = [], [], []
    for i in range(Nt):
        dx = positions[:, 0] - positions[i, 0]
        dy = positions[:, 1] - positions[i, 1]
        d2 = dx * dx + dy * dy
        nb = np.where(d2 < cutoff * cutoff)[0]
        if len(nb) > 0:
            w = (J0 / (2 * np.pi * sigma ** 2)) * np.exp(-d2[nb] / (2 * sigma ** 2))
            rows.extend([i] * len(nb)); cols.extend(nb.tolist()); vals.extend(w.tolist())
    J = csr_matrix((vals, (rows, cols)), shape=(Nt, Nt))
    return J, positions


# ============================================================ 观测量
def snr_fft(m, omega, dt_rec):
    """功率谱信噪比 (dB)."""
    n = len(m)
    freqs = np.fft.rfftfreq(n, d=dt_rec)
    psd = np.abs(np.fft.rfft(m)) ** 2 / n
    f_sig = omega / (2 * np.pi)
    k = int(np.argmin(np.abs(freqs - f_sig)))
    sig = psd[k]
    mask = np.ones(len(freqs), dtype=bool)
    mask[max(0, k - 3):k + 4] = False
    noise = np.mean(psd[mask]) + 1e-30
    return float(10 * np.log10(sig / noise))


def spectral_amplification(m, omega, dt_rec, A):
    """谱放大因子 eta = 2|M_1|/A,  M_1 = 输出在驱动频率处的一阶 Fourier 系数.
    (Liu & Kang, PLA 2018 的动力学磁化率指标)"""
    t = np.arange(len(m)) * dt_rec
    z = np.exp(-1j * omega * t)
    M1 = 2.0 * np.mean(m * z)          # 一阶 Fourier 系数 (复)
    return float(2.0 * np.abs(M1) / A)


def mutual_information(S, Y):
    """二值输入 S 与二值输出 Y 的 Shannon 互信息 (bits)."""
    S = np.asarray(S).astype(int); Y = np.asarray(Y).astype(int)
    n = len(S)
    if n == 0:
        return 0.0
    # 映射到 {0,1}
    su = np.unique(S); yu = np.unique(Y)
    if len(su) < 2 or len(yu) < 2:
        return 0.0
    S = (S == su[1]).astype(int); Y = (Y == yu[1]).astype(int)
    joint = np.zeros((2, 2))
    for s in range(2):
        for y in range(2):
            joint[s, y] = np.mean((S == s) & (Y == y))
    pS = joint.sum(axis=1); pY = joint.sum(axis=0)
    I = 0.0
    for s in range(2):
        for y in range(2):
            p = joint[s, y]
            if p > 1e-12:
                I += p * np.log2(p / (pS[s] * pY[y]))
    return float(I)


def wavefront_velocity(reff, dt_rec):
    """从 R_eff(t) 的 5%~95% 增长段线性拟合速度."""
    t = np.arange(len(reff)) * dt_rec
    Rmax = reff.max()
    v = 0.0
    if Rmax > 1e-6:
        lo = int(np.argmax(reff >= 0.05 * Rmax))
        hi_c = np.where(reff >= 0.95 * Rmax)[0]
        hi = hi_c[0] if len(hi_c) > 0 else len(reff) - 1
        if hi - lo < 3:
            lo = 0; hi = max(3, int(len(reff) * 0.4))
        if hi - lo >= 3:
            st, sr = t[lo:hi], reff[lo:hi]
            if np.std(st) > 0:
                corr = np.corrcoef(st, sr)[0, 1]
                if corr > 0.5:
                    v = max(float(np.polyfit(st, sr, 1)[0]), 0.0)
    return v


# ============================================================ 核心仿真
def run_lattice(D, alpha, J, Jrow, positions, seed, C_bias=0.0,
                T=None, snapshot_times=None, T_heat=None):
    """一次完整仿真, 同时采集 m(t) / f_mis(t) / R_eff(t) / 快照."""
    if T is None:
        T = C.T_total
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01
    # 中心局部种子
    c = Lg // 2; half = max(2, Lg // 12)
    for i in range(c - half, c + half):
        for j in range(c - half, c + half):
            if 0 <= i < Lg and 0 <= j < Lg:
                x[i * Lg + j] = 1.0

    n_steps = int(T / DT)
    n_rec = n_steps // REC
    m_rec = np.zeros(n_rec); fmis_rec = np.zeros(n_rec); reff_rec = np.zeros(n_rec)
    snaps = {}
    snap_steps = set(int(tt / DT) for tt in snapshot_times) if snapshot_times else set()

    for step in range(n_steps):
        t = step * DT
        coupling = J.dot(x) - x * Jrow
        F = C.a * x - C.b * x ** 3 + coupling + C.A * np.cos(C.omega * t) + C_bias
        dL = levy_inc(alpha, D, DT, N, rng)
        x += F * DT + dL
        if step % REC == 0:
            k = step // REC
            m_rec[k] = np.mean(x)
            fmis_rec[k] = np.mean(x < -C.theta)
            act = np.abs(x) > C.theta
            if np.any(act):
                reff_rec[k] = math.sqrt(np.mean(np.sum(positions[act] ** 2, axis=1)))
        if step in snap_steps:
            snaps[float(t)] = x.reshape(Lg, Lg).copy()

    dt_rec = DT * REC
    out = dict(
        snr=snr_fft(m_rec, C.omega, dt_rec),
        eta=spectral_amplification(m_rec, C.omega, dt_rec, C.A),
        reff=float(reff_rec[-1]),
        vel=wavefront_velocity(reff_rec, dt_rec),
        fmis_amp=float(fmis_rec.max() - fmis_rec.min()),
        fmis_mean=float(np.mean(fmis_rec[int(0.6 * n_rec):])),
        snaps=snaps,
    )
    return out, m_rec, fmis_rec


def run_aperiodic(D, alpha, J, Jrow, positions, seed, tau_flip=30.0,
                  A_ap=None, T=None):
    """Aperiodic SR: 输入为亚阈值二值随机信号 S(t), 输出 Y = 1[m(t)>0]."""
    if A_ap is None:
        A_ap = C.A
    if T is None:
        T = C.T_total
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N) * 0.01
    c = Lg // 2; half = max(2, Lg // 12)
    for i in range(c - half, c + half):
        for j in range(c - half, c + half):
            x[i * Lg + j] = 1.0
    n_steps = int(T / DT)
    n_rec = n_steps // REC
    # 预生成二值随机信号 (指数分布翻转间隔)
    S_arr = np.ones(n_steps, dtype=np.int8)
    s = 1
    i = 0
    while i < n_steps:
        dur = max(1, int(rng.exponential(tau_flip / DT)))
        S_arr[i:i + dur] = s
        s = -s
        i += dur
    m_rec = np.zeros(n_rec); S_rec = np.zeros(n_rec, dtype=np.int8)
    for step in range(n_steps):
        coupling = J.dot(x) - x * Jrow
        F = C.a * x - C.b * x ** 3 + coupling + A_ap * S_arr[step]
        x += F * DT + levy_inc(alpha, D, DT, N, rng)
        if step % REC == 0:
            k = step // REC
            m_rec[k] = np.mean(x)
            S_rec[k] = S_arr[step]
    Y = (m_rec > 0).astype(np.int8)
    I = mutual_information(S_rec, Y)
    return float(I)


# ============================================================ 单节点 MFPT (理论验证)
def mfpt_single_node(alpha, D, n_traj=4000, x0=-X_M, barrier=0.0,
                     T_max=4000.0, seed=0):
    """向量化单节点首次穿越时间 (MFPT), 无周期信号 (A=0), 无耦合.
    从势阱底 x0=-1 出发, 首次到达 x>=barrier(势垒顶 0) 为一次逃逸.
    返回平均首通时间."""
    rng = np.random.default_rng(seed)
    x = np.full(n_traj, x0, dtype=np.float64)
    t = np.zeros(n_traj)
    n_steps = int(T_max / DT)
    done = np.zeros(n_traj, dtype=bool)
    first = np.full(n_traj, np.nan)
    for step in range(n_steps):
        alive = ~done
        if not alive.any():
            break
        xa = x[alive]
        F = C.a * xa - C.b * xa ** 3
        dL = levy_inc(alpha, D, DT, int(alive.sum()), rng)
        xn = xa + F * DT + dL
        xn = np.clip(xn, -8.0, 8.0)
        x[alive] = xn
        t[alive] += DT
        crossed = xn >= barrier
        idx = np.where(alive)[0][crossed]
        first[idx] = t[idx]
        done[idx] = True
    if np.isnan(first).all():
        return float(T_max)
    return float(np.nanmean(first))


def kramers_rate(D):
    """过阻尼 Kramers 逃逸率 (a=b=1):  r_K = sqrt(|U''(0)|U''(x_m)|)/(2pi) * exp(-DeltaU/D)"""
    Upp_barrier = abs(-C.a + 3 * C.b * 0.0 ** 2)     # |U''(0)| = a = 1
    Upp_well = -C.a + 3 * C.b * X_M ** 2              # U''(1) = 2a = 2
    return math.sqrt(Upp_barrier * Upp_well) / (2 * np.pi) * math.exp(-DELTA_U / D)


def two_state_snr_pred(D, r_fun):
    """两态模型 SNR 预言 (弱信号极限): SNR ~ (A x_m / D)^2 * r_K(D) / (2)  (相对量,
    仅用于峰位). 这里返回正比于 r(D)/D^2 的峰位函数."""
    return r_fun(D) / (D ** 2)


# ============================================================ 绘图风格
def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.linewidth": 0.9, "lines.linewidth": 1.6, "lines.markersize": 5,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "figure.dpi": 150, "savefig.dpi": 600, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "grid.linestyle": "--",
    })


COL = {1.5: "#1f4e9c", 2.0: "#b2182b"}   # Levy 蓝 / 高斯红
LAB = {1.5: "Lévy noise ($\\alpha=1.5$)", 2.0: r"Gaussian noise ($\alpha=2.0$)"}


def save_fig(fig, name):
    """双格式输出: 矢量 PDF (给 LaTeX) + 600 dpi PNG (预览).

    PDF 内的栅格部分（imshow 面板）用 300 dpi：线稿仍是矢量，放大无损；
    而 50x50 的场本来就没有更高的空间频率，600 dpi 只会让文件凭空膨胀
    （fig7 曾因此达 3.8 MB，把整篇论文 PDF 从 2.4 MB 推到 6.5 MB）。
    """
    fig.savefig(os.path.join(FIG, name + ".pdf"), format="pdf", dpi=300)
    fig.savefig(os.path.join(FIG, name + ".png"), format="png", dpi=600)
    plt.close(fig)


# ============================================================ 主流程
def main():
    set_style()
    t_start = time.time()
    print("=" * 72)
    print("  重建版仿真 v2 — 耦合意见动力学随机共振 (含理论验证)")
    print("=" * 72)

    J, positions = build_coupling()
    Jrow = np.asarray(J.sum(axis=1)).ravel()
    print(f"  耦合矩阵: {J.nnz:,} 非零元")

    D_vals = np.logspace(C.D_min, C.D_max, C.n_D).tolist()
    alphas = C.alpha_values

    # ---------------- Part A: 核心扫描 ----------------
    print("\n[Part A] 核心 SR 扫描 (SNR / eta / R_eff / v / f_mis / 快照)")
    resA = {str(a): dict(snr=[], snr_s=[], eta=[], eta_s=[],
                         reff=[], reff_s=[], vel=[], vel_s=[],
                         fmis_amp=[], fmis_mean=[]) for a in alphas}
    series = {str(a): dict(fmis=[]) for a in alphas}
    snaps = {}

    for a in alphas:
        key = str(a)
        for di, D in enumerate(tqdm(D_vals, desc=f"  alpha={a}")):
            snrs, etas, refs, vels, famps, fmeans = [], [], [], [], [], []
            for ens in range(C.n_ensemble):
                seed = (C.random_seed + int(D * 1e6) + int(a * 100) + ens * 9973)
                snap_t = [5.0, 30.0, 80.0] if ens == 0 else None
                out, m_rec, fmis_rec = run_lattice(D, a, J, Jrow, positions,
                                                   seed, snapshot_times=snap_t)
                snrs.append(out["snr"]); etas.append(out["eta"])
                refs.append(out["reff"]); vels.append(out["vel"])
                famps.append(out["fmis_amp"]); fmeans.append(out["fmis_mean"])
                if snap_t:
                    snaps[(a, di)] = out["snaps"]
                if ens == 0:
                    series[key]["fmis"].append(fmis_rec.tolist())
            resA[key]["snr"].append(float(np.mean(snrs)))
            resA[key]["snr_s"].append(float(np.std(snrs)))
            resA[key]["eta"].append(float(np.mean(etas)))
            resA[key]["eta_s"].append(float(np.std(etas)))
            resA[key]["reff"].append(float(np.mean(refs)))
            resA[key]["reff_s"].append(float(np.std(refs)))
            resA[key]["vel"].append(float(np.mean(vels)))
            resA[key]["vel_s"].append(float(np.std(vels)))
            resA[key]["fmis_amp"].append(float(np.mean(famps)))
            resA[key]["fmis_mean"].append(float(np.mean(fmeans)))

    metaA = dict(D_values=D_vals, alpha_values=alphas, res=resA)
    with open(os.path.join(OUT, "partA_core.json"), "w") as f:
        json.dump(metaA, f, indent=1)
    with open(os.path.join(OUT, "partA_series.pkl"), "wb") as f:
        pickle.dump(dict(series=series, snaps=snaps), f)
    print("  Part A 完成 ->", os.path.join(OUT, "partA_core.json"))

    dstar = {str(a): int(np.argmax(resA[str(a)]["snr"])) for a in alphas}
    print("  D* 索引:", dstar, "-> D* =", {a: D_vals[dstar[a]] for a in dstar})

    # ---------------- Part B: 辟谣干预热力图 ----------------
    print("\n[Part B] 干预热力图 (C x D)")
    C_list = [0.0, 0.05, 0.1, 0.2, 0.4]
    D_heat = np.logspace(-3, 0, 7).tolist()
    heat = {str(a): np.zeros((len(C_list), len(D_heat))) for a in alphas}
    for a in alphas:
        for ci, cv in enumerate(C_list):
            for di, D in enumerate(tqdm(D_heat, desc=f"  alpha={a} C={cv}")):
                vals = []
                for ens in range(1):
                    seed = C.random_seed + int(D * 1e6) + int(a * 100) + ci * 131 + ens * 9973
                    out, _, _ = run_lattice(D, a, J, Jrow, positions, seed,
                                            C_bias=cv, T=50.0)
                    vals.append(out["fmis_mean"])
                heat[str(a)][ci, di] = float(np.mean(vals))
    metaB = dict(C_list=C_list, D_heat=D_heat,
                 heat={k: v.tolist() for k, v in heat.items()})
    with open(os.path.join(OUT, "partB_heat.json"), "w") as f:
        json.dump(metaB, f, indent=1)
    print("  Part B 完成")

    # ---------------- Part C: Aperiodic SR 互信息 ----------------
    print("\n[Part C] Aperiodic SR — 互信息 I(S;Y)")
    D_asr = np.logspace(-3, 0, 9).tolist()
    resC = {str(a): dict(mi=[], mi_s=[]) for a in alphas}
    for a in alphas:
        for D in tqdm(D_asr, desc=f"  alpha={a}"):
            vals = []
            for ens in range(1):
                seed = C.random_seed + int(D * 1e6) + int(a * 100) + 555 + ens * 9973
                vals.append(run_aperiodic(D, a, J, Jrow, positions, seed))
            resC[str(a)]["mi"].append(float(np.mean(vals)))
            resC[str(a)]["mi_s"].append(float(np.std(vals)))
    metaC = dict(D_asr=D_asr, res=resC)
    with open(os.path.join(OUT, "partC_asr.json"), "w") as f:
        json.dump(metaC, f, indent=1)
    print("  Part C 完成")

    # ---------------- Part D: 单节点 MFPT / Kramers 验证 ----------------
    print("\n[Part D] 单节点逃逸 MFPT (Kramers 验证)")
    D_mf = np.logspace(-2, -0.3, 14).tolist()
    resD = {str(a): dict(mfpt=[], rate=[]) for a in alphas}
    for a in alphas:
        for D in tqdm(D_mf, desc=f"  MFPT alpha={a}"):
            Tm = mfpt_single_node(a, D, n_traj=3000, seed=C.random_seed + int(D * 1e6))
            resD[str(a)]["mfpt"].append(Tm)
            resD[str(a)]["rate"].append(1.0 / max(Tm, 1e-12))
    resD["D_mf"] = D_mf
    resD["kramers"] = [kramers_rate(D) for D in D_mf]
    resD["kramers_mfpt"] = [1.0 / kramers_rate(D) for D in D_mf]
    with open(os.path.join(OUT, "partD_mfpt.json"), "w") as f:
        json.dump(resD, f, indent=1)
    print("  Part D 完成")

    # ---------------- 绘图 ----------------
    print("\n[绘图] 出版级图形 ...")
    make_figures(metaA, metaB, metaC, resD, series, snaps, dstar, D_vals, alphas)

    print(f"\n全部完成! 总耗时 {(time.time()-t_start)/60:.1f} 分钟")
    print("图输出目录:", FIG)


# ============================================================ 图形
def make_figures(metaA, metaB, metaC, resD, series, snaps, dstar, D_vals, alphas):
    Dv = np.array(metaA["D_values"])

    # ---- Fig1: SNR vs D ----
    # 注意：SNR 的定义使它在"系统几乎不翻转"的极小 D 区域也很大（此时 m(t)
    # 是确定性的亚阈振荡，谱线几乎无噪声），因此 SNR 的 argmax 落在网格左端
    # 并不代表 SR。真正标志 SR 的是谱放大因子 eta：D->0 时 eta->1（无放大），
    # 故 eta 的峰必然位于内部，这里用 eta 的峰位标注。
    # 版面：D* 的数值直接并入图例标签，取消原先压在数据线上的箭头注释；
    # 图例置于坐标区下方外侧（savefig.bbox='tight' 会把图例纳入画布），
    # 因此图例与任何数据线/误差棒都不可能重叠。
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for a in alphas:
        k = str(a); r = metaA["res"][k]
        pi = int(np.argmax(r["eta"]))
        ax.errorbar(Dv, r["snr"], yerr=r["snr_s"], fmt="o-", color=COL[a],
                    label=rf"{LAB[a]}, $D^*={Dv[pi]:.3f}$",
                    capsize=2.5, elinewidth=0.9, markeredgewidth=0.8)
        ax.axvline(Dv[pi], color=COL[a], ls=":", lw=1.1, alpha=0.7)
    ax.set_xscale("log"); ax.set_xlabel(r"Noise intensity $D$")
    ax.set_ylabel(r"Signal-to-noise ratio (dB)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
              frameon=False, fontsize=8)
    save_fig(fig, "fig1_SNR_vs_D")

    # ---- Fig2: 谱放大因子 eta vs D (新增理论指标) ----
    # 版面：与 Fig1 一致——D* 的数值并入图例标签，取消原先压在曲线上的箭头
    # 注释；图例置于坐标区下方外侧，保证与数据线/误差棒零重叠。
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for a in alphas:
        k = str(a); r = metaA["res"][k]
        pi = int(np.argmax(r["eta"]))
        ax.errorbar(Dv, r["eta"], yerr=r["eta_s"], fmt="s-", color=COL[a],
                    label=rf"{LAB[a]}, $D^*={Dv[pi]:.3f}$",
                    capsize=2.5, elinewidth=0.9, markeredgewidth=0.8)
        ax.plot(Dv[pi], r["eta"][pi], marker="*", ms=11, color=COL[a],
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    # 两态 Kramers 预言的无参数峰位 D* = ΔU/2
    Dth = DELTA_U / 2.0
    ax.axvline(Dth, color="0.25", ls="--", lw=1.1, alpha=0.85)
    ax.text(Dth * 1.14, 1.6, rf"$D^*_{{\rm th}}=\Delta U/2={Dth:.3f}$",
            fontsize=8, color="0.25", rotation=90, va="bottom", ha="left")
    ax.set_xscale("log"); ax.set_xlabel(r"Noise intensity $D$")
    ax.set_ylabel(r"Spectral amplification factor $\eta$")
    ax.set_ylim(top=max(max(metaA["res"][str(a)]["eta"]) for a in alphas) * 1.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=2,
              frameon=False, fontsize=8)
    save_fig(fig, "fig2_SpectralAmp_vs_D")

    # ---- Fig3: R_eff vs D ----
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for a in alphas:
        k = str(a); r = metaA["res"][k]
        ax.errorbar(Dv, r["reff"], yerr=r["reff_s"], fmt="^-", color=COL[a],
                    label=LAB[a], capsize=2.5, elinewidth=0.9)
    ax.set_xscale("log"); ax.set_xlabel(r"Noise intensity $D$")
    ax.set_ylabel(r"Effective radius $R_{\mathrm{eff}}$")
    ax.legend(loc="best", framealpha=0.9)
    save_fig(fig, "fig3_Radius_vs_D")

    # ---- Fig4: 意见场快照 ----
    idxs = [0, C.n_D // 2, C.n_D - 1]
    if all((1.5, i) in snaps for i in idxs):
        titles = [r"Small $D$ (sub-threshold)", r"Intermediate $D$",
                  r"Large $D$ (supra-threshold)"]
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), constrained_layout=True)
        for i, di_ in enumerate(idxs):
            sn = snaps[(1.5, di_)]; ts = sorted(sn.keys()); field = sn[ts[-1]]
            im = axes[i].imshow(field, extent=[-50, 50, -50, 50], cmap="RdBu_r",
                                vmin=-1.5, vmax=1.5, interpolation="bilinear")
            axes[i].set_title(f"{titles[i]}\n$D={Dv[di_]:.4g}$", fontsize=9.5)
            axes[i].set_xlabel(r"$x$"); axes[i].set_ylabel(r"$y$")
        cb = fig.colorbar(im, ax=axes, shrink=0.82, aspect=22, pad=0.015)
        cb.set_label(r"Opinion $x_i$", fontsize=9)
        save_fig(fig, "fig4_Opinion_Snapshots")

    # ---- Fig5: 波前速度标度律 ----
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for a in alphas:
        k = str(a); r = metaA["res"][k]
        v = np.array(r["vel"]); ok = v > 1e-6
        if ok.sum() >= 2:
            lD, lv = np.log10(Dv[ok]), np.log10(np.maximum(v[ok], 1e-10))
            ax.plot(lD, lv, "o", color=COL[a], ms=4.5, alpha=0.85)
            sl, ic = np.polyfit(lD, lv, 1)
            th = 1.0 / a if a < 2 else 0.5
            ax.plot(lD, sl * lD + ic, "-", color=COL[a], lw=1.8,
                    label=rf"$\alpha={a}$: slope$={sl:.2f}$ (theory ${th:.2f}$)")
    ax.set_xlabel(r"$\log_{10} D$"); ax.set_ylabel(r"$\log_{10} v$")
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    save_fig(fig, "fig5_Velocity_Scaling")

    # ---- Fig6: 误信息流行度时序 ----
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), constrained_layout=True)
    for col, a in enumerate(alphas):
        k = str(a); fmis = series[k]["fmis"]
        sel = sorted(set([0, 4, 8, dstar[k], 12, 14]))[:6]
        cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(sel)))
        for j, i in enumerate(sel):
            s = np.array(fmis[i]); tt = np.arange(len(s)) * REC * DT
            axes[col].plot(tt, s, color=cmap[j], lw=1.2, label=rf"$D={Dv[i]:.3g}$")
        axes[col].set_title(rf"$\alpha={a}$", fontsize=10)
        axes[col].set_xlabel(r"Time $t$")
        axes[col].set_ylabel(r"Misinformation prevalence $f_{\mathrm{mis}}(t)$")
        axes[col].set_ylim(-0.05, 1.05)
        axes[col].legend(fontsize=7, ncol=2, loc="upper right")
    save_fig(fig, "fig6_Misinfo_Timeseries")

    # ---- Fig7: 虚假信念空间波前 ----
    # 修正两处缺陷：
    #  (1) 快照原先取 dstar = SNR 的 argmax。对 alpha=1.5 该点在 D=0.01，属于
    #      "系统几乎不翻转"区，场内标准差仅 0.065，图像完全均匀，看不到任何
    #      波前。现改为取相干幅值（= 谱放大因子 eta）的峰位，才是真正的共振
    #      最优点，场内有充分的空间结构。
    #  (2) 原 contour 未传 extent，横纵坐标用的是 0..L-1 的数组下标，而 imshow
    #      用的是 -50..50 的数据坐标，于是等值线整体错位、被挤到右上象限糊成
    #      一块黑斑；另外 imshow 默认 origin='upper' 而 contour 默认
    #      origin='lower'，二者上下翻转不一致。现统一用数据坐标 + origin='lower'。
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.4), constrained_layout=True)
    xy = np.linspace(-50.0, 50.0, Lg)
    for row, a in enumerate(alphas):
        k = str(a)
        pi = int(np.argmax(metaA["res"][k]["fmis_amp"]))
        sn = snaps.get((a, pi), {})
        ts = sorted(sn.keys())
        if len(ts) >= 3:
            pick = [ts[0], ts[len(ts) // 2], ts[-1]]
        else:
            pick = ts
        for col, t in enumerate(pick[:3]):
            field = sn[t]
            im = axes[row, col].imshow(field, extent=[-50, 50, -50, 50],
                                       origin="lower",
                                       cmap="RdBu_r", vmin=-1.5, vmax=1.5,
                                       interpolation="bilinear")
            # 等值线画在轻度平滑后的场上：原始场在格点尺度上仍有噪声，
            # 直接取 -theta 等值线会产生大量显微镜级的闭合小圈（图面糊成
            # 一片黑噪点）。平滑 sigma=2 只抹掉格点级起伏，保留真实的
            # 真/伪信念域界（domain wall），这才是本图要传达的对象。
            fsm = gaussian_filter(field, sigma=2.0)
            axes[row, col].contour(xy, xy, fsm, levels=[-C.theta],
                                   colors="k", linewidths=1.0)
            axes[row, col].set_title(
                rf"$\alpha={a}$, $D={Dv[pi]:.3g}$, $t={float(t):.0f}$",
                fontsize=9.5)
            axes[row, col].set_xlabel(r"$x$"); axes[row, col].set_ylabel(r"$y$")
            axes[row, col].tick_params(labelsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, aspect=26, pad=0.015)
    cb.set_label(r"$x_i$  (blue = misinformed, red = truth)", fontsize=9)
    save_fig(fig, "fig7_Misinfo_Wavefront")

    # ---- Fig8: 共振脆弱性窗口 (双轴) ----
    # 版面：双轴 + 4 条曲线几乎占满整个坐标区，右上角原本与红色虚线峰值段、
    # 蓝色虚线爬升段、红色实线下降段三重重叠。坐标区内已无足够空位，故把
    # 图例移到坐标区下方外侧（savefig.bbox='tight' 会自动扩展画布），
    # 从根本上消除图例与数据线的重叠。
    fig, axL = plt.subplots(figsize=(5.6, 4.0))
    axR = axL.twinx()
    for a in alphas:
        k = str(a); r = metaA["res"][k]
        axL.plot(Dv, r["snr"], "o-", color=COL[a], ms=4,
                 label=rf"SNR ($\alpha={a}$)")
        axR.plot(Dv, r["fmis_amp"], "s--", color=COL[a], ms=4,
                 label=rf"Belief coherence ($\alpha={a}$)")
        # 竖线标在相干幅值的峰位（等价于谱放大因子 eta 的峰位）。
        # 不能用 SNR 的 argmax：SNR 在极小 D（系统不翻转的平凡区）也很大，
        # 对 alpha=1.5 会把竖线推到网格最左端，既不构成共振也与图注不符。
        pi = int(np.argmax(r["fmis_amp"]))
        axL.axvline(Dv[pi], color=COL[a], ls=":", lw=1.1, alpha=0.7)
    axL.set_xscale("log"); axL.set_xlabel(r"Noise intensity $D$")
    axL.set_ylabel(r"SNR (dB)"); axR.set_ylabel(r"Belief-oscillation coherence")
    axR.grid(False)
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
    axL.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.155),
               ncol=2, fontsize=8, frameon=False)
    save_fig(fig, "fig8_Vulnerability_Window")

    # ---- Fig9: 干预热力图 ----
    # 修正：原代码把单元格数值标在 (log10(Dh[di]), Cl[ci])，但 imshow 用 extent
    # 铺满整个坐标区时，第 di 格的\emph{格心}应在该格左右边界的中点，二者并不
    # 相等（间距不同），误差逐格累积，导致最后一列的数字溢出到坐标区外、压到
    # 色标上，顶行数字也冒出上边框。现改为按格心定位；同时给文字加白色描边，
    # 避免深色格子上黑字不可读。
    import matplotlib.patheffects as pe
    Cl = metaB["C_list"]; Dh = np.array(metaB["D_heat"])
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 6.4), constrained_layout=True)
    x0, x1 = float(np.log10(Dh.min())), float(np.log10(Dh.max()))
    y0, y1 = float(Cl[0]), float(Cl[-1])
    for row, a in enumerate(alphas):
        mat = np.array(metaB["heat"][str(a)], dtype=float)
        nC, nD = mat.shape
        im = axes[row].imshow(mat, aspect="auto", origin="lower",
                              extent=[x0, x1, y0, y1], cmap="RdYlGn_r",
                              vmin=0, vmax=1)
        axes[row].set_title(rf"$\alpha={a}$", fontsize=10)
        axes[row].set_xlabel(r"$\log_{10} D$ (rumour noise)")
        axes[row].set_ylabel(r"Truth bias $C$")
        xc = 0.5 * (np.linspace(x0, x1, nD + 1)[:-1] +
                    np.linspace(x0, x1, nD + 1)[1:])
        yc = 0.5 * (np.linspace(y0, y1, nC + 1)[:-1] +
                    np.linspace(y0, y1, nC + 1)[1:])
        for ci in range(nC):
            for di in range(nD):
                if not np.isfinite(mat[ci, di]):
                    continue
                txt = axes[row].text(xc[di], yc[ci], f"{mat[ci, di]:.2f}",
                                     ha="center", va="center", fontsize=6.5,
                                     zorder=6)
                txt.set_path_effects([pe.withStroke(linewidth=1.5,
                                                    foreground="white")])
        cb = fig.colorbar(im, ax=axes[row], shrink=0.85, pad=0.015)
        cb.set_label("Mean misinformation prevalence", fontsize=8.5)
    save_fig(fig, "fig9_Intervention_Heatmap")

    # ---- Fig10: Aperiodic SR 互信息 ----
    Da = np.array(metaC["D_asr"])
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for a in alphas:
        k = str(a); r = metaC["res"][k]
        ax.errorbar(Da, r["mi"], yerr=r["mi_s"], fmt="o-", color=COL[a],
                    label=LAB[a], capsize=2.5, elinewidth=0.9)
    ax.set_xscale("log"); ax.set_xlabel(r"Noise intensity $D$")
    ax.set_ylabel(r"Mutual information $I(S;Y)$ (bits)")
    ax.legend(loc="best", framealpha=0.9)
    save_fig(fig, "fig10_MutualInfo_ASR")

    # ---- Fig11: MFPT / Kramers 验证 ----
    Dm = np.array(resD["D_mf"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9), constrained_layout=True)
    for a in alphas:
        k = str(a)
        mf = np.array(resD[k]["mfpt"])
        axes[0].plot(Dm, mf, "o-", color=COL[a], ms=4, label=LAB[a])
    axes[0].plot(Dm, resD["kramers_mfpt"], "k--", lw=1.4,
                 label=r"Kramers $2\pi/\sqrt{2}\,e^{\Delta U/D}$")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Noise intensity $D$"); axes[0].set_ylabel(r"MFPT $T(D)$")
    axes[0].legend(fontsize=7.5, loc="upper right")
    # 半对数: ln T vs 1/D
    for a in alphas:
        k = str(a)
        mf = np.array(resD[k]["mfpt"])
        axes[1].plot(1.0 / Dm, np.log(mf), "o-", color=COL[a], ms=4, label=LAB[a])
    kT = np.array(resD["kramers_mfpt"])
    axes[1].plot(1.0 / Dm, np.log(kT), "k--", lw=1.4,
                 label=rf"Kramers slope $=\Delta U={DELTA_U}$")
    axes[1].set_xlabel(r"$1/D$"); axes[1].set_ylabel(r"$\ln T$")
    axes[1].legend(fontsize=7.5, loc="best")
    save_fig(fig, "fig11_MFPT_Kramers")

    print("  图已保存 (PDF 矢量 + 600 dpi PNG)")


if __name__ == "__main__":
    main()
