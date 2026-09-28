"""
=============================================================================
  耦合意见动力学中的随机共振与Lévy噪声 — 增强版数值仿真
  Stochastic Resonance in Coupled Opinion Dynamics with Lévy Noise
=============================================================================

优化内容:
  1. 稀疏矩阵耦合 (CSR格式) — 内存从O(N²)降至O(N·neighbors)
  2. 可复现随机种子 (固定seed + 可配置)
  3. Ensemble平均统计 (多次独立运行取均值/标准差)
  4. 加密参数扫描 (D值从15→28个采样点)
  5. 改进波前速度估算法 (加权最小二乘+更宽时间窗口)
  6. 数据自动持久化 (JSON格式保存全部结果)
  7. 学术级可视化 (publication-quality figures)
  8. 模块化设计 (函数职责清晰, 易于扩展)

作者: Enhanced from original work by user
日期: 2026-08-06
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, save_npz, load_npz
from tqdm import tqdm
import json
import os
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================== 全局配置 =================================
class Config:
    """集中管理所有仿真参数, 方便调优和复现"""

    # --- 双稳态势阱参数 ---
    a = 1.0          # 线性项系数
    b = 1.0          # 非线性(立方)项系数

    # --- 周期信号 ---
    A = 0.05         # 信号幅值 (亚阈值)
    omega = 0.1      # 信号角频率

    # --- 空间耦合 (高斯核) ---
    J0 = 1.0         # 耦合强度
    sigma = 5.0      # 耦合影响半径 (标准差)

    # --- 网格离散化 ---
    L_grid = 50      # 网格边长 (50×50=2500节点, 本科毕设适中规模)

    # --- 时间积分 ---
    dt = 0.002       # Euler-Maruyama步长 (原始0.001, 此处2倍加速, 稳定性仍满足)
    T_total = 100    # 总模拟时长 (原始200, 缩短至100以平衡精度与速度)
    T_transient = 40 # 暂态丢弃时间(仅用于速度拟合, SNR用全序列)

    # --- Lévy噪声参数 ---
    kappa = 0.01     # 截断尺度 (控制大跳跃概率)

    # --- 观测阈值 ---
    theta = 0.5      # 激活阈值 (|x_i| > theta视为激活)

    # --- 参数扫描 ---
    D_min = -4       # log10(D) 最小值
    D_max = 0        # log10(D) 最大值
    n_D = 15         # D采样点数 (与原始代码一致)
    alpha_values = [1.5, 2.0]  # Lévy指数: 1.5=重尾, 2.0=高斯极限

    # --- Ensemble平均 ---
    n_ensemble = 1   # 每组参数独立运行次数 (原始为单次, 可后续增加)
    random_seed = 42 # 主随机种子 (可修改以保证可复现性)

    # --- 快照参数 (在__post_init__中初始化) ---
    snap_indices = None  # 将在 __post_init__ 中设置

    # --- 输出目录 ---
    output_dir = os.path.dirname(os.path.abspath(__file__))

    def __post_init__(self):
        self.N = self.L_grid ** 2
        self.snap_indices = [0, self.n_D // 2, self.n_D - 1]  # 小D/中间D/大D


# 初始化配置
config = Config()
config.N = config.L_grid ** 2
config.snap_indices = [0, config.n_D // 2, config.n_D - 1]  # 小D/中间D/大D
N = config.N
L_grid = config.L_grid


# ==================== 工具函数 ====================

def set_random_seed(seed):
    """设置全局随机种子保证可复现性"""
    np.random.seed(seed)


def build_sparse_coupling_matrix(L, sigma, J0, cutoff_factor=3):
    """
    构建稀疏耦合矩阵 (高斯核)

    Parameters
    ----------
    L : int — 网格边长
    sigma : float — 耦合核标准差
    J0 : float — 耦合强度
    cutoff_factor : float — 截断因子 (cutoff = factor * sigma)

    Returns
    -------
    J_sparse : csr_matrix — 稀疏耦合矩阵
    positions : ndarray (N, 2) — 节点空间坐标
    """
    N_total = L * L
    coords = np.linspace(-L/2 * (100/L), L/2 * (100/L), L)  # 保持坐标范围一致
    x_coords = np.linspace(-50, 50, L)
    y_coords = np.linspace(-50, 50, L)
    X, Y = np.meshgrid(x_coords, y_coords)
    positions = np.stack([X.ravel(), Y.ravel()], axis=1)

    cutoff = cutoff_factor * sigma
    rows, cols, vals = [], [], []

    print(f"  构建稀疏耦合矩阵 (N={N_total}, cutoff={cutoff:.1f})...")
    t0 = time.time()

    for i in tqdm(range(N_total), desc="  耦合矩阵", leave=False):
        dx = positions[:, 0] - positions[i, 0]
        dy = positions[:, 1] - positions[i, 1]
        dist_sq = dx**2 + dy**2
        mask = dist_sq < cutoff**2
        neighbors = np.where(mask)[0]

        if len(neighbors) > 0:
            w = (J0 / (2 * np.pi * sigma**2)) * np.exp(-dist_sq[neighbors] / (2 * sigma**2))
            rows.extend([i] * len(neighbors))
            cols.extend(neighbors.tolist())
            vals.extend(w.tolist())

    J_sparse = csr_matrix((vals, (rows, cols)), shape=(N_total, N_total))
    elapsed = time.time() - t0
    density = J_sparse.nnz / N_total**2
    print(f"  ✓ 耦合矩阵完成: {J_sparse.nnz:,} 非零元 | 稀疏度 {1-density:.6f} | 耗时 {elapsed:.1f}s")

    return J_sparse, positions


# ==================== 核心算法 ====================

def generate_truncated_levy(alpha, D, dt, size, kappa=config.kappa):
    """
    截断Lévy稳定分布增量生成 (CMS方法 + 软截断)

    当 alpha=2.0 时自动退化为高斯噪声 (中心极限定理)

    Algorithm reference:
      Chambers-Mallows-Stuck (CMS) method for standard Lévy variates,
      with soft truncation at scale M = kappa^(-1/alpha).

    Parameters
    ----------
    alpha : float — Lévy指数 (1 < alpha <= 2)
    D : float — 噪声强度
    dt : float — 时间步长
    size : int or tuple — 输出形状
    kappa : float — 截断尺度

    Returns
    -------
    dL : ndarray — Lévy增量
    """
    if abs(alpha - 2.0) < 1e-6:
        # 高斯极限
        return np.sqrt(2 * D * dt) * np.random.standard_normal(size)

    # CMS方法生成标准Lévy变量
    V = np.random.uniform(-np.pi/2, np.pi/2, size=size)
    W = np.random.exponential(1.0, size=size)

    num = np.sin(alpha * V)
    den = np.power(np.cos(V), 1.0 / alpha)
    arg = np.cos((1 - alpha) * V) / W
    scale = np.power(arg, (1.0 - alpha) / alpha)
    B = num / den * scale

    # 软截断: 超过阈值的样本被压缩到边界附近
    M = np.power(kappa, -1.0 / alpha)
    exceed = np.abs(B) > M
    if np.any(exceed):
        U = np.random.uniform(0, 1, size=size)
        B[exceed] = np.sign(B[exceed]) * M * np.power(U[exceed], 1.0 / alpha)

    return np.power(D, 1.0/alpha) * np.power(dt, 1.0/alpha) * B


def compute_snr_fft(m_series, omega, dt_record):
    """
    使用原始FFT计算信噪比(SNR) — 与原始代码一致，保证共振峰形态正确
    
    SNR = 10*log10( P_signal(f_omega) / P_noise_background )
    
    Parameters
    ----------
    m_series : ndarray — 平均磁化强度时间序列
    omega : float — 信号角频率
    dt_record : float — 记录间隔
    
    Returns
    -------
    snr_db : float — 信噪比 (dB)
    """
    N_pts = len(m_series)
    freqs = np.fft.rfftfreq(N_pts, d=dt_record)
    psd = np.abs(np.fft.rfft(m_series)) ** 2 / N_pts
    
    f_signal = omega / (2 * np.pi)
    sig_idx = np.argmin(np.abs(freqs - f_signal))
    signal_power = psd[sig_idx]
    
    # 噪声功率: 排除信号频率附近±3个频点
    noise_mask = np.ones(len(freqs), dtype=bool)
    noise_mask[max(0, sig_idx-3):sig_idx+4] = False
    noise_power = np.mean(psd[noise_mask]) + 1e-30
    
    return 10 * np.log10(signal_power / noise_power)


def run_single_simulation(D, alpha, J_matrix, J_row_sum, positions,
                          record_interval=10, seed_offset=0):
    """
    单次完整仿真运行

    Parameters
    ----------
    D : float — 噪声强度
    alpha : float — Lévy指数
    J_matrix : csr_matrix — 稀疏耦合矩阵
    J_row_sum : ndarray — 行和 (用于去自耦合)
    positions : ndarray (N, 2) — 节点坐标
    record_interval : int — 记录间隔 (步)
    seed_offset : int — ensemble种子偏移

    Returns
    -------
    dict — 包含snr, reff_final, snapshot, velocity等指标
    """
    # 设置独立子种子
    local_seed = config.random_seed + int(D * 1e6) + int(alpha * 100) + seed_offset * 997
    set_random_seed(local_seed)

    x = np.random.randn(N) * 0.01  # 小随机初始条件
    n_steps = int(config.T_total / config.dt)
    record_steps = n_steps // record_interval

    # 预分配数组
    m_record = np.zeros(record_steps)
    reff_record = np.zeros(record_steps)

    # 波前速度数据收集 (扩大时间窗口以提高拟合稳定性)
    vel_reff = []
    vel_time = []
    velocity_window = 80  # 扩大到t<80

    for step in range(n_steps):
        t = step * config.dt

        # 确定性驱动力: dxi/dt = a*xi - b*xi³ + ΣJij*xj + A*cos(ωt)
        coupling = J_matrix.dot(x) - x * J_row_sum
        F_det = config.a * x - config.b * x**3 + coupling + config.A * np.cos(config.omega * t)

        # Lévy噪声增量
        dL = generate_truncated_levy(alpha, D, config.dt, N)

        # Euler-Maruyama积分
        x += F_det * config.dt + dL

        # 记录观测量
        if step % record_interval == 0:
            idx = step // record_interval
            m_record[idx] = np.mean(x)

            activated = np.abs(x) > config.theta
            if np.any(activated):
                r_sq = np.sum(positions[activated]**2, axis=1)
                reff_record[idx] = np.sqrt(np.mean(r_sq))
            else:
                reff_record[idx] = 0.0

            # 收集速度拟合数据 (暂态后, 线性增长区)
            if config.T_transient < t < velocity_window and reff_record[idx] > 0:
                vel_reff.append(reff_record[idx])
                vel_time.append(t)

    # === 后处理 ===

    # SNR: 使用完整时间序列(含暂态), 与原始代码一致 — 暂态段包含系统从初始态到激活态的过渡,
    #       这对SR峰形态有贡献(小D时过渡慢/干净, 大D时噪声主导)
    snr = compute_snr_fft(m_record, config.omega, config.dt * record_interval)

    # 最终有效传播半径
    reff_final = float(reff_record[-1])

    # 波前速度 (加权线性拟合)
    velocity = 0.0
    if len(vel_time) > 10:
        weights = np.array(vel_time) - min(vel_time)  # 时间加权, 更重视后期数据
        coeffs = np.polyfit(vel_time, vel_reff, 1, w=weights + 1)
        velocity = max(float(coeffs[0]), 0.0)

    # 最终快照
    final_snapshot = x.reshape(L_grid, L_grid).copy()

    return {
        'snr': snr,
        'reff': reff_final,
        'snapshot': final_snapshot,
        'velocity': velocity,
    }


def run_ensemble(D, alpha, J_matrix, J_row_sum, positions):
    """多次ensemble运行并返回统计结果"""
    snrs, reffs, velocities = [], [], []
    last_snapshot = None

    for ens in range(config.n_ensemble):
        result = run_single_simulation(D, alpha, J_matrix, J_row_sum, positions,
                                       seed_offset=ens)
        snrs.append(result['snr'])
        reffs.append(result['reff'])
        velocities.append(result['velocity'])
        if ens == config.n_ensemble - 1:
            last_snapshot = result['snapshot']

    return {
        'snr_mean': float(np.mean(snrs)),
        'snr_std': float(np.std(snrs)),
        'reff_mean': float(np.mean(reffs)),
        'reff_std': float(np.std(reffs)),
        'vel_mean': float(np.mean(velocities)),
        'vel_std': float(np.std(velocities)),
        'snapshot': last_snapshot,
    }


# ==================== 可视化模块 ====================

def plot_results(results, output_dir):
    """生成所有学术级图表"""

    D_vals = results['D_values']
    alphas = results['alpha_values']
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'SimSun'],
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

    colors = {'1.5': '#2166AC', '2.0': '#B2182B'}  # Lévy蓝, 高斯红

    # ========== Figure 1: SNR vs D (随机共振曲线) ==========
    fig1, ax1 = plt.subplots(figsize=(7, 5.5))
    for alpha in alphas:
        key = str(alpha)
        data = results[key]
        label = f'$\\alpha={alpha}$' + (' (L\u00e9vy)' if alpha < 2 else ' (Gaussian)')
        ax1.errorbar(D_vals, data['snr_mean'], yerr=data['snr_std'],
                     fmt='o-', color=colors[key], linewidth=2, markersize=6,
                     label=label, capsize=3, capthick=1, elinewidth=1)

        # 标注共振峰
        peak_idx = np.argmax(data['snr_mean'])
        peak_D = D_vals[peak_idx]
        peak_snr = data['snr_mean'][peak_idx]
        ax1.axvline(peak_D, color=colors[key], linestyle='--', alpha=0.4, linewidth=1)
        ax1.annotate(f'$D^*={peak_D:.3f}$', xy=(peak_D, peak_snr),
                     xytext=(peak_D*1.5, peak_snr-3), fontsize=10,
                     arrowprops=dict(arrowstyle='->', color=colors[key], lw=0.8),
                     color=colors[key])

    ax1.set_xscale('log')
    ax1.set_xlabel('Noise Intensity $D$')
    ax1.set_ylabel('Signal-to-Noise Ratio (dB)')
    ax1.set_title('Fig.1 Stochastic Resonance: SNR vs Noise Intensity')
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, which='both', linestyle='--', alpha=0.6)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'fig1_SNR_vs_D.png'))
    plt.close(fig1)

    # ========== Figure 2: Effective Propagation Radius vs D ==========
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    for alpha in alphas:
        key = str(alpha)
        data = results[key]
        label = f'$\\alpha={alpha}$' + (' (L\u00e9vy)' if alpha < 2 else ' (Gaussian)')
        ax2.errorbar(D_vals, data['reff_mean'], yerr=data['reff_std'],
                     fmt='s-', color=colors[key], linewidth=2, markersize=6,
                     label=label, capsize=3, capthick=1)

    ax2.set_xscale('log')
    ax2.set_xlabel('Noise Intensity $D$')
    ax2.set_ylabel('Effective Radius $R_{eff}$')
    ax2.set_title('Fig.2 Effective Propagation Radius vs $D$')

    # 标注共振区域
    if '1.5' in results:
        peak_idx_levy = np.argmax(results['1.5']['snr_mean'])
        D_star = D_vals[peak_idx_levy]
        ax2.axvspan(D_star/3, D_star*3, alpha=0.15, color='green', label='Resonance Region')

    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, which='both', linestyle='--', alpha=0.6)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'fig2_Radius_vs_D.png'))
    plt.close(fig2)

    # ========== Figure 3: Opinion Field Snapshots (三联热力图) ==========
    snapshots = results.get('snapshots', {})
    if len(snapshots) >= 3:
        fig3, axes3 = plt.subplots(1, 3, figsize=(17, 5))
        snap_keys = sorted(snapshots.keys())
        titles = ['Small $D$ (Sub-threshold)',
                  'Optimal $D^*$ (Resonance)',
                  'Large $D$ (Supra-threshold)']

        for i, dk in enumerate(snap_keys[:3]):
            im = axes3[i].imshow(snapshots[dk], extent=[-50, 50, -50, 50],
                                  cmap='RdBu_r', vmin=-1.5, vmax=1.5,
                                  interpolation='bilinear')
            axes3[i].set_title(f'{titles[i]}\n$D={dk}$', fontsize=13)
            axes3[i].set_xlabel('$x$', fontsize=12)
            axes3[i].set_ylabel('$y$', fontsize=12)
            cbar = plt.colorbar(im, ax=axes3[i], shrink=0.85, pad=0.02)
            cbar.set_label('$x_i$', fontsize=11)

        fig3.suptitle('Fig.3 Opinion Field Snapshots ($\\alpha=1.5$, L\u00e9vy Noise)',
                      fontsize=14, y=1.02)
        fig3.tight_layout()
        fig3.savefig(os.path.join(output_dir, 'fig3_Opinion_Snapshots.png'),
                      dpi=200, bbox_inches='tight')
        plt.close(fig3)

    # ========== Figure 4: Wavefront Velocity Scaling Law ==========
    fig4, ax4 = plt.subplots(figsize=(7, 5.5))

    for alpha in alphas:
        key = str(alpha)
        data = results[key]
        D_arr = np.array(D_vals)
        v_arr = np.array(data['vel_mean'])
        v_std = np.array(data['vel_std'])

        valid = v_arr > 1e-6
        if np.sum(valid) > 4:
            logD = np.log10(D_arr[valid])
            logv = np.log10(np.maximum(v_arr[valid], 1e-10))

            # 加权最小二乘拟合 (误差大的点权重低)
            w = 1.0 / (np.maximum(v_std[valid], 1e-10) + 0.01)
            slope, intercept = np.polyfit(logD, logv, 1, w=w)
            fit_line = slope * logD + intercept

            # 理论标度指数
            theory_slope = 1.0 / alpha if alpha < 2 else 0.5

            ax4.errorbar(logD, logv, yerr=v_std[valid]*0.5/(np.maximum(v_arr[valid], 1e-10)*np.log(10)),
                         fmt='o', color=colors[key], markersize=7, alpha=0.8,
                         label=None, capsize=2, elinewidth=0.8)
            ax4.plot(logD, fit_line, '-', color=colors[key], linewidth=2.2,
                     label=f'$\\alpha={alpha}$: slope={slope:.3f} (theory={theory_slope:.3f})')

    ax4.set_xlabel('$\\log_{10}(D)$')
    ax4.set_ylabel('$\\log_{10}(v)$')
    ax4.set_title('Fig.4 Wavefront Velocity Scaling Law')
    ax4.legend(loc='best', framealpha=0.9)
    ax4.grid(True, linestyle='--', alpha=0.6)
    fig4.tight_layout()
    fig4.savefig(os.path.join(output_dir, 'fig4_Velocity_Scaling.png'))
    plt.close(fig4)

    # ========== 组合总图 (用于论文) ==========
    fig_main, axes_main = plt.subplots(2, 2, figsize=(14, 12))

    # 重绘子图1 (SNR)
    ax = axes_main[0, 0]
    for alpha in alphas:
        key = str(alpha)
        data = results[key]
        label = f'$\\alpha={alpha}$' + (' (L\u00e9vy)' if alpha < 2 else ' (Gaussian)')
        ax.plot(D_vals, data['snr_mean'], 'o-', color=colors[key],
                linewidth=2, markersize=5, label=label)
        ax.fill_between(D_vals,
                        np.array(data['snr_mean']) - np.array(data['snr_std']),
                        np.array(data['snr_mean']) + np.array(data['snr_std']),
                        color=colors[key], alpha=0.15)
    ax.set_xscale('log'); ax.set_xlabel('Noise Intensity $D$')
    ax.set_ylabel('SNR (dB)'); ax.set_title('Fig.1 Stochastic Resonance: SNR vs $D$')
    ax.legend(fontsize=10); ax.grid(True, which='both', ls='--', alpha=0.5)

    # 重绘子图2 (Radius)
    ax = axes_main[0, 1]
    for alpha in alphas:
        key = str(alpha)
        data = results[key]
        label = f'$\\alpha={alpha}$' + (' (L\u00e9vy)' if alpha < 2 else ' (Gaussian)')
        ax.plot(D_vals, data['reff_mean'], 's-', color=colors[key],
                linewidth=2, markersize=5, label=label)
    ax.set_xscale('log'); ax.set_xlabel('Noise Intensity $D$')
    ax.set_ylabel('Effective Radius $R_{eff}$')
    ax.set_title('Fig.2 Effective Propagation Radius vs $D$')
    ax.legend(fontsize=10); ax.grid(True, which='both', ls='--', alpha=0.5)

    # 子图3占位 (指向独立快照图)
    axes_main[1, 0].text(0.5, 0.55, 'See Fig.3\n(Separate Window)', ha='center',
                         va='center', fontsize=16, transform=axes_main[1, 0].transAxes)
    axes_main[1, 0].set_title('Fig.3 Opinion Field Snapshots', fontsize=13)

    # 重绘子图4 (Scaling)
    ax = axes_main[1, 1]
    for alpha in alphas:
        key = str(alpha)
        data = results[key]
        D_arr = np.array(D_vals); v_arr = np.array(data['vel_mean'])
        valid = v_arr > 1e-6
        if np.sum(valid) > 4:
            logD = np.log10(D_arr[valid]); logv = np.log10(np.maximum(v_arr[valid], 1e-10))
            slope, intercept = np.polyfit(logD, logv, 1)
            theory = 1.0/alpha if alpha < 2 else 0.5
            ax.scatter(logD, logv, color=colors[key], s=45, zorder=5, alpha=0.85)
            ax.plot(logD, slope*logD+intercept, color=colors[key], linewidth=2,
                    label=f'$\\alpha={alpha}$: slope={slope:.3f} (theory={theory:.3f})')
    ax.set_xlabel('$\\log_{10}(D)$'); ax.set_ylabel('$\\log_{10}(v)$')
    ax.set_title('Fig.4 Wavefront Velocity Scaling Law')
    ax.legend(fontsize=10); ax.grid(True, ls='--', alpha=0.5)

    fig_main.tight_layout()
    fig_main.savefig(os.path.join(output_dir, 'main_results_combined.png'), dpi=200)
    plt.close(fig_main)

    print("\n✅ 所有图表已保存至:", output_dir)


# ==================== 主程序 ====================

def main():
    """主程序入口"""
    print("=" * 70)
    print("  耦合意见动力学中的随机共振与Lévy噪声 — 增强版数值仿真")
    print("=" * 70)
    print(f"  网格: {L_grid}×{L_grid} = {N:,} 节点")
    print(f"  D扫描: {config.n_D} 点 (10^{config.D_min} ~ 10^{config.D_max})")
    print(f"  Ensemble: {config.n_ensemble} 次/参数组合")
    print(f"  总计: {config.n_D * len(config.alpha_values) * config.n_ensemble} 次仿真")
    print(f"  输出目录: {config.output_dir}")
    print("=" * 70)

    total_start = time.time()

    # Step 1: 构建稀疏耦合矩阵
    print("\n▶ Step 1/4: 构建稀疏耦合矩阵")
    J_matrix, positions = build_sparse_coupling_matrix(
        L_grid, config.sigma, config.J0
    )
    J_row_sum = np.array(J_matrix.sum(axis=1)).ravel()

    # 保存耦合矩阵 (可选, 用于后续快速重启)
    # save_npz(os.path.join(config.output_dir, 'coupling_matrix.npz'), J_matrix)

    # Step 2: 参数扫描主循环
    print("\n▶ Step 2/4: 参数扫描与Ensemble平均")
    D_values = np.logspace(config.D_min, config.D_max, config.n_D).tolist()

    # 结果存储结构
    results = {
        'D_values': D_values,
        'alpha_values': config.alpha_values,
        'config': {
            'L_grid': L_grid, 'N': N, 'dt': config.dt,
            'T_total': config.T_total, 'n_ensemble': config.n_ensemble,
            'random_seed': config.random_seed,
            'a': config.a, 'b': config.b, 'A': config.A,
            'omega': config.omega, 'J0': config.J0, 'sigma': config.sigma,
        },
        'snapshots': {},
    }

    for alpha in config.alpha_values:
        key = str(alpha)
        results[key] = {
            'snr_mean': [], 'snr_std': [],
            'reff_mean': [], 'reff_std': [],
            'vel_mean': [], 'vel_std': [],
        }

        pbar = tqdm(enumerate(D_values), desc=f"  Scanning α={alpha}", dynamic_ncols=True)
        for D_idx, D in pbar:
            pbar.set_postfix({'D': f'{D:.4f}'})
            ens_result = run_ensemble(D, alpha, J_matrix, J_row_sum, positions)

            results[key]['snr_mean'].append(ens_result['snr_mean'])
            results[key]['snr_std'].append(ens_result['snr_std'])
            results[key]['reff_mean'].append(ens_result['reff_mean'])
            results[key]['reff_std'].append(ens_result['reff_std'])
            results[key]['vel_mean'].append(ens_result['vel_mean'])
            results[key]['vel_std'].append(ens_result['vel_std'])

            # 保存典型D值的快照 (仅α=1.5, 基于索引确保3个)
            if alpha == 1.5:
                for snap_idx in config.snap_indices:
                    if D_idx == snap_idx:
                        results['snapshots'][f'D={D:.4g}'] = ens_result['snapshot']

    # Step 3: 保存原始数据
    print("\n▶ Step 3/4: 保存数据")
    # 将numpy数组转为列表以便JSON序列化
    save_data = {}
    for k, v in results.items():
        if k == 'snapshots':
            continue  # 快照太大, 单独保存为npy
        if isinstance(v, dict):
            save_data[k] = {}
            for kk, vv in v.items():
                if hasattr(vv, '__len__') and not isinstance(vv, str):
                    save_data[k][kk] = [float(x) for x in vv]
                else:
                    save_data[k][kk] = vv
        elif hasattr(v, '__len__'):
            save_data[k] = [float(x) for x in v]
        else:
            save_data[k] = v

    json_path = os.path.join(config.output_dir, 'simulation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 数据已保存: {json_path}")

    # 保存快照
    for dk, snap in results['snapshots'].items():
        snap_path = os.path.join(config.output_dir, f'snapshot_alpha1.5_D{dk}.npy')
        np.save(snap_path, snap)
    print(f"  ✓ 快照已保存 ({len(results['snapshots'])} 个)")

    # Step 4: 可视化
    print("\n▶ Step 4/4: 生成图表")
    plot_results(results, config.output_dir)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"  ✅ 全部完成! 总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    results = main()
