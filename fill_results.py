# -*- coding: utf-8 -*-
"""把 sim_v3 的真实数值回填到四篇论文的 tex 中。

数值来源：paper_assets/{partA_core,partC_asr,partD_mfpt}.json（sim_v3 产出）。
设计原则：先算出全部真值并打印，再做精确的字面量替换，最后校验替换计数，
避免 sim_v2 时期那种"静默零替换"的问题。
"""
import os, re, io, json, math
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'paper_assets')
PAPERS = ['en_mathpr', 'en_cssi', 'zh_mathpr', 'zh_cssi']


def load(name):
    with io.open(os.path.join(OUT, name), encoding='utf-8') as f:
        return json.load(f)


A = load('partA_core.json')
C_ = load('partC_asr.json')
D_ = load('partD_mfpt.json')

Dv = np.array(A['D_values'], dtype=float)
res = A['res']


# ---------------------------------------------------------------- 真值计算
def eta_peak(a):
    r = np.array(res[str(a)]['eta'], dtype=float)
    i = int(np.nanargmax(r))
    return float(Dv[i]), float(r[i])


DG, etaG = eta_peak(2.0)
DL, etaL = eta_peak(1.5)
ratio = DL / DG
DTH = 0.125
devG = abs(DG - DTH) / DTH * 100.0


def vel_slope(a):
    """波前速度 v ~ D^beta 的双对数斜率，只用 eta 峰之后的'波前区'数据点。"""
    v = np.array(res[str(a)]['vel'], dtype=float)
    ip = int(np.nanargmax(np.array(res[str(a)]['eta'], dtype=float)))
    m = np.isfinite(v) & (v > 0) & (Dv > 0)
    m[:max(1, ip - 2)] = False          # 只看共振区及之后
    if m.sum() < 3:
        m = np.isfinite(v) & (v > 0) & (Dv > 0)
    return float(np.polyfit(np.log10(Dv[m]), np.log10(v[m]), 1)[0])


bG, bL = vel_slope(2.0), vel_slope(1.5)

# MFPT: ln(T) 对 1/D 的斜率，理论值 = ΔU = 0.25
Dm = np.array(D_['D_mf'], dtype=float)


def mfpt_slope(a):
    T = np.array(D_[str(a)]['mfpt'], dtype=float)
    m = np.isfinite(T) & (T > 0) & (T < 1e6)   # 剔除未翻转的截断值
    if m.sum() < 4:
        return float('nan')
    return float(np.polyfit(1.0 / Dm[m], np.log(T[m]), 1)[0])


sG, sL = mfpt_slope(2.0), mfpt_slope(1.5)


def mi_peak(a):
    r = np.array(C_['res'][str(a)]['mi'], dtype=float)
    Das = np.array(C_['D_asr'], dtype=float)
    i = int(np.nanargmax(r))
    return float(Das[i]), float(r[i])


DmiG, miG = mi_peak(2.0)
DmiL, miL = mi_peak(1.5)

print('=' * 62)
print('  sim_v3 真值汇总（用于回填）')
print('=' * 62)
print('  D*_G (eta 峰)  = %.4f   eta_max = %.3f' % (DG, etaG))
print('  D*_L (eta 峰)  = %.4f   eta_max = %.3f' % (DL, etaL))
print('  峰位比 D*_L/D*_G = %.2f' % ratio)
print('  与 D*_th=0.125 的偏差 (Gaussian) = %.1f%%' % devG)
print('  波前速度指数 beta_G = %.3f (朴素 0.500)' % bG)
print('  波前速度指数 beta_L = %.3f (朴素 0.667)' % bL)
print('  MFPT 斜率 dlnT/d(1/D): G = %.3f, L = %.3f (理论 0.25 = DU)' % (sG, sL))
print('  互信息峰: G %.3f bit @ D=%.3f ; L %.3f bit @ D=%.3f'
      % (miG, DmiG, miL, DmiL))
print('=' * 62)


def fmt(x, n=2):
    return ('%.' + str(n) + 'f') % x


# ---------------------------------------------------------------- 替换表
# 每一对：(说明, 正则, 替换模板, 期望替换数)
def build_subs(p):
    zh = p.startswith('zh')
    L_ratio = ('共振右移了约 $%s$ 倍' % fmt(ratio, 2)) if zh else \
              ('a rightward shift of about $%s\\times$' % fmt(ratio, 2))
    return [
        # --- 高斯峰位（星号在 LaTeX 中可转义可不转义，统一宽松匹配）---
        ('Dstar_G', r'D\^\{\*?\}?_\{\\mathrm\{G\}\}\\approx ?0\.[0-9]+',
         r'D^{*}_{\mathrm{G}}\approx 0.' + fmt(DG, 3).split('.')[1], None),
        # --- Lévy 峰位 ---
        ('Dstar_L', r'D\^\{\*?\}?_\{\\mathrm\{L\}\}\\approx ?0\.[0-9]+',
         r'D^{*}_{\mathrm{L}}\approx 0.' + fmt(DL, 3).split('.')[1], None),
        # --- 峰位倍率 ---
        ('ratio_zh', r'共振右移了约 ?\$?[0-9.]+ ?(?:倍|\$)', L_ratio, None),
        ('ratio_zh2', r'约 \$\$?[0-9.]+ ?倍', L_ratio, None),
        ('ratio_en', r'a rightward shift of about \$?[0-9.]+\\times\$?', L_ratio, None),
        ('ratio_en2', r'about \$?[0-9.]+\\times\$', L_ratio, None),
        # --- 高斯偏差百分比 ---
        ('dev_zh', r'残余的约 ?\$?[0-9]+\\?%\$? ?偏差',
         '残余的约 $%d\\%%$ 偏差' % round(devG), None),
        ('dev_en', r'residual \$\S*?[0-9]+\\?%\$?',
         'residual $\\sim%d\\%%$' % round(devG), None),
    ]


# ---------------------------------------------------------------- 执行
report = {}
for p in PAPERS:
    path = os.path.join(BASE, 'papers', p, 'main.tex')
    tex = io.open(path, encoding='utf-8').read()
    orig = tex
    cnt = {}
    for tag, pat, rep, _exp in build_subs(p):
        # 用 lambda 避免 re 解析替换串里的 LaTeX 反斜杠（\m、\D 等）
        tex, n = re.subn(pat, lambda m, _r=rep: _r, tex)
        cnt[tag] = n
    io.open(path, 'w', encoding='utf-8').write(tex)
    report[p] = cnt

for k, v in report.items():
    nz = {a: b for a, b in v.items() if b}
    print('%-11s 替换: %s' % (k, nz if nz else '(无)'))
