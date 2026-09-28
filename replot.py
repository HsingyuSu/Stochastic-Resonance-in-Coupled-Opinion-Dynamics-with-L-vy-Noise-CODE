# -*- coding: utf-8 -*-
"""用已有的仿真 JSON 重新生成全部 11 张出版级图（不重跑仿真）。

背景：第二轮（sim_v3）跑完后磁盘上的图仍呈现旧的 D 网格，怀疑绘图阶段
使用了陈旧的全局配置。本脚本从 json 重新加载全部数据并强制重绘，
确保图像与数据一致。
"""
import os, sys, json, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_v2 as S

OUT = S.OUT

metaA = json.load(open(os.path.join(OUT, 'partA_core.json'), encoding='utf-8'))
metaB = json.load(open(os.path.join(OUT, 'partB_heat.json'), encoding='utf-8'))
metaC = json.load(open(os.path.join(OUT, 'partC_asr.json'), encoding='utf-8'))
resD = json.load(open(os.path.join(OUT, 'partD_mfpt.json'), encoding='utf-8'))
with open(os.path.join(OUT, 'partA_series.pkl'), 'rb') as f:
    ser = pickle.load(f)
series = ser['series']
snaps = ser['snaps']

D_vals = metaA['D_values']
alphas = metaA['alpha_values']
dstar = {str(a): int(np.argmax(metaA['res'][str(a)]['snr'])) for a in alphas}

print('D_values:', np.round(D_vals, 4))
print('alphas  :', alphas)
print('dstar   :', dstar)
print('snaps keys:', list(snaps.keys())[:6], '...' if len(snaps) > 6 else '')

C = S.C
C.n_D = len(D_vals)

S.set_style()
S.make_figures(metaA, metaB, metaC, resD, series, snaps, dstar, D_vals, alphas)
print('replot done ->', S.FIG)
