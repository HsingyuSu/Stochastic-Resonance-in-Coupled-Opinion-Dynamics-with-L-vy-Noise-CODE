# -*- coding: utf-8 -*-
"""修正 fig11（逃逸统计）正文说明。

原句（由本轮 fill_results 写入）称 Lévy 数据"弯到直线下方"，与图 11 右panel
的实际数据不符：仿真中 \Levy 的 ln T 位于 Kramers 直线\emph{上方}，并随 1/D
增大而\emph{饱和}（次线性）。高斯数据才是位于直线\emph{下方}（预因子差异）。
本脚本按真实数据改写，并补上最左三点被右删失在 T_max=3000 的事实。
"""
import io, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(BASE, 'papers')
BS = chr(92)


def B(n):
    return BS + n


EN_OLD = (
    "The right panel plots $" + B('ln') + " T$ against $1/D$: the Gaussian data fall on a\n"
    "straight line of fitted slope $0.260$, within $4" + BS + "%$ of the predicted barrier\n"
    "height $" + B('Delta') + " U=0.25$, whereas the " + B('Levy') + BS + " data (fitted slope $0.271$) bend\n"
    "systematically " + B('emph') + "{below} the line over the low-$D$ range---the escape is no\n"
    "longer exponentially suppressed because large jumps cross the barrier\n"
    "directly."
)

EN_NEW = (
    "The right panel plots $" + B('ln') + " T$ against $1/D$. The Gaussian data fall on a\n"
    "straight line of fitted slope $0.260$, within $4" + BS + "%$ of the predicted barrier\n"
    "height $" + B('Delta') + " U=0.25$; the Gaussian curve lies slightly " + B('emph') + "{below} the\n"
    "Kramers line, the offset being the prefactor that the Kramers formula\n"
    "neglects. The " + B('Levy') + BS + " data behave qualitatively differently: they lie\n"
    "" + B('emph') + "{above} the Kramers line at small $1/D$ and then saturate, so that\n"
    "$" + B('ln') + " T$ grows far more slowly with $1/D$ than the exponential\n"
    "$e^{" + B('Delta') + " U/D}$ of the activated theory---large jumps carry the state over the\n"
    "barrier directly. The three " + B('Levy') + BS + " points at the smallest $D$ are\n"
    "right-censored at the simulation horizon $T_{" + B('max') + "}=3000$."
)

ZH_OLD = (
    "右图画出 $" + B('ln') + " T$ 对 $1/D$：高斯数据落在一条斜率拟合值为 $0.260$ 的直线上，\n"
    "与预言的势垒高度 $" + B('Delta') + " U=0.25$ 相差仅 $4" + BS + "%$；而 " + B('Levy') + BS + " 数据（拟合斜率 $0.271$）\n"
    "在低 $D$ 区系统性地" + B('emph') + "{弯到直线下方}——因为大跳跃直接跨越势垒，逃逸不再被指数压制。"
)

ZH_NEW = (
    "右图画出 $" + B('ln') + " T$ 对 $1/D$。高斯数据落在一条斜率拟合值为 $0.260$ 的直线上，\n"
    "与预言的势垒高度 $" + B('Delta') + " U=0.25$ 相差仅 $4" + BS + "%$；高斯曲线整体略" + B('emph') + "{低于} Kramers\n"
    "直线，这一偏移来自 Kramers 公式所忽略的预因子。" + B('Levy') + BS + " 数据则性质不同：在小的 $1/D$\n"
    "处它位于 Kramers 直线" + B('emph') + "{之上}，随后趋于饱和，故 $" + B('ln') + " T$ 随 $1/D$ 的增长\n"
    "远慢于激活理论给出的指数 $" + B('Delta') + " U=0.25$ 律——因为大跳跃可直接把状态带过势垒。"
    "最小的 $D$ 处三个 " + B('Levy') + BS + " 点被右删失在仿真时域 $T_{" + B('max') + "}=3000$。"
)

JOBS = [('en_mathpr', EN_OLD, EN_NEW), ('en_cssi', EN_OLD, EN_NEW),
        ('zh_mathpr', ZH_OLD, ZH_NEW), ('zh_cssi', ZH_OLD, ZH_NEW)]

for name, old, new in JOBS:
    path = os.path.join(P, name, 'main.tex')
    t = io.open(path, encoding='utf-8').read()
    assert old in t, 'anchor not found: ' + name
    t = t.replace(old, new)
    io.open(path, 'w', encoding='utf-8').write(t)
    print('%-11s fig11 text corrected' % name)

BAD = [chr(7), chr(9), chr(13)]
for name, *_ in JOBS:
    t = io.open(os.path.join(P, name, 'main.tex'), encoding='utf-8').read()
    hits = {hex(ord(c)): t.count(c) for c in BAD if c in t}
    print('%-11s control chars: %s' % (name, hits if hits else 'none'))
