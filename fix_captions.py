# -*- coding: utf-8 -*-
"""重写 fig8 图注（原先的反斜杠被 shell 转义吞掉，产生控制字符）。

用 BS = chr(92) 显式构造 LaTeX 反斜杠，彻底避开转义层。
"""
import io, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(BASE, 'papers')
BS = chr(92)


def cmd(name):
    return BS + name


EN_NEW = (
    "Misinformation vulnerability window. Solid lines (left axis):\n"
    "signal-to-noise ratio; dashed lines (right axis): belief-oscillation "
    "coherence\namplitude. The vertical dotted lines mark the noise intensity "
    "$D^{*}$ that\nmaximises the coherence amplitude, which coincides with the "
    "peak of the\nspectral amplification factor $" + cmd('eta') + "$ in Fig.~" +
    cmd('ref') + "{fig:eta}. For $" + cmd('alpha') + "=2$\nthe SNR maximum lies "
    "at the adjacent grid point; for $" + cmd('alpha') + "=1.5$ the SNR\n"
    "decreases monotonically once barrier crossings are activated, so its global\n"
    "maximum at $D" + cmd('to') + "0$ is the trivial no-flip limit and carries "
    "no resonance\ncontent.}"
)

ZH_NEW = (
    "误信息脆弱性窗口。实线（左轴）：信噪比；虚线（右轴）：信念振荡相干幅值。\n"
    "竖直虚线标出使相干幅值达到极大的噪声强度 $D^{*}$，它与图~" + cmd('ref') +
    "{fig:eta} 中谱放大因子\n$" + cmd('eta') + "$ 的峰位一致。对 $" + cmd('alpha') +
    "=2$，SNR 的极大值位于相邻网格点；对 $" + cmd('alpha') + "=1.5$，\n"
    "一旦势垒跨越被激活，SNR 便单调下降，其 $D" + cmd('to') + "0$ 处的全局极大只是"
    "\u201c不翻转\u201d的平凡极限，\n不含共振内容。}"
)

JOBS = [
    ('en_mathpr', 'Misinformation vulnerability window.', 'content.}', EN_NEW),
    ('en_cssi', 'Misinformation vulnerability window.', 'content.}', EN_NEW),
    ('zh_mathpr', '误信息脆弱性窗口。', '不含共振内容。}', ZH_NEW),
    ('zh_cssi', '误信息脆弱性窗口。', '不含共振内容。}', ZH_NEW),
]

for name, start, end, new in JOBS:
    path = os.path.join(P, name, 'main.tex')
    t = io.open(path, encoding='utf-8').read()
    i = t.find(start)
    assert i >= 0, 'start not found: ' + name
    j = t.find(end, i)
    assert j > i, 'end not found: ' + name
    j += len(end)
    t = t[:i] + new + t[j:]
    io.open(path, 'w', encoding='utf-8').write(t)
    print('%-11s caption rewritten (%d chars)' % (name, len(new)))

# 校验：确认不再有控制字符
BAD = [chr(7), chr(9), chr(13)]
for name, *_ in JOBS:
    t = io.open(os.path.join(P, name, 'main.tex'), encoding='utf-8').read()
    hits = {hex(ord(c)): t.count(c) for c in BAD if c in t}
    print('%-11s control chars: %s' % (name, hits if hits else 'none'))
