# -*- coding: utf-8 -*-
"""从中文 math.PR 版派生中文 cs.SI 版。"""
import os, re, io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'papers', 'zh_mathpr', 'main.tex')
DST = os.path.join(ROOT, 'papers', 'zh_cssi', 'main.tex')
FRAG = os.path.join(ROOT, 'code')

def frag(name):
    with io.open(os.path.join(FRAG, name), encoding='utf-8') as f:
        return f.read()

tex = io.open(SRC, encoding='utf-8').read()
orig_len = len(tex)

def cut_between(s, start, end, replacement, tag):
    i = s.find(start)
    j = s.find(end, i + len(start))
    assert i >= 0 and j > i, 'anchor not found: ' + tag
    return s[:i] + replacement + s[j:], True

# ---- 1. 标题 ----
new_title = r'''\title{\bfseries 耦合意见动力学中的随机共振与误信息脆弱性\\[0.3em]
\large 面向社交网络的重尾 \Levy\ 噪声视角}
\author{苏星宇\thanks{独立学生工作，完成于西安交通大学数学与统计学院就读期间。}\\
\textit{西安交通大学，西安 710049，中国} \\
\texttt{suxingyu@stu.xjtu.edu.cn}}
\date{\today}
'''
tex, ok1 = cut_between(tex, '\\title{', '\\begin{document}', new_title, 'title')
print('title replaced:', ok1)

# ---- 2. 摘要 ----
tex, ok2 = cut_between(tex, '\\begin{abstract}', '\\end{abstract}',
                       '\\begin{abstract}\n' + frag('frag_zh_cssi_abstract.tex'),
                       'abstract')
print('abstract replaced:', ok2)

# ---- 3. 引言 ----
tex, ok3 = cut_between(tex, '\\section{引言}', '\\section{模型}', frag('frag_zh_cssi_intro.tex'), 'intro')
print('intro replaced:', ok3)

# ---- 4. 平均场小节：摘除旧版并插到 Kramers 之前 ----
mf_start = tex.find('\\subsection{平均场约化、矩闭包与序参量}')
assert mf_start > 0
mf_end = tex.find('\\subsection{', mf_start + 10)
old_mf = tex[mf_start:mf_end]
tex = tex[:mf_start] + tex[mf_end:]
kram_anchor = '\\subsection{Kramers 逃逸率与峰位的解析预测}'
assert kram_anchor in tex
tex = tex.replace(kram_anchor, frag('frag_zh_cssi_mf.tex') + kram_anchor)
print('meanfield moved & expanded:', len(old_mf))

# ---- 5. 治理小节插入到局限性之前 ----
gov_anchor = '\\subsection{局限性}'
assert gov_anchor in tex
tex = tex.replace(gov_anchor, frag('frag_zh_cssi_gov.tex') + gov_anchor)
print('governance inserted:', True)

# ---- 6. 结论 ----
tex, ok6 = cut_between(tex, '\\section{结论}', '\\begin{thebibliography}',
                       frag('frag_zh_cssi_concl.tex'), 'conclusion')
print('conclusion replaced:', ok6)

# ---- 7. 交叉引用标签修正 ----
tex = tex.replace('\\eqref{eq:mf-closure}', '\\eqref{eq:mf-order}')
tex = tex.replace('\\eqref{eq:vc}', '\\eqref{eq:mf-critical}')
tex = tex.replace('\\ref{sec:meanfield}', '\\ref{sec:theory-mf}')

# ---- 8. 补足 cs.SI 方向的新参考文献 ----
def insert_bib(tex, key, text, after_key):
    if key in tex:
        return tex
    anchor = '\\bibitem[%s]{%s}' % (after_key, after_key)
    assert anchor in tex, after_key
    i = tex.find(anchor)
    # 找到该 bibitem 条目的结尾（下一个 \bibitem 或 \end{thebibliography}）
    j = tex.find('\\bibitem[', i + 10)
    k = tex.find('\\end{thebibliography}')
    end = min(x for x in [j, k] if x > 0)
    return tex[:end] + text + tex[end:]

bibs = [
    ('granovetter1978',
     '\\bibitem[Granovetter(1978)]{granovetter1978}\n'
     'M.~Granovetter.\n'
     '\\newblock Threshold models of collective behavior.\n'
     '\\newblock \\emph{Amer.\\ J.\\ Sociol.}, 83:1420--1443, 1978.\n\n',
     'goel2012'),
    ('kempe2003',
     '\\bibitem[Kempe et al.(2003)]{kempe2003}\n'
     'D.~Kempe, J.~Kleinberg, and \\\'E.~Tardos.\n'
     '\\newblock Maximizing the spread of influence through a social network.\n'
     '\\newblock In \\emph{Proc.\\ 9th ACM SIGKDD}, pp.~137--146, 2003.\n\n',
     'leskovec2007'),
    ('watts2002',
     '\\bibitem[Watts(2002)]{watts2002}\n'
     'D.~J.~Watts.\n'
     '\\newblock A simple model of global cascades on random networks.\n'
     '\\newblock \\emph{Proc.\\ Natl.\\ Acad.\\ Sci.\\ USA}, 99:5766--5771, 2002.\n\n',
     'vosoughi2018'),
]
for key, text, after in bibs:
    tex = insert_bib(tex, key, text, after)
print('biblio ok')

os.makedirs(os.path.dirname(DST), exist_ok=True)
with io.open(DST, 'w', encoding='utf-8') as f:
    f.write(tex)
print('written:', DST, len(tex), '(orig %d)' % orig_len)
