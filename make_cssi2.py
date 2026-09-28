# -*- coding: utf-8 -*-
"""为 cs.SI 版补充: (1) 新参考文献 (2) 平均场/相变理论小节 (3) 治理讨论小节."""
import re, io

P = r"D:\Code Test\Stochastic_Resonance_Enhanced\papers\en_cssi\main.tex"
s = io.open(P, encoding="utf-8").read()
orig = s

# ---------------- (1) 参考文献 ----------------
BIB_NEW = {
 "goel": r"""\bibitem[Goel et al.(2012)]{goel2012}
S.~Goel, D.~J.~Watts, and D.~G.~Goldstein.
\newblock The structure of online diffusion networks.
\newblock In \emph{Proc.~13th ACM Conf.~Electronic Commerce}, 623--638, 2012.

""",
 "granovetter": r"""\bibitem[Granovetter(1978)]{granovetter1978}
M.~Granovetter.
\newblock Threshold models of collective behavior.
\newblock \emph{Am.~J.~Sociol.}, 83:1420--1443, 1978.

""",
 "kempe": r"""\bibitem[Kempe et al.(2003)]{kempe2003}
D.~Kempe, J.~Kleinberg, and \'E.~Tardos.
\newblock Maximizing the spread of influence through a social network.
\newblock In \emph{Proc.~9th ACM SIGKDD}, 137--146, 2003.

""",
 "leskovec": r"""\bibitem[Leskovec et al.(2007)]{leskovec2007}
J.~Leskovec \emph{et al.}
\newblock The dynamics of viral marketing.
\newblock \emph{ACM Trans.~Web}, 1(1):5, 2007.

""",
 "watts": r"""\bibitem[Watts(2002)]{watts2002}
D.~J.~Watts.
\newblock A simple model of global cascades on random networks.
\newblock \emph{Proc.~Natl.~Acad.~Sci.~USA}, 99:5766--5771, 2002.

""",
}
for key, txt in BIB_NEW.items():
    if "\\bibitem" in txt and ("{%s}" % (list(re.findall(r"\{(\w+?)20", txt) or ["x"])[0])) not in s:
        pass
# 按字母位置插入
s = s.replace(r"\bibitem[Galam(2002)]{galam2002}",
              BIB_NEW["goel"] + BIB_NEW["granovetter"] + r"\bibitem[Galam(2002)]{galam2002}")
s = s.replace(r"\bibitem[Kosko(2009)]{kosko2009}",
              BIB_NEW["kempe"] + r"\bibitem[Kosko(2009)]{kosko2009}")
s = s.replace(r"\bibitem[Liggett(1985)]{liggett1985}",
              BIB_NEW["leskovec"] + r"\bibitem[Liggett(1985)]{liggett1985}")
s = s.replace(r"\bibitem[Wellens et al.(2004)]{wellens2004}",
              BIB_NEW["watts"] + r"\bibitem[Wellens et al.(2004)]{wellens2004}")
print("biblio inserted:", s != orig)

# ---------------- (2) 平均场 / 相变 理论小节 ----------------
MEANFIELD = r"""\subsection{Mean-field reduction, moment closure and the order parameter}
\label{sec:theory-mf}
The lattice coupling makes an exact solution infeasible, but a
\emph{mean-field} reduction makes the collective physics transparent and
exposes an order--disorder balance that is invisible in the single-particle
picture. Let
\begin{equation}
  \bar J:=\sum_j J_{ij},\qquad
  m(t):=\E[x_i(t)],\qquad
  V(t):=\E[x_i^2(t)]-m^2(t),
  \label{eq:mf-defs}
\end{equation}
denote, respectively, the total coupling weight, the \emph{order parameter}
(mean opinion, measuring consensus) and the fluctuation (measuring
disagreement). Approximating the coupling term by
$\sum_j J_{ij}(x_j-x_i)\approx \bar J\,(m-x_i)$ and closing the moment
hierarchy with a Gaussian closure---$\E[x^3]=m^3+3mV$ and
$\E[x^4]=m^4+6m^2V+3V^2$---the dynamics of the moments becomes
\begin{align}
  \dot m &= a m-b\bigl(m^{3}+3mV\bigr)+A\cos(\omega t)+C,
  \label{eq:mf-m}\\
  \dot V &= 2\Bigl[a(m^{2}+V)-b\bigl(m^{4}+6m^{2}V+3V^{2}\bigr)-\bar J V\Bigr]
           \nonumber\\
         &\quad +2\Bigl[A\cos(\omega t)+C\Bigr]m+2D .
  \label{eq:mf-V}
\end{align}
Several structural facts follow immediately and are worth stating because they
govern the interpretation of the simulations.

\paragraph{Order parameter and critical fluctuation.}
Setting $A=C=0$ and looking for a stationary ordered state ($m\neq0$,
$\dot m=0$), equation~\eqref{eq:mf-m} gives
\begin{equation}
  m^{2}=\frac{a}{b}-3V .
  \label{eq:mf-order}
\end{equation}
Thus the order parameter is \emph{suppressed by fluctuations}: as the noise
raises $V$, the consensus $|m|$ decreases, and it vanishes at the critical
fluctuation
\begin{equation}
  V_c=\frac{a}{3b}\qquad (a=b=1\Rightarrow V_c=1/3).
  \label{eq:mf-critical}
\end{equation}
For $V>V_c$ the ordered solution ceases to exist and the symmetric
disordered state $m=0$ is the only stationary one. This is the mean-field
order--disorder transition of the model, and it is the analogue of the
Curie--Weiss transition for social conformism.

\paragraph{Role of the coupling.}
Equation~\eqref{eq:mf-V} shows that the coupling enters the variance equation
as $-\bar J V$: social influence \emph{suppresses} disagreement and therefore,
through \eqref{eq:mf-order}, \emph{promotes} consensus. Equivalently, strong
coupling stabilises the ordered branch against noise. This is the mechanism
behind the coherent domains seen in the snapshots of
Figure~\ref{fig:snap}: the coupling keeps $V$ low enough that
$V<V_c$ and a large $|m|$ is sustainable.

\paragraph{Connection to the resonance.}
The mean-field picture clarifies why the resonance must sit at
\emph{intermediate} noise. At small $D$, $V$ is small and $m$ is locked at
$m\approx\pm\sqrt{a/b}$: the population is ordered but \emph{frozen}, unable
to follow the weak drive, so the coherent (driven) part of the response is
negligible. At large $D$, $V$ exceeds $V_c$, the ordered state is destroyed,
and $m\to0$: the population is disordered and again carries no coherent
signal. Only at intermediate $V$---neither frozen nor disordered---is the
population both ordered enough to respond collectively and mobile enough to
switch, which is precisely the resonance condition. The Kramers analysis of
the next subsection renders this balance quantitatively.

\paragraph{Remark on closure accuracy.}
The Gaussian closure is exact for $\alpha=2$ only in the linear (single-well)
limit; for heavy-tailed noise the true increment distribution has power-law
tails and the closure underestimates the probability of large excursions.
We therefore use \eqref{eq:mf-m}--\eqref{eq:mf-critical} for qualitative
interpretation and rely on the direct simulation for quantitative claims.

"""

anchor_mf = r"\subsection{Kramers escape rate and the analytic location of the peak}"
assert anchor_mf in s, "mean-field anchor not found"
s = s.replace(anchor_mf, MEANFIELD + anchor_mf, 1)
print("meanfield inserted")

# ---------------- (3) 治理讨论小节 ----------------
GOVERNANCE = r"""\subsection{Implications for platform governance and empirical calibration}
The results suggest three concrete, and to some degree counter-intuitive,
implications for the governance of information ecosystems.

\paragraph{(i) Synchronisation is a risk indicator, not a health indicator.}
Platforms and observers routinely interpret high alignment---trending
uniformity, rapid collective attention shifts---as evidence of a functioning
public sphere. The vulnerability window of Figure~\ref{fig:vuln} inverts this
reading: the noise level that maximises the population's coherent response is
\emph{also} the level at which false beliefs are most efficiently organised.
A practical consequence is that measures of collective coherence (the
fluctuation of aggregate sentiment, the synchrony of sharing behaviour) could
be monitored as early-warning indicators of susceptibility, much as
critical slowing down is used as an early-warning signal near ecological and
climatic tipping points.

\paragraph{(ii) Intervene where the system is most excitable.}
The debunking heat map (Figure~\ref{fig:heat}) shows that the marginal
effect of a truth bias is largest inside the vulnerability window. Because
beliefs there are most ``organisable'', a given fact-checking effort buys more
reduction in prevalence than the same effort applied in a quiescent or a
chaotic regime. This argues for \emph{adaptive} allocation of moderation and
fact-checking resources---concentrating effort during periods of high
collective excitability---rather than a constant, untargeted budget.

\paragraph{(iii) Target the wavefront, not only the source.}
The spatial wavefronts of Figure~\ref{fig:mis-wf} show false belief spreading
as a coherent front emanating from a focal seed, rather than as independent
conversions. Interventions that remove only the originating account or
``fact-check the source'' therefore address a vanishing fraction of the
dynamics; containing a resonant cascade requires acting on the neighbourhood
into which the front is advancing.

\paragraph{Pathway to empirical calibration.}
The model is deliberately a mechanistic idealisation, but its parameters
admit empirical counterparts: the stability index $\alpha$ can be estimated
from the tail exponent of cascade-size or inter-event-time distributions
\citep{barabasi2005,goel2012}; the noise intensity $D$ from the overall
activity volume; the coupling $J_0,\sigma$ from the interaction range revealed
by retweet/mention networks; and the truth bias $C$ from the measured reach of
correction messages relative to the original claim. Estimating these from
platform data---rather than assuming them---would convert the present
framework from a thought experiment into a calibrated instrument, and would
allow the prediction $D^{*}\approx\Delta U/2$ to be tested against observed
periods of heightened susceptibility. We regard this as the most valuable
next step.

"""

anchor_gov = r"\subsection{Limitations}"
assert anchor_gov in s, "governance anchor not found"
s = s.replace(anchor_gov, GOVERNANCE + anchor_gov, 1)
print("governance inserted")

io.open(P, "w", encoding="utf-8").write(s)
print("final length:", len(s))
