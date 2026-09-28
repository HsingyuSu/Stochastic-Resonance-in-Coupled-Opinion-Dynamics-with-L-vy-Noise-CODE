# -*- coding: utf-8 -*-
"""把 en_mathpr/main.tex 的骨架改写为 cs.SI 导向: 替换摘要与引言."""
import re, io

P = r"D:\Code Test\Stochastic_Resonance_Enhanced\papers\en_cssi\main.tex"
s = io.open(P, encoding="utf-8").read()

NEW_ABS = r"""\begin{abstract}
Information diffusion on social platforms is emphatically \emph{heavy-tailed}:
a tiny number of cascades reach enormous audiences while the overwhelming
majority die out, and empirical work has established that false content
propagates faster, deeper and more broadly than true content. This paper asks
whether the resulting public susceptibility to misinformation is governed by a
\emph{resonance} mechanism, and studies the question in a spatially coupled
bistable opinion model whose stochastic driving is an $\alpha$-stable
\Levy\ noise.

We combine three complementary analytical devices. (i) A semi-analytic
\emph{response theory}: a non-perturbative moment expansion of the
space-fractional Fokker--Planck equation, yielding the \emph{spectral
amplification factor} as an amplitude-normalised resonance index. (ii) An
\emph{escape-rate analysis}: the Kramers rate of the bistable potential gives
the parameter-free prediction $D^{*}=\Delta U/2$ for the optimal noise level,
which we test against simulation and against measured mean first-passage
times. (iii) An \emph{information-theoretic analysis}: the input--output
mutual information together with the forbidden-interval theorem establishes
aperiodic stochastic resonance for irregular---not merely periodic---streams
of public signals.

Simulating a $50\times50$ lattice ($N=2500$) with ensemble averaging, we find:
stochastic resonance for both Gaussian ($\alpha=2$) and heavy-tailed
($\alpha=1.5$) noise; a pronounced \emph{rightward shift and broadening} of the
resonance under heavy-tailed driving, which we trace to jump-mediated rather
than thermally activated barrier crossing; a \emph{misinformation
vulnerability window} in which the signal-to-noise ratio and the coherence of
collective belief switching peak at the same noise level; and strongly
non-linear returns to debunking, the marginal effect of a truth bias being
largest precisely inside that window.

We argue that these results carry a specific governance implication: a highly
synchronised, highly aligned public sphere is not self-evidently healthy, and
the resonance framework identifies both \emph{when} a population is most
vulnerable and \emph{where} intervention is most effective.

\medskip
\noindent\textbf{Keywords:} misinformation; opinion dynamics; social networks;
stochastic resonance; heavy-tailed dynamics; \Levy\ noise; Kramers escape;
mutual information; fact-checking; cascade dynamics.
\end{abstract}"""

NEW_INTRO = r"""\section{Introduction}
\label{sec:intro}

\subsection{The empirical problem: heavy-tailed, asymmetric information diffusion}
The diffusion of information through online social networks has become a
central object of study in network science, and the empirical picture that has
emerged over the last decade is both robust and uncomfortable. Content
popularity is heavy-tailed: the vast majority of cascades are tiny, while a
small number reach millions
\citep{leskovec2007,goel2012,barabasi2005}. Human communication itself is
bursty, characterised by long inactive periods punctuated by intense activity,
a pattern well described by heavy-tailed inter-event-time distributions
\citep{barabasi2005}. Most importantly, the diffusion of \emph{false} content
differs systematically from that of true content: analysing roughly 126\,000
cascades on Twitter, \citet{vosoughi2018} showed that falsehood diffused
significantly farther, faster, deeper and more broadly than the truth, with
the effect being most pronounced for political news. Related work documents
the role of confirmation bias and homophily in shaping which content is
believed and forwarded \citep{delvicario2016,cinelli2021}, and the
structural conditions---echo chambers, algorithmic amplification, bot
activity---that facilitate the spread of inaccurate information.

What is largely missing from this empirically rich literature is a
\emph{generative mechanism} explaining why a population should be
\emph{most} susceptible to false content under particular conditions. The
dominant modelling paradigms---epidemic-style cascade models
\citep{kempe2003,leskovec2007} and threshold/cascade models on
networks \citep{granovetter1978,watts2002}---describe \emph{how} a cascade
spreads once seeded, but typically treat the population's susceptibility as a
fixed parameter rather than as an emergent, noise-dependent property.

\subsection{Opinion dynamics and sociophysics: the mechanical lineage}
Parallel to the empirical work, a long tradition in
\emph{sociophysics} \citep{galam2008} and computational social science
\citep{castellano2009,liggett1985} has modelled opinion formation as an
emergent property of many locally interacting agents. The lineage includes
binary-choice Ising-type social models, the Sznajd ``united we convince''
model \citep{sznajd2000}, majority-rule models in which a committed minority
can overturn a majority \citep{galam2002}, and continuous-opinion
bounded-confidence models \citep{hegselmann2002,lorenz2007}. A recurring
theme is that the balance between local conformity and noise determines
whether a population polarises, fragments, or reaches consensus---and that
this balance can be studied with the tools of statistical mechanics.

Crucially, the standard social-physics models take noise to be
\emph{Gaussian} (or to enter simply as a temperature-like parameter). Given
the heavy-tailed nature of online activity documented above, this is a
significant idealisation: it misses the rare but enormous events---a viral
post, an influencer's statement, a coordinated amplification campaign---that
empirically dominate misinformation dynamics.

\subsection{Stochastic resonance: a mechanism for noise-dependent susceptibility}
\emph{Stochastic resonance} (SR) offers precisely the missing mechanism.
Introduced by \citet{benzi1981,benzi1982} to explain the periodicity of ice
ages and since observed across physics, chemistry and biology
\citep{gammaitoni1998,hanggi2002,wellens2004}, SR is the counter-intuitive
phenomenon whereby a nonlinear system's response to a weak, sub-threshold
input is \emph{optimised} by an intermediate amount of noise. In the canonical
bistable setting, too little noise leaves the system trapped in one state, too
much drowns the signal, and at an intermediate intensity the noise-induced
switching becomes synchronised with the weak drive.
If a population's opinion dynamics are bistable---as the epistemic
``true/false'' reading of the two wells suggests---then the population should
possess a characteristic noise level at which it is \emph{maximally
influenceable}, for true and false content alike.

Bringing SR into a social setting, however, requires handling two features
that the classical theory does not provide.
First, the noise must be allowed to be \emph{heavy-tailed}. Replacing Gaussian
noise by an $\alpha$-stable \Levy\ process replaces the ordinary
Fokker--Planck equation by a \emph{space-fractional} one
\citep{jespersen1999,chechkin2003}, in which large jumps transport the state
directly across the barrier; this is known to modify both escape statistics
and the resonance itself \citep{liu2018levy}.
Second, the classical signal-to-noise-ratio index presupposes a
\emph{periodic} input, whereas public discourse is irregular. The appropriate
index is then the \emph{input--output mutual information}, and the existence
of \emph{aperiodic} stochastic resonance can be established via the
forbidden-interval theorem \citep{patel2008,kosko2009,kang2020aperiodic}.

\subsection{Gaps and contributions}
To our knowledge, no existing study combines (i) a spatially coupled opinion
model, (ii) heavy-tailed \Levy\ driving, (iii) a rigorous resonance analysis
based on escape rates and spectral amplification, and (iv) an
information-theoretic treatment of aperiodic, irregular public signals, in a
single framework with an explicit misinformation interpretation. This paper
does so. Our contributions are:
\begin{enumerate}[leftmargin=2em,itemsep=2pt]
  \item \textbf{A heavy-tailed coupled opinion model with epistemic semantics.}
        We equip the two wells of a bistable potential with an explicit
        reading---$x>0$ = holding a correct belief, $x<0$ = being misled---and
        interpret the \Levy\ component as viral, rare but extreme events. A
        truth-bias term $C$ models sustained fact-checking.
  \item \textbf{Semi-analytic response theory.} We derive the
        space-fractional Fokker--Planck equation and a non-perturbative moment
        expansion, and compute the spectral amplification factor as an
        amplitude-normalised resonance index.
  \item \textbf{Escape-rate prediction and test.} The Kramers rate yields a
        parameter-free prediction $D^{*}=\Delta U/2$ for the optimal noise
        level. We test it both against the lattice simulation and against
        directly measured mean first-passage times, and show that the
        heavy-tailed peak shift follows from jump-mediated (non-activated)
        barrier crossing.
  \item \textbf{Aperiodic resonance.} Using the mutual information and the
        forbidden-interval theorem we prove and numerically confirm that the
        population transmits \emph{irregular} sub-threshold signals optimally
        at intermediate noise.
  \item \textbf{Comprehensive numerics and a vulnerability window.} Eleven
        figures document the resonance, the spatial wavefronts, the
        vulnerability window, and the non-linear returns to debunking.
\end{enumerate}

The remainder of the paper is organised as follows. Section~\ref{sec:model}
introduces the model and its observables. Sections~\ref{sec:prelim}--\ref{sec:theory}
develop the theoretical apparatus. Section~\ref{sec:numerics} describes the
numerical scheme, Section~\ref{sec:results} presents results,
Section~\ref{sec:discussion} discusses implications for platform governance
and empirical calibration, and Section~\ref{sec:conclusion} concludes.

"""

s2 = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}",
            lambda m: NEW_ABS, s, flags=re.S)
s3 = re.sub(r"\\section\{Introduction\}.*?(% =+\s*\n\\section\{Model\})",
            lambda m: NEW_INTRO + m.group(1), s2, flags=re.S)
io.open(P, "w", encoding="utf-8").write(s3)
print("abstract replaced:", s2 != s)
print("intro replaced:", s3 != s2)
print("len before/after:", len(s), len(s3))
