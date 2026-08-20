---
layout: post
title:  "Handling the Unseen: Censored Distributions, Measure Theory, and Entropy"
description: "Part 1 of a series on censored data: why censoring breaks standard regression, how measure theory rescues it, and a closed-form entropy for the Censored Normal distribution."
type: card-dated
date:   2026-08-19 00:00:00 +0100
categories: post
author: Andrea Karlova
published: true
---

In machine learning, we usually assume that our data is observed exactly. However, real instruments have limits, and when those limits are reached, the data they return is censored. This is a structural feature of many scientific pipelines, not just a rare glitch. Censoring quietly breaks standard regression models, and addressing it properly requires a bit of measure theory.

This is the first post in a short series. Here, I focus on the motivation, the censored (Tobit) measure, and a closed-form entropy for the Censored Normal distribution. Later posts will build on that entropy to derive information-theoretic acquisition functions (Censored BALD and Censored PES) for active learning and Bayesian optimization. All of the material here is drawn from our paper, [COBALT](https://proceedings.mlr.press/v337/karlova26a.html), with code available in the [repository](https://github.com/AndreaKarlova/cobalt).

### Motivation: Censoring is structural, not anomalous

Most measurements come from instruments with bounded scales. In drug discovery, for example, a high-throughput binding assay cannot resolve affinities beyond its detection limit: it clips non-binders at a threshold $l$, so all we know about such a point is that its true value satisfies $y^\ast \ge l$. The measurement is not missing; it is present, but only as an inequality.

Faced with this, two common shortcuts both introduce bias:

- **Drop the censored points.** Training a regressor only on the fully-resolved values truncates the distribution. This discards the information that non-binders provide ($y^\ast \ge l$) and degrades uncertainty right at the decision boundary, where it is most important.
- **Clamp and pretend.** Treating each clipped value as if it were an exact observation and fitting a Gaussian likelihood (or equivalently, minimizing MSE) biases the fit regardless of model architecture; the loss is simply incorrect for those points.

The principled alternative is to model the mixed continuous-and-discrete nature of the data directly, using a Tobit likelihood.

### The Tobit likelihood

Let $f$ be the latent function we want to recover, and let each observation carry an indicator $c_i \in \{-1, 0, 1\}$ recording whether it is left-censored, uncensored, or right-censored. With per-point bounds $l_i, u_i$ and noise $\varepsilon \sim \mathcal{N}(0,\sigma^2)$, the observation process is

$$
y_i =
\begin{cases}
l_i, & f(\mathbf{x}_i)+\varepsilon \le l_i \quad (c_i=-1),\\
f(\mathbf{x}_i)+\varepsilon, & l_i < f(\mathbf{x}_i)+\varepsilon < u_i \quad (c_i=0),\\
u_i, & f(\mathbf{x}_i)+\varepsilon \ge u_i \quad (c_i=1).
\end{cases}
$$

If we write $\phi$ and $\Phi$ for the standard normal PDF and CDF, and $f_i = f(\mathbf{x}_i)$, the resulting likelihood is a continuous density in the interior and a *probit atom* at each bound:

$$
p(y_i \mid f_i, c_i) =
\begin{cases}
\Phi\!\left(\tfrac{y_i - f_i}{\sigma}\right), & c_i = -1,\\[4pt]
\tfrac{1}{\sigma}\,\phi\!\left(\tfrac{y_i - f_i}{\sigma}\right), & c_i = 0,\\[4pt]
1 - \Phi\!\left(\tfrac{y_i - f_i}{\sigma}\right), & c_i = 1.
\end{cases}
$$

This setup matches the physics of the instrument closely, but it creates a mathematical issue.

### A distribution that is neither continuous nor discrete

Fix a latent mean $\mu$ and censor to $[l, u]$. The observed variable then follows the Censored Normal distribution, which mixes a Gaussian density on the open window $(l,u)$ with two point masses at the boundaries:

$$
p(x;\mu,\sigma,l,u) \;=\; \underbrace{\Phi\!\left(\tfrac{l-\mu}{\sigma}\right)\delta_l(x)}_{\text{mass clipped to } l} \;+\; \underbrace{\phi(x;\mu,\sigma)\,\mathbb{I}_{(l,u)}(x)}_{\text{Gaussian interior}} \;+\; \underbrace{\left[1-\Phi\!\left(\tfrac{u-\mu}{\sigma}\right)\right]\delta_u(x)}_{\text{mass clipped to } u},
$$

where $\delta_z$ is a Dirac mass at $z$ and $\mathbb{I}_{(l,u)}$ indicates the window. Each atom's weight is exactly the Gaussian tail probability that is collapsed onto the boundary.

![The Censored Normal law: a Gaussian density on the window (l, u) with two boundary atoms whose masses are the collapsed tail probabilities, over the reference measure's domain.](/assets/posts/censored_gauss_wide.png)
*The Censored Normal distribution as a mixed measure. The window $(l,u)$ carries the interior mass $\Phi(\tfrac{u-\mu}{\sigma})-\Phi(\tfrac{l-\mu}{\sigma})$ (red), while the two arrows are the boundary atoms with weights $\Phi(\tfrac{l-\mu}{\sigma})$ and $1-\Phi(\tfrac{u-\mu}{\sigma})$ — the collapsed lower and upper tails. The red baseline marks the "measure domain," the support of the reference measure $\rho=\lambda+\delta_l+\delta_u$.*

The problem is that this law is not absolutely continuous with respect to the Lebesgue measure $\lambda$: no ordinary density can put positive probability on the single points $l$ and $u$. So the usual definitions of density, expectation, and entropy, all written as Lebesgue integrals, do not apply as they are.

### The measure-theoretic fix

The clean solution is the Lebesgue decomposition theorem: any measure splits uniquely into an absolutely continuous (diffuse) part and a singular part. Here, the diffuse part is the Gaussian interior, and the singular part is the two atoms. Instead of forcing everything onto $\lambda$, we use a reference measure that already contains the atoms,

$$
\rho \;=\; \lambda + \delta_l + \delta_u,
$$

and describe the Censored Normal through its Radon–Nikodym derivative $\tfrac{d\nu}{d\rho}$ with respect to $\rho$. With this reference, the law is absolutely continuous ($\nu \ll \rho$), and the density is exactly the mixed object above: it equals $\phi(x;\mu,\sigma)$ on $(l,u)$ and equals the tail masses $\Phi(\tfrac{l-\mu}{\sigma})$, $1-\Phi(\tfrac{u-\mu}{\sigma})$ at the atoms.

Entropy is then well defined as the $\rho$-integral

$$
H[\nu] \;=\; -\int_{\mathbb{R}} \frac{d\nu}{d\rho}\,\log\frac{d\nu}{d\rho}\;d\rho,
$$

which automatically sums a continuous (differential-entropy) contribution over $(l,u)$ and two discrete (Shannon) contributions at the atoms. (Note the sign convention: this is the standard $-\!\int p\log p$.)

### Warm-up: integrating against the mixed measure

Before entropy, the mean shows the mechanics in miniature. Integrating $x$ against $\rho$ splits into the interior integral plus the two atom evaluations:

$$
\mathbb{E}[Y] \;=\; \underbrace{\mu\big[\Phi(\zeta_u)-\Phi(\zeta_\ell)\big] - \sigma\big[\phi(\zeta_u)-\phi(\zeta_\ell)\big]}_{\text{truncated Gaussian interior}} \;+\; \underbrace{l\,\Phi(\zeta_\ell) + u\big[1-\Phi(\zeta_u)\big]}_{\text{Bernoulli-like atoms}},
$$

where $\zeta_x = \tfrac{x-\mu}{\sigma}$ are the standardized bounds. The atom terms are simply each boundary value weighted by its collapsed tail probability. Notice that the correction $-\sigma[\phi(\zeta_u)-\phi(\zeta_\ell)]$ vanishes when the bounds are symmetric about the mean ($\zeta_u=-\zeta_\ell$, so $\phi(\zeta_u)=\phi(\zeta_\ell)$): symmetric clipping leaves the mean at $\mu$. Keep this in mind; the analogous term in the entropy behaves differently.

### Closed-form entropy of the Censored Normal

Carrying the same decomposition through the entropy integral gives a fully closed-form result — no Monte Carlo needed. With $\zeta_x = \tfrac{x-\mu}{\sigma}$ and window mass $\Delta\Phi = \Phi(\zeta_u)-\Phi(\zeta_\ell)$:

$$
H[Y] \;=\; \log\!\big(\sqrt{2\pi e}\,\sigma\big)\,\Delta\Phi \;-\; \tfrac{1}{2}\big[\zeta_u\phi(\zeta_u)-\zeta_\ell\phi(\zeta_\ell)\big] \;-\; \Phi(\zeta_\ell)\log\Phi(\zeta_\ell) \;-\; \Phi(-\zeta_u)\log\Phi(-\zeta_u).
$$

The three pieces have clear interpretations:

- **Base entropy.** $\log(\sqrt{2\pi e}\,\sigma)\,\Delta\Phi$ is the ordinary Gaussian differential entropy, scaled down by $\Delta\Phi$, the probability mass that actually falls inside the observable window.
- **Boundary (second-moment) correction.** $-\tfrac{1}{2}\big[\zeta_u\phi(\zeta_u)-\zeta_\ell\phi(\zeta_\ell)\big]$ accounts for how clipping changes the spread of the interior; it is the truncated-variance correction to the differential-entropy term. Unlike the mean's correction above, this one does **not** vanish for symmetric bounds: there $\zeta_u\phi(\zeta_u)-\zeta_\ell\phi(\zeta_\ell)=2\tfrac{\Delta}{\sigma}\phi(\tfrac{\Delta}{\sigma})$, leaving $-\tfrac{\Delta}{\sigma}\phi(\tfrac{\Delta}{\sigma})$.
- **Atom entropy.** $-\Phi(\zeta_\ell)\log\Phi(\zeta_\ell)-\Phi(-\zeta_u)\log\Phi(-\zeta_u)$ is the Shannon entropy of the two collapsed tails, each treated as a discrete outcome with its tail probability.

Plotting the three terms as the censored fraction increases makes the trade-off clear:

![Decomposition of the Censored Normal entropy into base entropy, boundary correction, and atom entropy, as a function of the fraction of mass censored.](/assets/posts/censored_entropy_decomposition.png)
*The three terms of $H[Y]$ as more mass is censored (standard normal, symmetric bounds). The base entropy (blue) fades as the observable window shrinks; the atom entropy (green) grows as probability accumulates at the boundaries; the boundary correction (orange) stays negative. Their sum (pink) interpolates smoothly between the Gaussian entropy $\log\sqrt{2\pi e}\,\sigma$ with no censoring and $\log 2$ — two equally likely atoms — under full censoring.*

**Sanity check (uncensoring).** As $l\to-\infty$ and $u\to+\infty$ we have $\Delta\Phi\to 1$; both $\zeta\phi(\zeta)$ terms vanish; and since $t\log t\to 0$ as $t\to 1$, the atom terms vanish as well. What remains is

$$
\lim_{l\to-\infty,\,u\to+\infty} H[Y] \;=\; \log\!\big(\sqrt{2\pi e}\,\sigma\big),
$$

which is exactly the entropy of the underlying Gaussian — the formula smoothly reduces to the familiar case.

**Symmetric bounds.** For $l=\mu-\Delta,\ u=\mu+\Delta$ the expression becomes

$$
H[Y] \;=\; \log\!\big(\sqrt{2\pi e}\,\sigma\big)\big[\Phi(\tfrac{\Delta}{\sigma})-\Phi(-\tfrac{\Delta}{\sigma})\big] \;-\; \tfrac{\Delta}{\sigma}\phi(\tfrac{\Delta}{\sigma}) \;-\; \Phi(-\tfrac{\Delta}{\sigma})\log\Phi(-\tfrac{\Delta}{\sigma}) \;-\; \Phi(\tfrac{\Delta}{\sigma})\log\Phi(\tfrac{\Delta}{\sigma}),
$$

and taking $\Delta\to\infty$ recovers the Gaussian entropy again. This is a useful way to see how much entropy is removed by clipping as the window narrows.

### Why this matters

A closed-form, differentiable entropy is the key that enables information-theoretic acquisition under censoring. Once we can evaluate $H[Y]$ exactly, we can build analytic versions of BALD (Bayesian Active Learning by Disagreement) and Predictive Entropy Search that account for the atoms instead of being misled by them. The practical result: instead of models becoming overconfident at detection limits, they can deliberately probe those boundaries, enabling efficient active learning and Bayesian optimization that thrive on censored data instead of failing on it.

That construction — turning this entropy into acquisition functions with consistency and $D$-optimality guarantees — is the subject of the next post in the series.

### References

- **Paper.** A. Karlová, R. Kabra, D. A. de Souza, B. Paige. *COBALT: Censored Optimization and Bayesian Active Learning Techniques.* Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence (UAI), PMLR 337:2713–2743, 2026. [proceedings](https://proceedings.mlr.press/v337/karlova26a.html) · [PDF](https://raw.githubusercontent.com/mlresearch/v337/main/assets/karlova26a/karlova26a.pdf) · [OpenReview](https://openreview.net/forum?id=47OJlKgIgG)
- **Code.** [`github.com/AndreaKarlova/cobalt`](https://github.com/AndreaKarlova/cobalt)

<div class="bibtex-wrapper">
<button class="copy-btn" onclick="copyBibtex(this)">📋 Copy</button>
<pre><code id="bibtex-citation">@InProceedings{pmlr-v337-karlova26a,
  title     = { {COBALT}: Censored Optimization and {Bayesian} Active Learning Techniques},
  author    = {Karlova, Andrea and Kabra, Rishabh and de Souza, Daniel Augusto and Paige, Brooks},
  booktitle = {Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence},
  pages     = {2713--2743},
  year      = {2026},
  editor    = {Perkovi{\'c}, Emilija and Malinsky, Daniel},
  volume    = {337},
  series    = {Proceedings of Machine Learning Research},
  month     = {17--21 Aug},
  publisher = {PMLR},
  url       = {https://proceedings.mlr.press/v337/karlova26a.html}
}</code></pre>
</div>

<script>
function copyBibtex(btn) {
  const text = document.getElementById('bibtex-citation').textContent;
  navigator.clipboard.writeText(text).then(function() {
    btn.textContent = '✓ Copied!';
    btn.classList.add('copied');
    setTimeout(function() {
      btn.textContent = '📋 Copy';
      btn.classList.remove('copied');
    }, 2000);
  });
}
</script>