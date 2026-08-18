---
layout: post
title:  "Handling the Unseen: A Guide to Censored Distributions, Measure Theory, and Entropy"
description: "A guide to understanding censored distributions and their entropy calculations."
type: card-dated
date:   2026-08-19 00:00:00 +0100
categories: post
author: Andrea Karlova
published: true
---
In real-world machine learning, we often assume our data is perfectly observed. But what happens when our measuring instruments hit their limits? Welcome to the world of censored data—a structural reality in many scientific pipelines that breaks standard regression models.

Here is a breakdown of why censoring happens, the mathematical headaches it causes, and how we can use measure theory to unlock closed-form entropy for better Bayesian active learning.

### Motivation: Censoring is Structural, Not Anomalous

Most real-world measurements come from bounded-scale instruments. For example, in drug discovery, high-throughput binding assays simply cannot detect values beyond a certain limit. They clip "non-binders" at a detection limit $l$, meaning we only know that the true affinity satisfies $y^* \ge l$.  

When faced with this, data scientists often try two flawed approaches:

- **Filtering the data**: Training a regressor only on the strictly observed values truncates the distribution, injects systematic bias, and degrades uncertainty exactly where it matters most—at the decision boundary.  
- **Clamping the data**: Applying a standard Gaussian likelihood (or Mean Squared Error) to clamped data creates bias regardless of the model architecture used.  

To properly handle this, we need to model the mixed continuous (regression) and discrete (classification) nature of the data using a Tobit likelihood.  

### The Censored Distribution (Tobit Likelihood)

The Tobit likelihood transitions smoothly between different states of observation. Let's say we have varying lower bounds $l_i$ and upper bounds $u_i$. The likelihood behaves differently depending on where the true latent function falls:  

- **Uncensored**: If the value is within the instrument's readable window, it follows a standard continuous Gaussian density.  
- **Censored (Left/Right)**: If the value hits the upper or lower bounds, the observation becomes a discrete point mass (a probit atom).  

While this brilliantly captures the physical reality of the instrument, it introduces a severe mathematical problem.

### The Measure-Theoretical Hiccup

In continuous probability, we usually rely on the Lebesgue measure to calculate things like entropy. However, the censored distribution's measure is not absolutely continuous with respect to Lebesgue.  

Because it mixes a continuous Gaussian density in the middle with discrete point masses at the boundaries, standard integrals break down. To fix this, we have to step into measure theory. We treat the distribution via the Radon–Nikodym derivative with respect to a new, mixed base measure: $\rho = \lambda + \delta_l + \delta_u$.  

By combining the Lebesgue measure ($\lambda$) with Dirac delta measures at the bounds ($\delta_l$ and $\delta_u$), the mathematics stabilize, and the entropy of this hybrid distribution becomes perfectly well-defined.  

### Deriving Closed-Form Entropy

Once the measure theory is sorted, we can do something highly useful: calculate the exact, closed-form entropy of the Censored Normal distribution.  

Rather than relying on computationally heavy Monte Carlo approximations, the exact entropy breaks down elegantly into three interpretable components:  

- **The Base Entropy**: The standard normal entropy, scaled by the probability mass that falls within the observable window.  
- **The Asymmetry Penalty**: A mathematical correction for the asymmetry introduced when the distribution is chopped off at one or both ends.  
- **The Atom Entropy**: The standard entropy of the two collapsed, discrete tails.  

### Why Does This Matter?

Having an analytic, closed-form entropy is the golden key for information-theoretic acquisition functions. It allows us to derive Analytic Censored BALD and Censored PES (Predictive Entropy Search).  

Instead of our models getting confused at the detection limits, they can actively and intelligently target these decision boundaries. This allows for highly computationally efficient Bayesian optimization and active learning that thrives—rather than fails—when faced with censored data.
