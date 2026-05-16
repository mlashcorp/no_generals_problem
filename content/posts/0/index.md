---
title: "Tiny Language Models (TLM) Series"
date: 2026-05-16T21:30:00Z
draft: false
description: "Series introduction: tiny language models, experimental method, and nanochat baseline."
---

Most public discussion about language models is framed by scale: more parameters, more tokens, more GPUs, and larger pretraining budgets.

This series starts from the opposite constraint.

The goal is to study transformer architecture decisions in a regime where every choice is expensive, every instability is visible, and iteration speed matters as much as final quality. The target object is what I will call a **tiny language model** (TLM): operationally, a model that can be trained in roughly one week on a single RTX 5090.

This is not a universal taxonomy claim; it is a practical boundary for this series. The boundary is useful because it aligns model design with realistic solo or small-team iteration cycles: short enough to run controlled comparisons, large enough for architecture effects to be measurable.

Two questions motivate everything that follows:

1. How do transformer architecture choices affect model behavior under constrained compute?
2. How much useful performance can be extracted from small models before additional scale becomes the dominant lever?

To investigate those questions, the series needs a baseline that is modern, inspectable, and easy to perturb in controlled ways.

That baseline is **nanochat** ([GitHub](https://github.com/karpathy/nanochat)) [1].

## Why nanochat is a suitable baseline

Nanochat is an appropriate experimental substrate for this project for three reasons.

First, it is compact enough to be understood end-to-end without heavy framework indirection. For architecture research, this matters: if the control flow is opaque, attribution becomes weak.

Second, it is modern enough to be relevant. The baseline already reflects contemporary transformer design instincts, so experiments begin from a realistic point rather than from a purely pedagogical toy setup.

Third, it is tractable on commodity high-end hardware. The one-week / single-5090 constraint is not just narrative; it is the mechanism that keeps the series grounded in reproducible engineering.

Taken together, these properties make nanochat a strong middle baseline: not toy, not hyperscale.

## What this series is and is not

This series is closer to an engineering lab notebook than to a model launch report.

It is **not** designed to produce universal architecture laws from a single setup. External validity in language modeling depends on scale, data distribution, tokenizer behavior, optimizer dynamics, and evaluation protocol. Instead, the objective is cumulative evidence under fixed constraints: one decision at a time, one controlled comparison at a time, one explicit tradeoff at a time.

Negative results are first-class outcomes. In constrained regimes, "this idea fails under these conditions" is often more actionable than marginal gains in one run.

## Experimental method

Each post follows a consistent pattern:

- select one architecture decision (or tightly coupled decision pair)
- define baseline and variant(s) explicitly
- keep non-target variables fixed as much as possible (seed, data path, budget, schedule)
- report both outcome metrics and mechanism-facing diagnostics
- document limitations and plausible confounders
- provide reproducibility details where artifacts exist

Without methodological consistency across posts, comparisons become noisy and cumulative learning breaks down.

## What "performance" means in this series

Performance is multidimensional. Treating it as a single score hides the tradeoffs that matter in practice.

Depending on the question, posts will evaluate combinations of:

- quality metrics (for example BPB/perplexity)
- training stability (gradient behavior, spike frequency, divergence/collapse patterns)
- time-to-quality (how quickly useful quality is reached)
- resource efficiency (throughput, memory pressure, and compute overhead)

A choice that improves terminal quality but worsens stability may still be useful if the tradeoff is explicit. The reverse is also true.

## Reproducibility expectations

Where feasible, each post will include enough implementation detail for independent reproduction or close replication: configuration, run conditions, script paths, and outputs/figures derived from recorded metrics.

The standard is practical: not byte-identical determinism across all environments, but reproducible behavior sufficient to evaluate the claim.

## Starting point

Post 1 begins with a foundational transformer decision: normalization placement.

At code level, the difference is small. At training-dynamics level, the consequences can be large.

That asymmetry -- small structural edits, large behavioral effects -- is exactly what this series is designed to study.

## References

[1] Karpathy, A. *nanochat* (GitHub repository). https://github.com/karpathy/nanochat
