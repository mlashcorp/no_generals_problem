---
title: "Tiny Language Models (TLM) Series"
date: 2026-05-16T23:59:00Z
draft: false
description: "Series introduction: tiny language models, experimental method, and nanochat baseline."
---

Most public discussion about language models is framed by scale: more parameters, more tokens, more GPUs.

This series starts from the opposite direction. How small can you go, and still have a useful language model?

The goal is to study transformer architecture choices at a scale where experiments are cheap enough to repeat, but large enough for design decisions to matter. The target object is what I will call a tiny language model (TLM): operationally, a model that can be trained in roughly one week on a single RTX 5090.

This scale is intentionally constrained. Training dynamics, performance bottlenecks, and best practices at tiny scale do not necessarily transfer unchanged to larger models [[2]](#ref-2). That limitation is part of the point: the series studies what happens under this compute budget, not what must hold universally at every scale.

This design constraint is useful because it aligns model design with realistic solo or small-team iteration cycles: short enough to run controlled comparisons, large enough for architecture effects to be measurable.

Two questions motivate everything that follows:

1. How do transformer architecture choices affect model behavior under constrained compute?
2. How much useful performance can be extracted from small models before additional scale becomes the dominant lever?

To make those questions testable, the series needs a baseline that is modern, readable, and easy to modify.

That baseline is **nanochat** ([GitHub](https://github.com/karpathy/nanochat)) [[1]](#ref-1).

## Why nanochat is a suitable baseline

Nanochat sits in the right middle ground for this series: compact enough to understand end-to-end, modern enough to be relevant, and simple enough to modify.

The compactness matters because architectural experiments require attribution. If the training stack is buried under layers of framework abstraction, it becomes harder to tell whether a result comes from the intended model change or from surrounding machinery.

Nanochat also starts from a realistic transformer baseline rather than a pedagogic toy model, so it preserves enough real-world behavior for meaningful analysis. That makes the experiments more informative while keeping the system small enough for controlled iteration on commodity high-end hardware.


## Methodology

Each post follows a consistent pattern:

- select one architecture decision (or tightly coupled decision pair)
- define baseline and variant(s) explicitly
- keep non-target variables fixed as much as possible (seed, data path, budget, schedule)
- report both final metrics and training signals that explain the result
- document limitations and potential confounders
- provide reproducibility details where artifacts exist

Methodological consistency across posts keeps comparisons clean and enables cumulative learning.

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

Post 1 begins with normalization placement: a small architectural change with potentially large effects on training dynamics.

## References

- <span id="ref-1">[1]</span> Karpathy, A. *nanochat* (GitHub repository). https://github.com/karpathy/nanochat
- <span id="ref-2">[2]</span> Hoffmann, J., et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
