# TONAL-GUIDANCE.md

This file captures writing tone preferences inferred from Post 0 edits between playground and the blog repo. It is a living guide and will be expanded over time.

## Current tonal baseline

- Write in technical R&D voice: clear, precise, evidence-oriented.
- Keep prose compact but not terse; prefer short paragraphs with high information density.
- Avoid hype, marketing language, and grand universal claims.
- State scope limits explicitly (what claim is and is not).
- Favor practical framing over philosophical framing.

## Preferred style patterns

- Lead with constraint framing (compute budget, iteration speed, reproducibility).
- Use concrete language for mechanisms:
  - good: "gradient imbalance", "control flow opacity", "time-to-quality"
  - avoid: vague adjectives like "better", "stronger" without metric context
- Emphasize tradeoffs, not absolutes.
- Treat negative results as useful outcomes.

## Sentence-level preferences

- Prefer direct declarative sentences.
- Avoid rhetorical questions as primary framing.
- Avoid filler and redundancy.
- Keep terminology consistent within a post (for example, always "tiny language model (TLM)").

## Structural preferences

- Keep section headers explicit and functional.
- Use numbered questions when setting research agenda.
- Use bullets for method and evaluation dimensions.
- End introductions with a concrete "starting point" section.

## Citation and rigor preferences

- Use numbered citation style (`[1]`, `[2]`, ...).
- Keep references explicit and scoped to claims.
- Distinguish empirical claim, mechanism claim, and speculation.

## Do-not rules (current)

- Do not use markdown horizontal rules (`---`) in post body content.
- Do not present constrained findings as universal laws.

## Workflow note

- Draft and iterate in playground first.
- Only copy into blog repo when explicitly requested.

## Example transformations from this edit

Use these as concrete style targets.

1) Scope framing

- Previous sentence:
  - "This series starts from the opposite direction. How small can you go, and still have a useful language model?"
- Edited sentence:
  - "This series starts from the opposite constraint."
- Why this is preferred:
  - Removes rhetorical framing and states the setup directly.

2) Explicit non-universal claim

- Previous sentence:
  - "Training dynamics, performance bottlenecks, and best practices at tiny scale do not necessarily transfer unchanged to larger models [FREF]."
- Edited sentence:
  - "This is not a universal taxonomy claim; it is a practical boundary for this series."
- Why this is preferred:
  - Avoids placeholder references and states scope cleanly.

3) Mechanism-first language

- Previous sentence:
  - "Nanochat sits in the right middle ground for this series: compact enough to understand end-to-end, modern enough to be relevant, and simple enough to modify."
- Edited sentence:
  - "Nanochat is an appropriate experimental substrate for this project for three reasons."
- Why this is preferred:
  - More formal technical register and stronger transition into structured argument.

4) Stronger rigor framing

- Previous sentence:
  - "Negative results are part of that evidence."
- Edited sentence:
  - "Negative results are first-class outcomes."
- Why this is preferred:
  - Clearer epistemic stance and stronger research tone.

5) Sharper ending

- Previous sentence:
  - "Post 1 begins with normalization placement: a small architectural change with potentially large effects on training dynamics."
- Edited sentence:
  - "Post 1 begins with a foundational transformer decision: normalization placement."
  - "At code level, the difference is small. At training-dynamics level, the consequences can be large."
  - "That asymmetry -- small structural edits, large behavioral effects -- is exactly what this series is designed to study."
- Why this is preferred:
  - Ends with explicit motivation and carries a clearer conceptual hook.

## Write-to-ship rule (avoid being edited)

When drafting, optimize for minimal post-editing by applying this checklist before handing off:

- Replace rhetorical questions with declarative scope statements.
- Remove placeholders (`TODO`, `FREF`, "add citation later").
- Make one claim per sentence and tie major claims to evidence or citations.
- Prefer concrete technical nouns (constraint, baseline, mechanism, tradeoff) over vague adjectives.
- End sections with a clear takeaway sentence.
- Keep terminology stable throughout the post.

If a sentence is likely to be rewritten for tone, rewrite it now.
