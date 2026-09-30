# V15 — Research Lab

V15 is an experimental research layer. It does not replace frozen versions.

## Purpose

The objective is to learn which signals survive repeated out-of-sample tests, not to maximize a single historical score.

The lab adds:

- exact Lotomania hypergeometric baseline;
- independent random simulation baseline;
- bootstrap confidence intervals;
- rolling stability windows;
- walk-forward result series;
- single-feature ablation;
- grouped ablation;
- leave-one-feature-out analysis;
- explicit temporal train/validation/holdout separation.

## Promotion rule

A candidate should not be promoted because of one strong window. Promotion requires consistent improvement over the random baseline and the frozen control across multiple independent temporal blocks.

The final holdout must remain untouched during feature and weight selection.
