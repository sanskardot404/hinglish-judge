# hinglish-judge

**Can you trust an AI judge on Hinglish?**

LLM-as-a-judge is now the default way to evaluate AI systems, but almost all the evidence that judges are reliable comes from English. This project measures how well LLM judges agree with humans, and how biased they are, when the answers are in English, Hindi (Devanagari) and Hinglish (Roman-script, code-mixed). It also tests whether a free local open model can judge as well as a bigger hosted model.

> Status: work in progress. Week 1.

## What's measured
- Agreement with human labels (Cohen's kappa, bootstrap 95% CIs)
- Position bias (does the verdict flip when answers A and B are swapped?)
- Length bias (does the judge prefer longer answers?)
- Confidence on wrong verdicts

## Data
- Base: [`lmsys/mt_bench_human_judgments`](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- Extension: Hindi and Hinglish translations of a subset, plus my own human labels (released with the results).

## Results
_Coming soon._

## Run it
_Setup instructions will be added once the package is in place._

## `judge-check`
The reusable part of this repo will ship as `judge-check`, a small tool that produces a bias report for any LLM judge.
