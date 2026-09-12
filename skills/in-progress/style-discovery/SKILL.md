---
name: style-discovery
description: Discover empirical writing-style differences from paired drafts and final texts using function words, character n-grams, POS n-grams, dependency structure, Log Ratio, log-likelihood, dispersion, and concordances. Use when hand-built style rules or AI-tell lists should be tested or expanded from actual writing data.
---

# Discover style from paired writing

Use this skill to find candidate style patterns. Do not use it as a generic quality score, authorship detector, or automatic rule generator.

## Start from matched texts

Prefer pairs that express the same communicative task: an LLM draft and the final sent version of that message. This controls topic, recipient, dossier, and register far better than comparing unrelated corpora.

If only unmatched corpora are available, match genre and communicative setting before interpreting any result. Never compare fiscal email against general Dutch and call the resulting fiscal vocabulary "style".

Do not use TextTiling or topic segmentation for this job. Topic boundaries are not authorial-style evidence.

## Read existing rules, but do not obey them as ground truth

Before interpreting results, read the current writing-style skill and its references. Use them only to mark whether a finding has already been named. New corpus evidence may confirm, qualify, or contradict an old rule.

A literal baseline match is only a retrieval hint. It is not proof that the existing rule and the statistical feature mean the same thing.

## Run one evidence pipeline

Use `scripts/style_discovery.py` with side A as the LLM/source draft and side B as the human-reviewed final text.

The script extracts these feature families:

- function-word unigrams and short sequences;
- character 3-, 4-, and 5-grams, preserving punctuation and morphology;
- POS bi- and trigrams;
- dependency direction, dependency-length bins, and dependency-depth bins.

Sentence-length and general rhythm distributions stay outside this module when Etincel already measures them.

## Interpret three different quantities correctly

`Log Ratio` is effect size. Positive values mean the feature is relatively more frequent in side A; negative values mean more frequent in side B.

`G²` is log-likelihood evidence against equal relative frequency. Do not rank style importance from G² alone. Large corpora make tiny differences statistically conspicuous.

`dispersion` and `direction_consistency` test whether the difference recurs across pairs instead of coming from one long message or one dossier.

No single threshold promotes a feature to a style rule. Start broad, inspect the first run, then tighten thresholds only when the observed distribution justifies it.

## Read concordances before naming a pattern

For each strong feature, inspect examples from both sides. Character n-grams and POS/dependency features are discovery signals, not self-explanatory rules.

Translate evidence in this order:

`feature → contrast → effect size → G² → pair spread → concordance → linguistic interpretation → candidate rule`

A useful interpretation explains what construction the feature represents and what the writer repeatedly does instead.

Bad interpretation: `ADV CCONJ has G² 31.4`.

Good interpretation: `LLM drafts introduce explicit contrast with sentence-initial adverb + conjunction more often; final versions usually continue cumulatively or start a direct new main clause.`

## Promote only durable, inspectable findings

A candidate may enter a deterministic linter or style skill only when:

- enough absolute observations exist to inspect;
- the effect is materially large, not merely significant;
- the pattern occurs across multiple pairs;
- the direction repeats in a substantial share of informative pairs;
- concordances support one coherent linguistic interpretation;
- the rule improves held-out examples rather than merely explaining the training corpus.

Keep rejected or ambiguous findings in the report. Do not silently turn them into bans.

## Loop after every promoted rule

Re-run the paired corpus after adding a rule. Then test on held-out messages. A rule that removes a statistical tell but makes accepted writing worse is not a successful rule.

The endpoint is fewer unexplained edits by the human reviewer, not a higher style score.
