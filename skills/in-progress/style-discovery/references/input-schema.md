# Paired corpus input

Use JSONL when possible. One line is one communicative task with both versions of the text.

```json
{"pair_id":"mail-001","draft":"LLM concept...","final":"actually sent version..."}
{"pair_id":"mail-002","draft":"LLM concept...","final":"actually sent version..."}
```

The default fields are `pair_id`, `draft`, and `final`. Override them with `--id-field`, `--a-field`, and `--b-field`.

## Pairing rule

A pair must represent the same intended message. Do not pair texts merely because they share a topic. The strongest design is the literal draft that preceded the final sent mail.

If several LLM drafts preceded one final mail, either select the last draft before human editing or create a separate experiment for revision stages. Do not count one final message repeatedly in the same primary comparison.

## Compare message bodies, not mail-client artifacts

Before pairing, remove quoted reply history, automatic signatures, tracking footers, legal disclaimers, MIME/HTML residue, and other material that was not part of the writing decision under study. Apply the same normalization rule to both sides.

Do not aggressively clean punctuation, casing, spacing, or morphology. Character n-grams deliberately need those signals. Normalize only artifacts introduced by the transport or export layer.

## Corpus boundaries

Keep materially different genres separate when possible. External client email, internal colleague email, formal tax-authority correspondence, and marketing copy may have different style systems.

Run them separately first. Pool them only after checking that the same pattern points in the same direction within the component genres.

## Privacy

Keep raw mail, names, addresses, signatures, identifiers, and concordance outputs outside a public repository. This skill needs a local path to the corpus; it never requires the corpus itself to be committed.

If the output will leave the private machine, rerun with `--no-concordance` or redact the generated report first.

## Example command

```bash
python skills/in-progress/style-discovery/scripts/style_discovery.py \
  /private/pairs.jsonl \
  --a-field draft \
  --b-field final \
  --a-label "LLM draft" \
  --b-label "sent version" \
  --known /path/to/adnan-schrijfstijl/SKILL.md /path/to/adnan-schrijfstijl/references/taalpatronen.md \
  --out-dir /private/style-discovery-output
```

For a lexical smoke test without spaCy:

```bash
python skills/in-progress/style-discovery/scripts/style_discovery.py \
  /private/pairs.jsonl --no-syntax
```

For the full Dutch syntax layer, install spaCy plus a Dutch pipeline and select it with `--spacy-model` when it is not named `nl_core_news_sm`.
