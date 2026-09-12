# Input modes

The default mode is `paired`. It is the only mode that compares two versions
of the same communicative task and computes Log Ratio, G², dispersion across
pairs, and direction consistency.

For one candidate corpus without a valid reference corpus, use the explicit
`descriptive` mode. It reports feature counts, within-family relative
frequencies, document dispersion, and syntax metrics when a syntax model is
available. It does not compute Log Ratio or G² and must not be interpreted as
keyness.

```bash
python skills/in-progress/style-discovery/scripts/style_discovery.py \
  /private/texts.json --mode descriptive --text-field text --text-id-field id \
  --no-syntax --out-dir /private/descriptive-output
```

The input records for descriptive mode contain one text per object:

```json
{"id":"mail-001","text":"One complete text..."}
```

An optional `provenance` object is carried through extraction. It is metadata,
not text normalization, and is summarized in descriptive output without
including the source text:

```json
{
  "id":"mail-001",
  "text":"One complete text...",
  "provenance": {
    "authorship_status":"VERIFIED_USER_AUTHORED",
    "source_kind":"curated_examples"
  }
}
```

The loader does not normalize the stored input. It trims only outer transport
whitespace; the derived feature extraction keeps punctuation and casing in
character n-grams and keeps morphology in the surface text.

## Paired corpus input

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

`--no-syntax` still extracts the sentence/regex `construct` family. It only
omits POS and dependency features. A Dutch sentencizer is sufficient for a
sentence-aware adapter; the bundled fallback uses transparent regex sentence
boundaries and therefore does not require spaCy.

For the full Dutch syntax layer, install spaCy plus a Dutch pipeline and select it with `--spacy-model` when it is not named `nl_core_news_sm`.
