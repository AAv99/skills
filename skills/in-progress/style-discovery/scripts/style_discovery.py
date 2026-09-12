#!/usr/bin/env python3
"""Paired and descriptive corpus discovery for writing-style differences.

The engine finds candidate stylistic patterns. It does not decide that a pattern is
"good", "bad", "AI", or a durable style rule. Those interpretations require review
of the evidence packet and concordances.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FUNCTION_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}
DUTCH_FUNCTION_WORDS = {
    "aan", "als", "bij", "dan", "dat", "de", "der", "des", "die", "dit", "door",
    "een", "en", "er", "haar", "hem", "hen", "het", "hun", "ik", "in", "is", "je",
    "jij", "maar", "met", "mij", "mijn", "na", "naar", "niet", "of", "om", "onder",
    "ons", "onze", "ook", "op", "over", "te", "tegen", "tot", "uit", "u", "uw", "van",
    "voor", "was", "wat", "we", "wel", "werd", "wij", "wordt", "ze", "zij", "zijn", "zo",
    "zou", "zullen", "zonder"
}
TOKEN_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+(?:['’][\wÀ-ÖØ-öø-ÿ]+)?", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
SENTENCE_RE = re.compile(
    r"(?:[^.!?]|\.(?=\d))+(?:[.!?]+(?=\s|$)|$)", re.UNICODE
)

_CONSTRUCT_PATTERNS = {
    "condition_fronted": re.compile(
        r"^\s*(?:indien|mocht|lukt\b[^,]{0,100}|bent\s+u\s+akkoord)"
        r"\b[^.!?]{0,220},\s*(?:dan\s+)?\S",
        re.IGNORECASE,
    ),
    "purpose_zodat": re.compile(r"\bzodat\b", re.IGNORECASE),
    "paired_en_dash_aside": re.compile(r"–[^.!?–\n]{1,160}–"),
    "semicolon_link": re.compile(r";"),
    "negative_contrast": re.compile(
        r"(?:niet\s+[^.!?,\n]{1,120},\s*maar\s+[^.!?]{1,160}|"
        r"geen\s+[^.!?,\n]{1,120},\s*maar\s+[^.!?]{1,160}|"
        r"niet\s+zozeer\s+[^.!?]{1,120}\s+als\s+wel\s+[^.!?]{1,160})",
        re.IGNORECASE,
    ),
}
NEGATION_START_RE = re.compile(r"^(?:niet|geen)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Pair:
    pair_id: str
    a: str
    b: str


@dataclass(frozen=True)
class TextRecord:
    text_id: str
    text: str
    provenance: Mapping[str, Any] | None = None


@dataclass
class Extracted:
    counts: dict[str, Counter[str]]
    metrics: dict[str, float]


def load_nlp(model: str):
    try:
        import spacy  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required for POS/dependency features. Install spaCy and a Dutch model, "
            "or use --no-syntax for a lexical/construct-only run."
        ) from exc
    try:
        return spacy.load(model, disable=["ner"])
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model {model!r} is not installed. Install it or choose --no-syntax."
        ) from exc


def load_pairs(path: Path, a_field: str, b_field: str, id_field: str) -> list[Pair]:
    records = read_records(path, "pairs")
    pairs: list[Pair] = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {i} is not an object")
        a = str(record.get(a_field, "")).strip()
        b = str(record.get(b_field, "")).strip()
        if not a or not b:
            continue
        pair_id = str(record.get(id_field) or f"pair-{i + 1:04d}")
        pairs.append(Pair(pair_id=pair_id, a=a, b=b))
    if not pairs:
        raise ValueError("No complete pairs found")
    return pairs


def read_records(path: Path, object_key: str) -> list[Any]:
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        records = payload.get(object_key, [])
        return records if isinstance(records, list) else []
    return []


def load_texts(path: Path, text_field: str, id_field: str) -> list[TextRecord]:
    records = read_records(path, "texts")
    texts: list[TextRecord] = []
    for i, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Record {i} is not an object")
        text = str(record.get(text_field, "")).strip()
        if not text:
            continue
        text_id = str(record.get(id_field) or f"text-{i + 1:04d}")
        raw_provenance = record.get("provenance")
        if raw_provenance is not None and not isinstance(raw_provenance, Mapping):
            raise ValueError(f"Record {i} provenance must be an object")
        provenance = (
            dict(raw_provenance) if isinstance(raw_provenance, Mapping) else None
        )
        texts.append(TextRecord(text_id=text_id, text=text, provenance=provenance))
    if not texts:
        raise ValueError("No complete text records found")
    return texts


def surface_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.strip())


def normalize_text(text: str) -> str:
    return surface_text(text).casefold()


def simple_tokens(text: str) -> list[str]:
    return [m.group(0).casefold() for m in TOKEN_RE.finditer(text)]


def sentence_segments(text: str) -> list[str]:
    segments = [
        match.group(0).strip()
        for match in SENTENCE_RE.finditer(text)
        if match.group(0).strip()
    ]
    return segments or ([surface_text(text)] if surface_text(text) else [])


def ngrams(items: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    for i in range(len(items) - n + 1):
        yield tuple(items[i : i + n])


def bin_distance(value: int) -> str:
    if value <= 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3-4"
    if value <= 7:
        return "5-7"
    return "8+"


def bin_depth(value: int) -> str:
    if value <= 1:
        return str(value)
    if value == 2:
        return "2"
    if value == 3:
        return "3"
    return "4+"


def token_depth(token: Any) -> int:
    depth = 0
    seen: set[int] = set()
    current = token
    while current.head.i != current.i and current.i not in seen:
        seen.add(current.i)
        depth += 1
        current = current.head
        if depth > 100:
            break
    return depth


def add_fallback_function_words(counts: dict[str, Counter[str]], raw_tokens: Sequence[str]) -> None:
    counts["function_word"].update(t for t in raw_tokens if t in DUTCH_FUNCTION_WORDS)
    for n in (2, 3):
        family = f"function_word_{n}gram"
        for window in ngrams(raw_tokens, n):
            if all(token in DUTCH_FUNCTION_WORDS for token in window):
                counts[family][" ".join(window)] += 1


def add_construct_features(counts: dict[str, Counter[str]], text: str) -> None:
    """Extract sentence/regex constructions without POS or dependency parsing."""
    sentences = sentence_segments(text)
    for sentence in sentences:
        for feature, pattern in _CONSTRUCT_PATTERNS.items():
            hits = len(pattern.findall(sentence))
            if hits:
                counts["construct"][feature] += hits

    negative_listing_count = 0
    for first, second, third in zip(sentences, sentences[1:], sentences[2:]):
        if (
            NEGATION_START_RE.search(first)
            and NEGATION_START_RE.search(second)
            and not NEGATION_START_RE.search(third)
        ):
            negative_listing_count += 1
    if negative_listing_count:
        counts["construct"]["negative_listing"] += negative_listing_count


def extract(text: str, nlp: Any | None, include_syntax: bool = True) -> Extracted:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    normalized = surface_text(text)
    raw_tokens = simple_tokens(text)
    add_fallback_function_words(counts, raw_tokens)
    add_construct_features(counts, text)

    for n in (3, 4, 5):
        family = f"char_{n}gram"
        counts[family].update(
            normalized[i : i + n]
            for i in range(len(normalized) - n + 1)
            if normalized[i : i + n].strip()
        )

    metrics: dict[str, float] = {}
    if not include_syntax:
        return Extracted(dict(counts), metrics)
    if nlp is None:
        raise RuntimeError("Syntax was requested without an NLP pipeline")

    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    counts["function_word"] = Counter(
        t.text.casefold() for t in tokens if t.pos_ in FUNCTION_POS and not t.is_punct
    )

    for n in (2, 3):
        family = f"function_word_{n}gram"
        counts[family] = Counter()
        for i in range(len(tokens) - n + 1):
            window = tokens[i : i + n]
            if all(t.pos_ in FUNCTION_POS and not t.is_punct for t in window):
                counts[family][" ".join(t.text.casefold() for t in window)] += 1

    pos = [t.pos_ for t in tokens]
    counts["pos_2gram"].update(" ".join(g) for g in ngrams(pos, 2))
    counts["pos_3gram"].update(" ".join(g) for g in ngrams(pos, 3))

    directions: list[str] = []
    lengths: list[int] = []
    depths: list[int] = []
    for token in tokens:
        depth = token_depth(token)
        depths.append(depth)
        counts["dep_depth"][bin_depth(depth)] += 1
        if token.head.i == token.i:
            continue
        direction = "R" if token.i > token.head.i else "L"
        distance = abs(token.i - token.head.i)
        directions.append(direction)
        lengths.append(distance)
        counts["dep_direction"][f"{token.dep_}:{direction}"] += 1
        counts["dep_length"][f"{token.dep_}:{bin_distance(distance)}"] += 1

    if directions:
        metrics["right_dependency_share"] = sum(d == "R" for d in directions) / len(directions)
    if lengths:
        metrics["mean_dependency_length"] = statistics.fmean(lengths)
    if depths:
        metrics["mean_dependency_depth"] = statistics.fmean(depths)
    return Extracted(dict(counts), metrics)


def log_ratio(a: int, n_a: int, b: int, n_b: int) -> float:
    rate_a = (a + 0.5) / (n_a + 1.0)
    rate_b = (b + 0.5) / (n_b + 1.0)
    return math.log2(rate_a / rate_b)


def g2(a: int, n_a: int, b: int, n_b: int) -> float:
    table = ((a, n_a - a), (b, n_b - b))
    row_totals = (n_a, n_b)
    col_totals = (a + b, (n_a - a) + (n_b - b))
    grand = n_a + n_b
    score = 0.0
    for r in range(2):
        for c in range(2):
            observed = table[r][c]
            expected = row_totals[r] * col_totals[c] / grand if grand else 0.0
            if observed > 0 and expected > 0:
                score += observed * math.log(observed / expected)
    return 2.0 * score


def feature_rate(counter: Counter[str], feature: str) -> float:
    total = sum(counter.values())
    return counter[feature] / total if total else 0.0


def pair_evidence(
    feature: str,
    family: str,
    pairs: Sequence[Pair],
    extracted_a: Sequence[Extracted],
    extracted_b: Sequence[Extracted],
    corpus_direction: int,
) -> tuple[float, float, int, int]:
    observed = 0
    informative = 0
    agreeing = 0
    for ea, eb in zip(extracted_a, extracted_b):
        ca = ea.counts.get(family, Counter())
        cb = eb.counts.get(family, Counter())
        if ca[feature] + cb[feature] > 0:
            observed += 1
        delta = feature_rate(ca, feature) - feature_rate(cb, feature)
        if delta == 0:
            continue
        informative += 1
        if (1 if delta > 0 else -1) == corpus_direction:
            agreeing += 1
    dispersion = observed / len(pairs) if pairs else 0.0
    consistency = agreeing / informative if informative else 0.0
    return dispersion, consistency, observed, informative


def literal_baseline_match(feature: str, family: str, baseline: str) -> bool:
    if not baseline:
        return False
    if family.startswith("function_word"):
        pattern = r"(?<!\w)" + re.escape(feature.casefold()) + r"(?!\w)"
        return re.search(pattern, baseline) is not None
    return f"{family}:{feature}".casefold() in baseline


def build_rows(
    pairs: Sequence[Pair],
    extracted_a: Sequence[Extracted],
    extracted_b: Sequence[Extracted],
    baseline: str,
    min_count: int,
) -> list[dict[str, Any]]:
    families = sorted(
        set().union(*(e.counts.keys() for e in extracted_a), *(e.counts.keys() for e in extracted_b))
    )
    rows: list[dict[str, Any]] = []
    for family in families:
        corpus_a = Counter()
        corpus_b = Counter()
        for e in extracted_a:
            corpus_a.update(e.counts.get(family, Counter()))
        for e in extracted_b:
            corpus_b.update(e.counts.get(family, Counter()))
        n_a = sum(corpus_a.values())
        n_b = sum(corpus_b.values())
        if not n_a or not n_b:
            continue
        for feature in set(corpus_a) | set(corpus_b):
            a = corpus_a[feature]
            b = corpus_b[feature]
            if a + b < min_count:
                continue
            lr = log_ratio(a, n_a, b, n_b)
            direction = 1 if lr > 0 else -1
            dispersion, consistency, observed, informative = pair_evidence(
                feature, family, pairs, extracted_a, extracted_b, direction
            )
            rows.append(
                {
                    "family": family,
                    "feature": feature,
                    "a_count": a,
                    "b_count": b,
                    "a_total": n_a,
                    "b_total": n_b,
                    "a_per_10k": a / n_a * 10000,
                    "b_per_10k": b / n_b * 10000,
                    "log_ratio": lr,
                    "g2": g2(a, n_a, b, n_b),
                    "dispersion": dispersion,
                    "direction_consistency": consistency,
                    "observed_pairs": observed,
                    "informative_pairs": informative,
                    "baseline_literal_match": literal_baseline_match(feature, family, baseline),
                }
            )
    return rows


def descriptive_metric_summary(extracted: Sequence[Extracted]) -> dict[str, Any]:
    metrics = sorted(set().union(*(e.metrics.keys() for e in extracted)))
    result: dict[str, Any] = {}
    for metric in metrics:
        values = [e.metrics[metric] for e in extracted if metric in e.metrics]
        if not values:
            continue
        result[metric] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "documents": len(values),
        }
    return result


def provenance_summary(records: Sequence[TextRecord]) -> dict[str, Any]:
    with_provenance = [record for record in records if record.provenance]
    authorship_statuses = Counter(
        str(record.provenance.get("authorship_status", "UNSPECIFIED"))
        for record in with_provenance
    )
    source_kinds = Counter(
        str(record.provenance.get("source_kind", "UNSPECIFIED"))
        for record in with_provenance
    )
    return {
        "records_with_provenance": len(with_provenance),
        "records_without_provenance": len(records) - len(with_provenance),
        "authorship_statuses": dict(sorted(authorship_statuses.items())),
        "source_kinds": dict(sorted(source_kinds.items())),
    }


def build_descriptive_profile(
    records: Sequence[TextRecord],
    extracted: Sequence[Extracted],
    baseline: str,
    min_count: int,
) -> dict[str, Any]:
    families = sorted(set().union(*(e.counts.keys() for e in extracted)))
    rows: list[dict[str, Any]] = []
    for family in families:
        corpus = Counter()
        for e in extracted:
            corpus.update(e.counts.get(family, Counter()))
        total = sum(corpus.values())
        if not total:
            continue
        for feature, count in corpus.items():
            if count < min_count:
                continue
            documents = sum(
                1 for e in extracted if e.counts.get(family, Counter())[feature] > 0
            )
            rows.append(
                {
                    "family": family,
                    "feature": feature,
                    "count": count,
                    "total": total,
                    "per_10k": count / total * 10000,
                    "document_dispersion": documents / len(records) if records else 0.0,
                    "documents_present": documents,
                    "baseline_literal_match": literal_baseline_match(feature, family, baseline),
                }
            )
    return {
        "mode": "descriptive",
        "record_count": len(records),
        "corpus_stats": {
            "tokens": sum(len(simple_tokens(record.text)) for record in records),
            "characters": sum(len(surface_text(record.text)) for record in records),
        },
        "provenance": provenance_summary(records),
        "structural_metrics": descriptive_metric_summary(extracted),
        "filters": {"min_count": min_count},
        "features": sorted(rows, key=lambda r: (r["count"], r["document_dispersion"]), reverse=True),
    }


def add_descriptive_concordances(
    rows: list[dict[str, Any]],
    records: Sequence[TextRecord],
    nlp: Any | None,
    limit_rows: int,
    examples_per_corpus: int,
) -> None:
    ranked = sorted(rows, key=lambda r: (r["count"], r["document_dispersion"]), reverse=True)[:limit_rows]
    for row in ranked:
        examples: list[dict[str, str]] = []
        for record in records:
            if len(examples) >= examples_per_corpus:
                break
            snippet = surface_concordance(record.text, row["feature"], row["family"])
            snippet = snippet or syntax_concordance(record.text, row["feature"], row["family"], nlp)
            if snippet:
                examples.append({"text_id": record.text_id, "text": snippet})
        if examples:
            row["concordances"] = examples


def metric_summary(extracted_a: Sequence[Extracted], extracted_b: Sequence[Extracted]) -> dict[str, Any]:
    metrics = sorted(set().union(*(e.metrics.keys() for e in extracted_a), *(e.metrics.keys() for e in extracted_b)))
    result: dict[str, Any] = {}
    for metric in metrics:
        values_a = [e.metrics[metric] for e in extracted_a if metric in e.metrics]
        values_b = [e.metrics[metric] for e in extracted_b if metric in e.metrics]
        paired = [
            (ea.metrics[metric], eb.metrics[metric])
            for ea, eb in zip(extracted_a, extracted_b)
            if metric in ea.metrics and metric in eb.metrics
        ]
        if not values_a or not values_b or not paired:
            continue
        deltas = [a - b for a, b in paired]
        result[metric] = {
            "a_mean": statistics.fmean(values_a),
            "b_mean": statistics.fmean(values_b),
            "paired_mean_delta_a_minus_b": statistics.fmean(deltas),
            "paired_median_delta_a_minus_b": statistics.median(deltas),
            "a_higher_share": sum(d > 0 for d in deltas) / len(deltas),
            "pairs": len(deltas),
        }
    return result


def surface_concordance(text: str, feature: str, family: str, width: int = 90) -> str | None:
    if not feature.strip() or family.startswith("pos_") or family.startswith("dep_"):
        return None
    target = feature.strip() if family.startswith("function_word") else feature
    haystack = surface_text(text)
    if family.startswith("function_word"):
        match = re.search(
            r"(?<!\w)" + re.escape(target) + r"(?!\w)",
            haystack,
            flags=re.IGNORECASE,
        )
        idx = match.start() if match else -1
    else:
        match = re.search(re.escape(target), haystack, flags=re.IGNORECASE)
        idx = match.start() if match else -1
    if idx < 0:
        return None
    start = max(0, idx - width)
    end = min(len(haystack), idx + len(target) + width)
    return haystack[start:end]


def syntax_concordance(text: str, feature: str, family: str, nlp: Any | None) -> str | None:
    if nlp is None or not (family.startswith("pos_") or family.startswith("dep_")):
        return None
    doc = nlp(text)
    if family.startswith("pos_"):
        n = int(family.split("_")[1][0])
        tokens = [t for t in doc if not t.is_space]
        wanted = feature.split()
        for i in range(len(tokens) - n + 1):
            if [t.pos_ for t in tokens[i : i + n]] == wanted:
                return tokens[i].sent.text.strip()
        return None
    for token in doc:
        if family == "dep_direction" and token.head.i != token.i:
            direction = "R" if token.i > token.head.i else "L"
            if f"{token.dep_}:{direction}" == feature:
                return token.sent.text.strip()
        elif family == "dep_length" and token.head.i != token.i:
            distance = abs(token.i - token.head.i)
            if f"{token.dep_}:{bin_distance(distance)}" == feature:
                return token.sent.text.strip()
        elif family == "dep_depth" and bin_depth(token_depth(token)) == feature:
            return token.sent.text.strip()
    return None


def add_concordances(
    rows: list[dict[str, Any]],
    pairs: Sequence[Pair],
    nlp: Any | None,
    limit_rows: int,
    examples_per_side: int,
) -> None:
    ranked = sorted(rows, key=lambda r: (r["g2"], abs(r["log_ratio"])), reverse=True)[:limit_rows]
    for row in ranked:
        family, feature = row["family"], row["feature"]
        examples = {"a": [], "b": []}
        for pair in pairs:
            for side, text in (("a", pair.a), ("b", pair.b)):
                if len(examples[side]) >= examples_per_side:
                    continue
                snippet = surface_concordance(text, feature, family) or syntax_concordance(text, feature, family, nlp)
                if snippet:
                    examples[side].append({"pair_id": pair.pair_id, "text": snippet})
            if all(len(examples[s]) >= examples_per_side for s in ("a", "b")):
                break
        row["concordances"] = examples


def read_baseline(paths: Sequence[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        if path.exists() and path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts).casefold()


def select_rows(
    rows: Sequence[dict[str, Any]],
    min_abs_log_ratio: float,
    min_g2: float,
    min_dispersion: float,
    min_consistency: float,
) -> list[dict[str, Any]]:
    return [
        r for r in rows
        if abs(r["log_ratio"]) >= min_abs_log_ratio
        and r["g2"] >= min_g2
        and r["dispersion"] >= min_dispersion
        and r["direction_consistency"] >= min_consistency
    ]


def render_descriptive_markdown(
    payload: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    top_per_family: int,
) -> str:
    lines = [
        "# Descriptive style profile",
        "",
        f"Texts: **{payload['record_count']}**; tokens: **{payload['corpus_stats']['tokens']}**.",
        "",
        "No reference corpus is used here. Log Ratio and G² are therefore not computed; this report describes the candidate corpus only.",
        "",
        "Construct features use sentence boundaries and regular expressions and remain available without a syntax model.",
        "",
        (
            "Provenance: "
            f"{payload.get('provenance', {}).get('records_with_provenance', 0)} records carry metadata; "
            f"authorship statuses: {payload.get('provenance', {}).get('authorship_statuses', {})}."
        ),
        "",
        "## Structural metrics",
        "",
        "| Metric | Mean | Median | Min | Max | Documents |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, stats in payload.get("structural_metrics", {}).items():
        lines.append(
            f"| `{metric}` | {stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['min']:.4f} | {stats['max']:.4f} | {stats['documents']} |"
        )
    if not payload.get("structural_metrics"):
        lines.append("| _No syntax model used_ |  |  |  |  |  |")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family in sorted(by_family):
        lines += [
            "",
            f"## {family}",
            "",
            "| Feature | Count | Per 10k within family | Documents | Dispersion | Baseline literal |",
            "|---|---:|---:|---:|---:|---|",
        ]
        ranked = sorted(
            by_family[family],
            key=lambda r: (r["count"], r["document_dispersion"]),
            reverse=True,
        )[:top_per_family]
        for row in ranked:
            feature = str(row["feature"]).replace("|", "\\|")
            lines.append(
                f"| `{feature}` | {row['count']} | {row['per_10k']:.1f} | "
                f"{row['documents_present']} | {row['document_dispersion']:.1%} | "
                f"{'yes' if row['baseline_literal_match'] else 'no'} |"
            )

    evidence_rows = [r for r in rows if r.get("concordances")]
    if evidence_rows:
        lines += ["", "## Concordance packets"]
        for row in sorted(evidence_rows, key=lambda r: r["count"], reverse=True):
            lines += ["", f"### `{row['family']} :: {row['feature']}`", ""]
            for example in row["concordances"]:
                lines.append(f"- `{example['text_id']}` — {example['text']}")
    return "\n".join(lines) + "\n"


def render_markdown(
    payload: Mapping[str, Any],
    rows: Sequence[dict[str, Any]],
    top_per_family: int,
) -> str:
    a_label = payload["a_label"]
    b_label = payload["b_label"]
    lines = [
        "# Style discovery report",
        "",
        f"Paired texts: **{payload['pair_count']}**. Positive Log Ratio means more frequent in **{a_label}**; negative means more frequent in **{b_label}**.",
        "",
        "This report ranks evidence; it does not promote features to style rules automatically.",
        "",
        "## Structural metrics outside keyness",
        "",
        "| Metric | A mean | B mean | paired Δ A−B | A higher share |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, stats in payload.get("structural_metrics", {}).items():
        lines.append(
            f"| `{metric}` | {stats['a_mean']:.4f} | {stats['b_mean']:.4f} | "
            f"{stats['paired_mean_delta_a_minus_b']:.4f} | {stats['a_higher_share']:.1%} |"
        )
    if not payload.get("structural_metrics"):
        lines.append("| _No syntax model used_ |  |  |  |  |")

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family in sorted(by_family):
        lines += [
            "",
            f"## {family}",
            "",
            "| Direction | Feature | A/10k | B/10k | Log Ratio | G² | Dispersion | Consistency | Baseline literal |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        ranked = sorted(by_family[family], key=lambda r: (r["g2"], abs(r["log_ratio"])), reverse=True)[:top_per_family]
        for row in ranked:
            direction = a_label if row["log_ratio"] > 0 else b_label
            feature = str(row["feature"]).replace("|", "\\|")
            lines.append(
                f"| {direction} | `{feature}` | {row['a_per_10k']:.1f} | {row['b_per_10k']:.1f} | "
                f"{row['log_ratio']:.3f} | {row['g2']:.2f} | {row['dispersion']:.1%} | "
                f"{row['direction_consistency']:.1%} | {'yes' if row['baseline_literal_match'] else 'no'} |"
            )

    evidence_rows = [r for r in rows if r.get("concordances")]
    if evidence_rows:
        lines += ["", "## Concordance packets"]
        for row in sorted(evidence_rows, key=lambda r: r["g2"], reverse=True):
            lines += [
                "",
                f"### `{row['family']} :: {row['feature']}`",
                "",
                f"Log Ratio {row['log_ratio']:.3f}; G² {row['g2']:.2f}; dispersion {row['dispersion']:.1%}; direction consistency {row['direction_consistency']:.1%}.",
            ]
            for side, label in (("a", a_label), ("b", b_label)):
                examples = row["concordances"].get(side, [])
                if not examples:
                    continue
                lines += ["", f"**{label}**"]
                for ex in examples:
                    lines.append(f"- `{ex['pair_id']}` — {ex['text']}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover empirical writing-style patterns")
    parser.add_argument("input", type=Path, help="JSON/JSONL with paired or standalone texts")
    parser.add_argument(
        "--mode",
        choices=("paired", "descriptive"),
        default="paired",
        help="paired keyness (default) or a descriptive profile for one corpus",
    )
    parser.add_argument("--a-field", default="draft")
    parser.add_argument("--b-field", default="final")
    parser.add_argument("--id-field", default="pair_id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--text-id-field", default="id")
    parser.add_argument("--a-label", default="LLM draft")
    parser.add_argument("--b-label", default="final text")
    parser.add_argument("--spacy-model", default="nl_core_news_sm")
    parser.add_argument("--no-syntax", action="store_true")
    parser.add_argument("--known", nargs="*", type=Path, default=[])
    parser.add_argument("--out-dir", type=Path, default=Path("style-discovery-output"))
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--min-abs-log-ratio", type=float, default=0.0)
    parser.add_argument("--min-g2", type=float, default=0.0)
    parser.add_argument("--min-dispersion", type=float, default=0.0)
    parser.add_argument("--min-consistency", type=float, default=0.0)
    parser.add_argument("--top-per-family", type=int, default=20)
    parser.add_argument("--concordance-rows", type=int, default=20)
    parser.add_argument("--examples-per-side", type=int, default=2)
    parser.add_argument("--no-concordance", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "descriptive":
        records = load_texts(args.input, args.text_field, args.text_id_field)
        nlp = None if args.no_syntax else load_nlp(args.spacy_model)
        extracted = [extract(record.text, nlp, include_syntax=not args.no_syntax) for record in records]
        baseline = read_baseline(args.known)
        payload = build_descriptive_profile(records, extracted, baseline, args.min_count)
        payload.update(
            {
                "input": str(args.input),
                "spacy_model": None if args.no_syntax else args.spacy_model,
            }
        )
        if not args.no_concordance:
            add_descriptive_concordances(
                payload["features"],
                records,
                nlp,
                args.concordance_rows,
                args.examples_per_side,
            )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "style-discovery.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.out_dir / "style-discovery.md").write_text(
            render_descriptive_markdown(payload, payload["features"], args.top_per_family),
            encoding="utf-8",
        )
        print(f"Wrote {args.out_dir / 'style-discovery.json'}")
        print(f"Wrote {args.out_dir / 'style-discovery.md'}")
        return 0

    pairs = load_pairs(args.input, args.a_field, args.b_field, args.id_field)
    nlp = None if args.no_syntax else load_nlp(args.spacy_model)
    extracted_a = [extract(pair.a, nlp, include_syntax=not args.no_syntax) for pair in pairs]
    extracted_b = [extract(pair.b, nlp, include_syntax=not args.no_syntax) for pair in pairs]
    baseline = read_baseline(args.known)
    rows = build_rows(pairs, extracted_a, extracted_b, baseline, args.min_count)
    rows = select_rows(
        rows,
        args.min_abs_log_ratio,
        args.min_g2,
        args.min_dispersion,
        args.min_consistency,
    )
    if not args.no_concordance:
        add_concordances(rows, pairs, nlp, args.concordance_rows, args.examples_per_side)

    payload = {
        "pair_count": len(pairs),
        "a_label": args.a_label,
        "b_label": args.b_label,
        "input": str(args.input),
        "spacy_model": None if args.no_syntax else args.spacy_model,
        "structural_metrics": metric_summary(extracted_a, extracted_b),
        "filters": {
            "min_count": args.min_count,
            "min_abs_log_ratio": args.min_abs_log_ratio,
            "min_g2": args.min_g2,
            "min_dispersion": args.min_dispersion,
            "min_consistency": args.min_consistency,
        },
        "features": sorted(rows, key=lambda r: (r["g2"], abs(r["log_ratio"])), reverse=True),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "style-discovery.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out_dir / "style-discovery.md").write_text(
        render_markdown(payload, rows, args.top_per_family), encoding="utf-8"
    )
    print(f"Wrote {args.out_dir / 'style-discovery.json'}")
    print(f"Wrote {args.out_dir / 'style-discovery.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
