"""Summarize a prompt-experiment report: eligibility gates first, then ranking, then worst cells."""

import json
from pathlib import Path
from typing import Any

import polars as pl


def observations_frame(report: dict[str, Any]) -> pl.DataFrame:
    rows = []
    cases = {c["id"]: c for c in report["plan"]["cases"]}
    for row in report["observations"]:
        case = cases[row["case_id"]]
        metrics = row.get("metrics") or {}
        judgements = [j for j in row.get("judgements", []) if not j.get("self_judged")]
        scores = [j["score"] for j in judgements if "score" in j]
        usage = row.get("usage") or {}
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "model": row["model"],
                "case_id": row["case_id"],
                "replicate": row["replicate"],
                "valid_case": not case["expect_refusal"] and not case.get("synthetic_context"),
                "expect_refusal": case["expect_refusal"],
                "transport_error": row.get("transport_error") is not None,
                "parse_ok": bool(row.get("parse_ok")),
                "status": row.get("status"),
                "refusal_correct": row.get("refusal_correct"),
                "forbidden_term_hits": row.get("forbidden_term_hits"),
                "micro_skill_count": metrics.get("micro_skill_count"),
                "quote_exactness": metrics.get("quote_exactness"),
                "quote_support": metrics.get("quote_support"),
                "locator_exactness": metrics.get("locator_exactness"),
                "observable_rate": metrics.get("observable_rate"),
                "indicator_coverage": metrics.get("indicator_coverage"),
                "t0_share": metrics.get("t0_share"),
                "duplicate_rate": metrics.get("duplicate_rate"),
                "mean_cognitive_level": metrics.get("mean_cognitive_level"),
                "prerequisite_count": metrics.get("prerequisite_count"),
                "judge_mean": sum(scores) / len(scores) if scores else None,
                "judge_min": min(scores) if scores else None,
                "judge_verdicts": ",".join(sorted(j.get("verdict", "?") for j in judgements)),
                "prompt_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "thoughts_tokens": usage.get("thoughts_tokens"),
                "latency_ms": row.get("latency_ms"),
                "response_sha256": row.get("response_sha256"),
            }
        )
    return pl.DataFrame(rows)


def eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    """Decision 0013 gates 1 to 3 per (prompt, model)."""
    valid = frame.filter(pl.col("valid_case"))
    malformed = frame.filter(pl.col("expect_refusal"))
    injection = frame.filter(pl.col("forbidden_term_hits").is_not_null())
    keys = ["prompt_id", "model"]
    fabricated = (
        valid.filter(pl.col("status") == "ok")
        .group_by(keys)
        .agg(
            (pl.col("quote_support") < 1.0).sum().alias("cells_with_unsupported_quotes"),
            pl.col("quote_support").min().alias("min_quote_support"),
            pl.col("quote_exactness").mean().alias("mean_quote_exactness"),
        )
    )
    refusals = malformed.group_by(keys).agg(
        pl.col("refusal_correct").cast(pl.Float64).mean().alias("refusal_correctness")
    )
    # Refusing an injected context is acceptable; producing Python skills is not.
    injected = injection.group_by(keys).agg(
        pl.col("forbidden_term_hits").sum().alias("forbidden_term_hits"),
        (pl.col("parse_ok").not_()).sum().alias("injection_parse_failures"),
    )
    parse = frame.group_by(keys).agg(
        pl.col("parse_ok").cast(pl.Float64).mean().alias("schema_adherence"),
        pl.col("transport_error").sum().alias("transport_errors"),
    )
    wrong_refusals = valid.group_by(keys).agg(
        (pl.col("status") == "refused").sum().alias("unjustified_refusals")
    )
    table = (
        parse.join(fabricated, on=keys, how="left")
        .join(refusals, on=keys, how="left")
        .join(injected, on=keys, how="left")
        .join(wrong_refusals, on=keys, how="left")
        .fill_null(0)
    )
    return table.with_columns(
        (
            (pl.col("cells_with_unsupported_quotes") == 0)
            & (pl.col("refusal_correctness") >= 1.0)
            & (pl.col("forbidden_term_hits") == 0)
            & (pl.col("injection_parse_failures") == 0)
            & (pl.col("unjustified_refusals") == 0)
            & (pl.col("schema_adherence") >= 0.95)
        ).alias("eligible")
    ).sort(keys)


def ranking(frame: pl.DataFrame) -> pl.DataFrame:
    """Gates 4 to 7 over valid, non-refused cells."""
    valid = frame.filter(pl.col("valid_case") & (pl.col("status") == "ok"))
    return (
        valid.group_by(["prompt_id", "model"])
        .agg(
            pl.col("judge_mean").mean().alias("judge_mean"),
            pl.col("judge_min").min().alias("judge_worst_cell"),
            pl.col("indicator_coverage").mean().alias("indicator_coverage"),
            pl.col("quote_support").mean().alias("quote_support"),
            pl.col("quote_exactness").mean().alias("quote_exactness"),
            pl.col("observable_rate").mean().alias("observable_rate"),
            pl.col("t0_share").mean().alias("t0_share"),
            pl.col("micro_skill_count").mean().alias("micro_skills"),
            (
                (pl.col("output_tokens") + pl.col("thoughts_tokens")).sum()
                / pl.col("micro_skill_count").sum()
            ).alias("tokens_per_micro_skill"),
            pl.col("latency_ms").median().alias("latency_ms_median"),
        )
        .sort(
            ["judge_mean", "indicator_coverage", "tokens_per_micro_skill", "latency_ms_median"],
            descending=[True, True, False, False],
        )
    )


def worst_cells(frame: pl.DataFrame, limit: int = 12) -> pl.DataFrame:
    valid = frame.filter(pl.col("valid_case") & pl.col("parse_ok"))
    return (
        valid.sort(["judge_min", "quote_support"], nulls_last=True)
        .select(
            "prompt_id",
            "model",
            "case_id",
            "status",
            "quote_support",
            "judge_min",
            "judge_verdicts",
        )
        .head(limit)
    )


def behavioural(frame: pl.DataFrame, report: dict[str, Any]) -> pl.DataFrame:
    """Directional (grade) and invariance (paraphrase) pairs by prompt and model."""
    cases = {c["id"]: c for c in report["plan"]["cases"]}
    outputs = {
        (r["prompt_id"], r["model"], r["case_id"]): r
        for r in report["observations"]
        if r.get("parse_ok")
    }
    rows = []
    for (prompt_id, model, case_id), row in outputs.items():
        case = cases[case_id]
        pair = case.get("paraphrase_of") or case.get("directional_pair")
        if not pair or (prompt_id, model, pair) not in outputs:
            continue
        other = outputs[(prompt_id, model, pair)]
        kind = "invariance" if case.get("paraphrase_of") else "directional"
        left = (row.get("metrics") or {}).get("mean_cognitive_level")
        right = (other.get("metrics") or {}).get("mean_cognitive_level")
        rows.append(
            {
                "kind": kind,
                "prompt_id": prompt_id,
                "model": model,
                "case_id": case_id,
                "pair": pair,
                "grade": case["grade"],
                "pair_grade": cases[pair]["grade"],
                "cognitive_level": left,
                "pair_cognitive_level": right,
                "status": row.get("status"),
                "pair_status": other.get("status"),
            }
        )
    return pl.DataFrame(rows)


def invariance(
    report: dict[str, Any], outputs_by_key: dict[tuple[str, str, str], Any]
) -> pl.DataFrame:
    """Paraphrase invariance: signature Jaccard and quoted-indicator Jaccard per prompt and model."""
    from goes_tech_kg.eval import prompt_metrics

    cases = {c["id"]: c for c in report["plan"]["cases"]}
    rows = []
    for (prompt_id, model, case_id), output in outputs_by_key.items():
        pair = cases[case_id].get("paraphrase_of")
        if not pair or (prompt_id, model, pair) not in outputs_by_key:
            continue
        other = outputs_by_key[(prompt_id, model, pair)]
        rows.append(
            {
                "prompt_id": prompt_id,
                "model": model,
                "signature_jaccard": prompt_metrics.jaccard(output, other),
                "indicator_jaccard": prompt_metrics.indicator_set_jaccard(output, other),
                "micro_skills": len(output.micro_skills),
                "pair_micro_skills": len(other.micro_skills),
            }
        )
    return pl.DataFrame(rows)


def judge_agreement(report: dict[str, Any]) -> pl.DataFrame:
    """Pairwise judge agreement on verdict and score across all non-self-judged cells."""
    rows = []
    for row in report["observations"]:
        judgements = {
            j["judge_model"]: j
            for j in row.get("judgements", [])
            if "score" in j and not j.get("self_judged")
        }
        if len(judgements) == 2:
            (a, ja), (b, jb) = sorted(judgements.items())
            rows.append(
                {
                    "judge_a": a,
                    "judge_b": b,
                    "verdict_agree": ja["verdict"] == jb["verdict"],
                    "score_gap": abs(ja["score"] - jb["score"]),
                }
            )
    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .group_by(["judge_a", "judge_b"])
        .agg(
            pl.len().alias("cells"),
            pl.col("verdict_agree").cast(pl.Float64).mean().alias("verdict_agreement"),
            pl.col("score_gap").mean().alias("mean_score_gap"),
        )
    )


def load_report(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
