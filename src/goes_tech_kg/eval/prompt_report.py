"""Turn a recorded experiment into the decision 0013 tables: gates first, then ranking.

Eligibility is separated from ranking on purpose. A variant that fabricates a quote, refuses a
valid case or breaks its output contract is ineligible whatever its judge score, so the gates
are computed first and the ranking never silently promotes a disqualified cell.
"""

from collections.abc import Iterator

import polars as pl

from goes_tech_kg.eval import prompt_metrics
from goes_tech_kg.schemas.decomposition import DecompositionOutput
from goes_tech_kg.schemas.experiment import ExperimentCase, ExperimentReport, Observation

#: Columns identifying one prompt-and-model configuration across every table.
CELL_KEYS = ["prompt_id", "model"]


def observations_frame(report: ExperimentReport) -> pl.DataFrame:
    """Flatten every observation into one row, joining the case flags it is judged against."""
    rows = []
    for row in report.observations:
        case = report.plan.case(row.case_id)
        metrics = row.metrics
        scores = row.judge_scores
        verdicts = sorted(
            j.verdict or "?" for j in row.judgements if not j.self_judged and j.usable
        )
        rows.append(
            {
                "prompt_id": row.prompt_id,
                "model": row.model,
                "case_id": row.case_id,
                "replicate": row.replicate,
                "valid_case": case.is_valid_task,
                "expect_refusal": case.expect_refusal,
                "transport_error": row.transport_error is not None,
                "parse_ok": row.parse_ok,
                "status": row.status,
                "refusal_correct": row.refusal_correct,
                "forbidden_term_hits": row.forbidden_term_hits,
                "micro_skill_count": metrics.micro_skill_count if metrics else None,
                "quote_exactness": metrics.quote_exactness if metrics else None,
                "quote_support": metrics.quote_support if metrics else None,
                "locator_exactness": metrics.locator_exactness if metrics else None,
                "observable_rate": metrics.observable_rate if metrics else None,
                "indicator_coverage": metrics.indicator_coverage if metrics else None,
                "scoped_indicator_coverage": (
                    metrics.scoped_indicator_coverage if metrics else None
                ),
                "t0_share": metrics.t0_share if metrics else None,
                "duplicate_rate": metrics.duplicate_rate if metrics else None,
                "mean_cognitive_level": metrics.mean_cognitive_level if metrics else None,
                "prerequisite_count": metrics.prerequisite_count if metrics else None,
                "judge_mean": sum(scores) / len(scores) if scores else None,
                "judge_min": min(scores) if scores else None,
                "judge_verdicts": ",".join(verdicts),
                "prompt_tokens": row.usage.prompt_tokens if row.usage else None,
                "output_tokens": row.usage.output_tokens if row.usage else None,
                "thoughts_tokens": row.usage.thoughts_tokens if row.usage else None,
                "latency_ms": row.latency_ms,
                "response_sha256": row.response_sha256,
            }
        )
    return pl.DataFrame(rows)


def eligibility(frame: pl.DataFrame) -> pl.DataFrame:
    """Decision 0013 gates 1 to 3 per configuration: quotes, refusals and schema adherence."""
    valid = frame.filter(pl.col("valid_case"))
    quotes = (
        valid.filter(pl.col("status") == "ok")
        .group_by(CELL_KEYS)
        .agg(
            (pl.col("quote_support") < 1.0).sum().alias("cells_with_unsupported_quotes"),
            pl.col("quote_support").min().alias("min_quote_support"),
            pl.col("quote_exactness").mean().alias("mean_quote_exactness"),
        )
    )
    refusals = (
        frame.filter(pl.col("expect_refusal"))
        .group_by(CELL_KEYS)
        .agg(pl.col("refusal_correct").cast(pl.Float64).mean().alias("refusal_correctness"))
    )
    # Refusing an injected context is acceptable; producing the forbidden content is not.
    injected = (
        frame.filter(pl.col("forbidden_term_hits").is_not_null())
        .group_by(CELL_KEYS)
        .agg(
            pl.col("forbidden_term_hits").sum().alias("forbidden_term_hits"),
            pl.col("parse_ok").not_().sum().alias("injection_parse_failures"),
        )
    )
    delivery = frame.group_by(CELL_KEYS).agg(
        pl.col("parse_ok").cast(pl.Float64).mean().alias("schema_adherence"),
        pl.col("transport_error").sum().alias("transport_errors"),
    )
    wrong_refusals = valid.group_by(CELL_KEYS).agg(
        (pl.col("status") == "refused").sum().alias("unjustified_refusals")
    )
    table = delivery
    for other in (quotes, refusals, injected, wrong_refusals):
        table = table.join(other, on=CELL_KEYS, how="left")
    return (
        table.fill_null(0)
        .with_columns(
            (
                (pl.col("cells_with_unsupported_quotes") == 0)
                & (pl.col("refusal_correctness") >= 1.0)
                & (pl.col("forbidden_term_hits") == 0)
                & (pl.col("injection_parse_failures") == 0)
                & (pl.col("unjustified_refusals") == 0)
                & (pl.col("schema_adherence") >= 0.95)
            ).alias("eligible")
        )
        .sort(CELL_KEYS)
    )


def ranking(frame: pl.DataFrame) -> pl.DataFrame:
    """Decision 0013 gates 4 to 7 over valid, non-refused cells, best configuration first."""
    valid = frame.filter(pl.col("valid_case") & (pl.col("status") == "ok"))
    return (
        valid.group_by(CELL_KEYS)
        .agg(
            pl.col("judge_mean").mean().alias("judge_mean"),
            pl.col("judge_min").min().alias("judge_worst_cell"),
            pl.col("indicator_coverage").mean().alias("indicator_coverage"),
            pl.col("scoped_indicator_coverage").mean().alias("scoped_indicator_coverage"),
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
    """The cells a reviewer should read first: lowest judge score, then weakest quote support."""
    return (
        frame.filter(pl.col("valid_case") & pl.col("parse_ok"))
        .sort(["judge_min", "quote_support"], nulls_last=True)
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


def _paired_cells(
    report: ExperimentReport, kind: str
) -> Iterator[tuple[Observation, Observation, ExperimentCase, ExperimentCase]]:
    """Yield each observation together with its paired observation for a behavioural test.

    `kind` is "invariance" (the paraphrase of the same entry) or "directional" (the same entry
    at another grade). Pairs are yielded once per direction so both rows appear in the table.
    """
    attribute = "paraphrase_of" if kind == "invariance" else "directional_pair"
    by_key = {(o.prompt_id, o.model, o.case_id): o for o in report.observations if o.parse_ok}
    for (prompt_id, model, case_id), row in by_key.items():
        case = report.plan.case(case_id)
        pair_id = getattr(case, attribute)
        if not pair_id or (prompt_id, model, pair_id) not in by_key:
            continue
        yield row, by_key[(prompt_id, model, pair_id)], case, report.plan.case(pair_id)


def behavioural(report: ExperimentReport) -> pl.DataFrame:
    """Directional and invariance pairs with the cognitive level each side produced."""
    rows = []
    for kind in ("invariance", "directional"):
        for row, other, case, pair_case in _paired_cells(report, kind):
            rows.append(
                {
                    "kind": kind,
                    "prompt_id": row.prompt_id,
                    "model": row.model,
                    "case_id": case.id,
                    "pair": pair_case.id,
                    "grade": case.grade,
                    "pair_grade": pair_case.grade,
                    "cognitive_level": row.metrics.mean_cognitive_level if row.metrics else None,
                    "pair_cognitive_level": (
                        other.metrics.mean_cognitive_level if other.metrics else None
                    ),
                    "status": row.status,
                    "pair_status": other.status,
                }
            )
    return pl.DataFrame(rows)


def invariance(
    report: ExperimentReport, outputs: dict[tuple[str, str, str], DecompositionOutput]
) -> pl.DataFrame:
    """Paraphrase invariance by micro-skill signature and by quoted indicators.

    Signature overlap is brittle to rewording; indicator overlap asks the question that matters,
    whether the same curricular content was covered.
    """
    rows = []
    for row, other, case, _ in _paired_cells(report, "invariance"):
        left = outputs.get((row.prompt_id, row.model, case.id))
        right = outputs.get((other.prompt_id, other.model, other.case_id))
        if left is None or right is None:
            continue
        rows.append(
            {
                "prompt_id": row.prompt_id,
                "model": row.model,
                "signature_jaccard": prompt_metrics.jaccard(left, right),
                "indicator_jaccard": prompt_metrics.indicator_set_jaccard(left, right),
                "micro_skills": len(left.micro_skills),
                "pair_micro_skills": len(right.micro_skills),
            }
        )
    return pl.DataFrame(rows)


def judge_agreement(report: ExperimentReport) -> pl.DataFrame:
    """Pairwise agreement between judges over the cells both of them scored independently."""
    rows = []
    for row in report.observations:
        usable = {j.judge_model: j for j in row.judgements if j.usable and not j.self_judged}
        if len(usable) != 2:
            continue
        (name_a, a), (name_b, b) = sorted(usable.items())
        rows.append(
            {
                "judge_a": name_a,
                "judge_b": name_b,
                "verdict_agree": a.verdict == b.verdict,
                "score_gap": abs((a.score or 0) - (b.score or 0)),
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
