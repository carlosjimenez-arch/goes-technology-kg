"""Escaped, printable English curriculum proposal; no remote assets or scripts."""

import html
from typing import Any


def render_proposal(version: str, rows: list[dict[str, Any]], schedule: dict[str, Any]) -> str:
    def esc(value: object) -> str:
        return html.escape(str(value))

    blocks = [
        f"<h1>Technology curriculum proposal — {esc(version)}</h1>",
        "<p class='notice'>Research proposal for El Salvador, grades 2–6. Not an official allocation or a human-validated curriculum. Confidence is uncalibrated. Source gaps and external prerequisites remain visible.</p>",
        "<nav>"
        + " · ".join(
            f"<a href='#grade-{g}'>Grade {g}</a>" for g in sorted({r["grade"] for r in rows})
        )
        + "</nav>",
    ]
    for grade in sorted({r["grade"] for r in rows}):
        total = next(g for g in schedule["grades"] if g["grade"] == grade)
        blocks.append(
            f"<section id='grade-{grade}'><h2>Grade {grade}</h2><p>Proposed core: {total['required_minutes'] / 60:.2f} hours. Scenario capacity: {total['capacity_minutes'] / 60:.0f} hours. Unallocated capacity: {total['unused_minutes'] / 60:.2f} hours. These are planning assumptions, not official hours.</p>"
        )
        unit = None
        for row in [r for r in rows if r["grade"] == grade]:
            if row["unit_id"] != unit:
                unit = row["unit_id"]
                blocks.append(f"<h3>{esc(row['unit_title'])}</h3><p>{esc(row['progression'])}</p>")
            blocks.append(
                f"<article><h4>{esc(row['statement'])}</h4><p class='meta'>{esc(row['micro_skill_id'])} · {esc(row['strand'])} · {esc(row['min_tier'])} · {row['estimated_minutes']} instruction/practice minutes</p>"
            )
            for label, key in [
                ("Why this learning matters", "pedagogical_rationale"),
                ("Proposed grade placement rationale", "grade_rationale"),
                ("Evidence of mastery", "evidence_of_mastery"),
                ("Assessment task", "assessment_task"),
                ("Access and participation", "accessibility_support"),
                ("T0 alternative and limits", "t0_alternative"),
                ("Evidence limitations", "evidence_limitations"),
                ("Safety", "safety_notes"),
                ("Teacher preparation level (0–3)", "teacher_prep_level"),
            ]:
                blocks.append(
                    f"<p><strong>{label}:</strong> {esc('Not applicable' if row.get(key) is None else row[key])}</p>"
                )
            for label, key in [
                ("Success criteria", "success_criteria"),
                ("Teaching sequence", "teaching_sequence"),
                ("Misconceptions", "misconceptions"),
                ("Required resources", "required_resources"),
                ("External knowledge to verify", "external_knowledge"),
                ("Unit limitations", "unit_limitations"),
                ("Publication corrections after model review", "publication_changes"),
            ]:
                blocks.append(
                    f"<strong>{label}</strong><ul>"
                    + "".join(f"<li>{esc(v)}</li>" for v in row[key])
                    + "</ul>"
                )
            relevant_issues = [
                i
                for i in row["outstanding_review_issues"]
                if i.get("skill_slug") in (None, row["slug"])
            ]
            if relevant_issues:
                blocks.append(
                    "<details><summary>Uncalibrated model critique: points to review</summary><ul>"
                    + "".join(
                        f"<li>{esc(i['severity'])}: {esc(i['finding'])} Suggested correction: {esc(i['correction'])}</li>"
                        for i in relevant_issues
                    )
                    + "</ul></details>"
                )
            blocks.append(
                "<p><strong>Internal prerequisite IDs:</strong> "
                + esc(", ".join(row["prerequisite_ids"]) or "None proposed")
                + "</p>"
            )
            blocks.append(
                "<p><strong>Sources:</strong> "
                + "; ".join(
                    f"<a href='{esc(r['url'])}'>{esc(r['document_id'])}</a> (page {esc(r['page'] or 'HTML')}, {esc(r['section'])}, {esc(r['paragraph_id'])}, original characters {r['start']}–{r['end']})"
                    for r in row["sources"]
                )
                + "</p></article>"
            )
        blocks.append("</section>")
    style = "body{font:16px/1.55 system-ui,sans-serif;max-width:1050px;margin:40px auto;padding:0 24px;color:#172638}h1,h2{color:#133f58}h2{border-top:3px solid #167e83;padding-top:24px}h3{margin-top:32px}article{border:1px solid #ccd8df;border-radius:8px;padding:20px;margin:18px 0}.meta{font-size:12px;overflow-wrap:anywhere;color:#52606d}.notice{background:#fff2cd;padding:16px}a{color:#006478}nav{margin:24px 0}li{margin:4px 0}@media print{body{margin:0;font-size:11pt}article{break-inside:avoid}nav{display:none}a{color:inherit;text-decoration:none}}"
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Technology curriculum proposal</title><style>"
        + style
        + "</style><body>"
        + "".join(blocks)
        + "</body></html>\n"
    )
