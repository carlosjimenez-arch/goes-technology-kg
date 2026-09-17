"""Compare local and Vertex embeddings on frozen provisional evidence labels, not model judges."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml

from goes_tech_kg.config import Settings
from goes_tech_kg.corpus.embed import Embedder
from goes_tech_kg.llm.embeddings import VertexEmbeddings
from goes_tech_kg.llm.vertex import gcloud_token_provider
from goes_tech_kg.retrieval.release_context import EvidenceIndex, tokens
from goes_tech_kg.schemas.base import canonical_json, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    out = Path("evaluation/retrieval-v2")
    out.mkdir(parents=True, exist_ok=True)
    evidence = EvidenceIndex(Path("data"))
    experiment = json.loads(
        Path("data/processed/prompt_experiments/plan-decomposition-v4.json").read_text()
    )
    cases = {c["id"]: c for c in experiment["cases"]}
    queries = []
    documents = set()
    paths = sorted(p for p in Path("evaluation/golden").glob("*.yaml") if p.name != "protocol.yaml")
    for number, path in enumerate(paths):
        golden = yaml.safe_load(path.read_text())
        allowed = {
            pid for r in cases[golden["case_id"]]["context_refs"] for pid in r["paragraph_ids"]
        }
        for micro in golden["micro_skills"][:2]:
            expected = sorted(
                pid
                for pid in allowed
                if (
                    pid.split("-")[1] in micro["evidence_keys"]
                    or any(
                        re.search(
                            r"(?<![0-9])" + re.escape(key) + r"(?![0-9])",
                            evidence.paragraphs[pid][1]["text"],
                        )
                        for key in micro["evidence_keys"]
                        if re.fullmatch(r"[0-9]+\.[0-9]+", key)
                    )
                )
                and pid in evidence.paragraphs
                and 40 <= len(evidence.paragraphs[pid][1]["text"]) <= 5000
            )
            if expected:
                queries.append(
                    {
                        "id": micro["id"],
                        "query": micro["statement"],
                        "expected": expected,
                        "split": "development" if number < 4 else "holdout",
                        "case_id": golden["case_id"],
                    }
                )
                documents.update(expected)
    # Deterministic unrelated paragraphs provide distractors, including whole sections in other languages.
    for pid, (_chunk, para) in sorted(evidence.paragraphs.items()):
        if len(documents) >= 45:
            break
        if (
            100 <= len(para["text"]) <= 1200
            and pid not in documents
            and len(tokens(para["text"])) > 10
        ):
            documents.add(pid)
    ids = sorted(documents)
    plan = {
        "schema_version": "retrieval-comparison/1.0",
        "label_status": "provisional_preexisting_not_human_reviewed",
        "queries": queries,
        "document_paragraph_ids": ids,
        "selection_rule": "Promote cloud only if holdout MRR exceeds local by 0.05 and recall@3 does not decrease; this smoke test alone cannot establish deployment suitability.",
    }
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("retrieval plan drift")
    plan_path.write_text(canonical_json(plan) + "\n")
    local = Embedder(
        args.model_path, Path("data/processed/retrieval_local_embeddings"), replay=not args.online
    )
    cloud = VertexEmbeddings(
        Path("data/processed/retrieval_vertex_embeddings"),
        str(Settings().gcp_project),
        gcloud_token_provider() if args.online else None,
    )
    cells = []
    for name, embed_document, embed_query in [
        ("local_mpnet", local.embed, local.embed),
        (
            "vertex_gemini",
            lambda text: cloud.embed(text, "RETRIEVAL_DOCUMENT"),
            lambda text: cloud.embed(text, "RETRIEVAL_QUERY"),
        ),
    ]:
        matrix = np.stack([embed_document(evidence.paragraphs[pid][1]["text"]) for pid in ids])
        for query in queries:
            scores = matrix @ embed_query(query["query"])
            ranking = [ids[int(i)] for i in np.lexsort((np.asarray(ids), -scores))]
            ranks = [ranking.index(pid) + 1 for pid in query["expected"]]
            cells.append(
                {
                    "model": name,
                    "id": query["id"],
                    "split": query["split"],
                    "mrr": 1 / min(ranks),
                    "recall_at_3": sum(r <= 3 for r in ranks) / len(ranks),
                    "top_3": ranking[:3],
                }
            )
        print(name, "complete", flush=True)
    summary = []
    for model in ["local_mpnet", "vertex_gemini"]:
        for split in ["development", "holdout"]:
            selected = [c for c in cells if c["model"] == model and c["split"] == split]
            if selected:
                summary.append(
                    {
                        "model": model,
                        "split": split,
                        "queries": len(selected),
                        "mrr": sum(c["mrr"] for c in selected) / len(selected),
                        "recall_at_3": sum(c["recall_at_3"] for c in selected) / len(selected),
                    }
                )
    report = {
        "plan_sha256": digest(plan),
        "cells": cells,
        "summary": summary,
        "limitations": [
            "Small candidate pool and provisional labels; not an estimate of full-corpus retrieval quality.",
            "No independent human retrieval judgments.",
            "Curriculum generation uses pinned evidence packets; experimental embeddings do not silently change its context.",
        ],
    }
    (out / "report.json").write_text(canonical_json(report) + "\n")
    print(canonical_json(summary), flush=True)


if __name__ == "__main__":
    main()
