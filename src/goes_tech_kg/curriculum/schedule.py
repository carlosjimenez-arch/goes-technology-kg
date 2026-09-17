"""Deterministic full-coverage scheduling; resource and host assumptions stay visible."""

from collections import defaultdict
from typing import Any

from goes_tech_kg.schemas.curriculum import BudgetCharge, BudgetLedger, BudgetPolicy
from goes_tech_kg.schemas.release import ReleaseScenario, TeachingUnit, UnitPlan


def topological_order(prerequisites: dict[str, set[str]]) -> tuple[str, ...]:
    if any(not required <= prerequisites.keys() for required in prerequisites.values()):
        raise ValueError("unknown prerequisite")
    pending = {key: set(value) for key, value in prerequisites.items()}
    ordered: list[str] = []
    while pending:
        ready = sorted(k for k, v in pending.items() if not v)
        if not ready:
            raise ValueError("strict prerequisite cycle")
        for key in ready:
            ordered.append(key)
            del pending[key]
        for values in pending.values():
            values.difference_update(ready)
    return tuple(ordered)


def schedule(
    units: tuple[TeachingUnit, ...],
    plans: tuple[UnitPlan, ...],
    ids: dict[tuple[str, str], str],
    scenario: ReleaseScenario,
) -> dict[str, Any]:
    plan_by_id = {p.id: p for p in plans}
    if set(plan_by_id) != {u.unit_id for u in units} or len(units) != len(plans):
        raise ValueError("unit coverage differs from the declared map")
    rows: list[dict[str, Any]] = []
    totals: dict[int, int] = defaultdict(int)
    resources: set[str] = set()
    problems: list[dict[str, Any]] = []
    for unit in sorted(units, key=lambda u: (u.grade, plan_by_id[u.unit_id].order, u.unit_id)):
        by_slug = {m.slug: m for m in unit.micro_skills}
        order = topological_order({m.slug: set(m.prerequisites) for m in unit.micro_skills})
        for slug in order:
            micro = by_slug[slug]
            resources.update(micro.required_resources)
            if int(micro.min_tier.value[-1]) > int(scenario.material_tier[-1]):
                problems.append(
                    {
                        "code": "insufficient_tier",
                        "micro_skill_id": ids[(unit.unit_id, slug)],
                        "required": micro.min_tier.value,
                    }
                )
            if scenario.available_resources is not None:
                missing = sorted(set(micro.required_resources) - set(scenario.available_resources))
                if missing:
                    problems.append(
                        {
                            "code": "missing_resources",
                            "micro_skill_id": ids[(unit.unit_id, slug)],
                            "resources": missing,
                        }
                    )
            rows.append(
                {
                    "grade": unit.grade,
                    "unit_id": unit.unit_id,
                    "micro_skill_id": ids[(unit.unit_id, slug)],
                    "component": "instruction_and_practice",
                    "minutes": micro.estimated_minutes,
                    "start_minute": totals[unit.grade],
                }
            )
            totals[unit.grade] += micro.estimated_minutes
        for component in ("assessment", "setup", "review"):
            minutes = getattr(unit, component + "_minutes")
            rows.append(
                {
                    "grade": unit.grade,
                    "unit_id": unit.unit_id,
                    "micro_skill_id": ids[(unit.unit_id, order[0])],
                    "component": component,
                    "minutes": minutes,
                    "start_minute": totals[unit.grade],
                }
            )
            totals[unit.grade] += minutes
    grades: list[dict[str, Any]] = []
    ledgers = []
    for grade in sorted(totals):
        own_cap = int(scenario.minutes_per_grade * scenario.own_share)
        borrowed_cap = scenario.minutes_per_grade - own_cap
        policy = BudgetPolicy(
            mode=scenario.mode,
            grade=grade,
            own_minutes=own_cap,
            borrowed_minutes_cap=borrowed_cap,
            host_capacities=scenario.host_capacities,
            official_verified=False,
        )
        capacities = dict(scenario.host_capacities)
        own_remaining, borrowed_remaining = own_cap, borrowed_cap
        charges = []
        ledger_rows = []
        for row in (r for r in rows if r["grade"] == grade):
            remaining = row["minutes"]
            own = min(own_remaining, remaining)
            if own:
                charge = BudgetCharge(micro_skill_id=row["micro_skill_id"], minutes=own)
                charges.append(charge)
                ledger_rows.append(
                    {**row, "minutes": own, "host_subject": None, "host_indicator": None}
                )
            own_remaining -= own
            remaining -= own
            for host in sorted(capacities):
                borrowed = min(remaining, capacities[host], borrowed_remaining)
                if borrowed:
                    charges.append(
                        BudgetCharge(
                            micro_skill_id=row["micro_skill_id"],
                            minutes=borrowed,
                            host_subject=host,
                            host_indicator=scenario.host_indicator,
                        )
                    )
                    ledger_rows.append(
                        {
                            **row,
                            "minutes": borrowed,
                            "start_minute": row["start_minute"] + row["minutes"] - remaining,
                            "host_subject": host,
                            "host_indicator": scenario.host_indicator,
                        }
                    )
                remaining -= borrowed
                capacities[host] -= borrowed
                borrowed_remaining -= borrowed
            if remaining:
                problems.append(
                    {
                        "code": "insufficient_time_or_host_capacity",
                        "grade": grade,
                        "unit_id": row["unit_id"],
                        "unallocated_minutes": remaining,
                    }
                )
        ledger = BudgetLedger(policy=policy, charges=tuple(charges))
        if sum(c.minutes for c in charges) != totals[grade]:
            # Partial assignment is diagnostic only, never a valid curriculum allocation.
            status = "infeasible"
        else:
            status = "capacity_feasible"
        grades.append(
            {
                "grade": grade,
                "required_minutes": totals[grade],
                "capacity_minutes": scenario.minutes_per_grade,
                "unused_minutes": max(0, scenario.minutes_per_grade - totals[grade]),
                "status": status,
            }
        )
        ledgers.append(
            {
                "grade": grade,
                "validated_ledger": ledger.model_dump(mode="json"),
                "rows": ledger_rows,
            }
        )
    return {
        "schema_version": "curriculum-schedule/1.0",
        "scenario": scenario.model_dump(mode="json"),
        "status": "infeasible" if problems else "capacity_feasible",
        "deployable": False,
        "resource_verification": "not_verified"
        if scenario.available_resources is None
        else "explicit_inventory",
        "required_resources": sorted(resources),
        "grades": grades,
        "schedule": rows,
        "ledgers": ledgers,
        "diagnostics": problems,
    }
