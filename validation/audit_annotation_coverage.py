#!/usr/bin/env python3
"""Audit annotation-to-source coverage without assigning calls to missing inputs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def source_digest(path: Path) -> str:
    """Identify the CSV content identically before and after gzip packaging."""
    digest = hashlib.sha256()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analyze(records_path: Path, output: Path) -> pd.DataFrame:
    records = pd.read_csv(records_path, low_memory=False)
    records = records.loc[
        records.configuration.eq("early_10_90ms") & records.reference_positive.notna()
    ].copy()
    if records.empty or records.duplicated(["subject", "stimpair", "channel"]).any():
        raise ValueError("Expected a nonempty, unique early-configuration reference frame")
    reasons = records.eligibility_reasons.fillna("")
    bad_contact = reasons.str.contains(
        "recording_contact_not_good_ecog|stimulation_contact_not_good_ecog|"
        "(?:^|;)stimulation_contact(?:;|$)", regex=True,
    )
    categories = [
        "contact_or_stimulation_ineligible", "no_matching_source_events",
        "fewer_than_five_source_events", "spatially_ineligible",
        "documented_raw_loss", "detector_unavailable", "evaluable",
    ]
    conditions = [
        bad_contact, records.source_event_count.eq(0),
        records.source_event_count.between(1, 4), ~records.spatial_eligible,
        records.prediction_status.eq("missing_raw_no_prediction"),
        records.prediction_status.eq("detector_unavailable"),
        records.prediction_status.eq("evaluable"),
    ]
    records["flow_outcome"] = np.select(conditions, categories, default="unresolved")
    if records.flow_outcome.eq("unresolved").any():
        raise ValueError("Unresolved source/eligibility outcome; inspect the record join")

    results = []
    for cohort, subset in [
        ("independent13", records.loc[records.subject.ne("MAYO01")]),
        ("replication14", records),
    ]:
        groups = [("overall", "all", subset)]
        groups += [("site", str(key), group) for key, group in subset.groupby("site")]
        groups += [("subject", str(key), group) for key, group in subset.groupby("subject")]
        for level, name, group in groups:
            row = {
                "cohort": cohort, "level": level, "group": name,
                "n_participants": group.subject.nunique(),
                "valid_labeled_records": len(group),
            }
            row.update({key: int(group.flow_outcome.eq(key).sum()) for key in categories})
            row["source_eligible"] = int(group.source_eligible.sum())
            row["source_and_site_eligible"] = int(group.scoring_eligible.sum())
            eligible = row["source_and_site_eligible"]
            row["conditional_evaluable_fraction"] = row["evaluable"] / eligible if eligible else np.nan
            # Add only records whose sole exclusion is absence of source events.
            missing = group.loc[group.eligibility_reasons.eq("no_good_source_events_for_pair")]
            row["extra_sole_source_absent_records"] = len(missing)
            row["extra_sole_source_absent_pairs"] = len(missing[["subject", "stimpair"]].drop_duplicates())
            row["extra_sole_source_absent_n1_weight"] = float(missing.reference_positive.sum())
            row["extra_sole_source_absent_single_rater_records"] = int(missing.n_raters.eq(1).sum())
            expanded = eligible + len(missing)
            row["expanded_coverage_frame"] = expanded
            row["expanded_evaluable_fraction"] = row["evaluable"] / expanded if expanded else np.nan
            if sum(row[key] for key in categories) != len(group):
                raise ValueError("The mutually exclusive flow does not partition the reference frame")
            if row["documented_raw_loss"] + row["detector_unavailable"] + row["evaluable"] != eligible:
                raise ValueError("Evaluability outcomes disagree with the scoring denominator")
            results.append(row)

    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(results)
    table.to_csv(output / "annotation_source_flow.csv", index=False)
    missing = records.loc[records.eligibility_reasons.eq("no_good_source_events_for_pair")]
    missing.groupby(["subject", "site", "stimpair"], as_index=False).agg(
        labeled_records=("channel", "size"),
        n1_reference_weight=("reference_positive", "sum"),
        minimum_available_raters=("n_raters", "min"),
        maximum_available_raters=("n_raters", "max"),
    ).to_csv(output / "source_event_absent_pairs.csv", index=False)
    overall = table.loc[table.level.eq("overall")]
    report = {
        "source_sha256": source_digest(records_path),
        "configuration": "one copy per channel/pair (early configuration), finite reference labels",
        "hierarchy": categories,
        "hierarchy_note": "Apply mutually exclusive outcomes in listed order; underlying overlapping reasons remain unchanged in record tables.",
        "interpretation": "95.33% is conditional coverage after source and site eligibility. Expanded coverage adds only otherwise eligible labels lacking matching source events; it is a feasibility audit, not detector accuracy. No missing prediction is assigned an observed negative result.",
        "paper_scope": "The published ER-detect table reports 9 UMCU20 pairs and 697 shared-rater records, consistent with its narrow released-source slice. Extra annotated pairs have no verified matching input in the pinned ds004774 recordings.",
        "overall": overall.to_dict("records"),
    }
    (output / "annotation_source_flow.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(overall.to_string(index=False))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.records, args.output)
