#!/usr/bin/env python3
"""Score frozen ERPy predictions against individual released ER-detect labels.

No detector is fitted here, and BH is never recomputed after a label join.
The primary pooled target is a uniformly chosen available rater within each
record: each channel/pair contributes total weight one, regardless of raters.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

KEY = ["subject", "stimpair", "channel"]
CONFIGURATIONS = ("early_10_90ms", "broad_15_300ms")
METRICS = ("sensitivity", "specificity", "balanced_accuracy", "ppv", "npv", "accuracy", "kappa", "coverage")
COUNTS = ("tp", "fn", "tn", "fp", "available", "eligible")
DEVELOPMENT_OVERLAP_SUBJECTS = frozenset({"MAYO01"})


def canonical_channel(value: str) -> str:
    return str(value).strip().upper()


def canonical_pair(value: str) -> str:
    parts = [canonical_channel(part) for part in str(value).split("-")]
    if len(parts) != 2 or not all(parts) or parts[0] == parts[1]:
        raise ValueError(f"Invalid stimulation pair: {value!r}")
    return "-".join(sorted(parts))


def subject_of(value: str) -> str:
    match = re.search(r"(?:sub-)?((?:MAYO|UMCU)\d+)", str(value), re.I)
    if not match:
        raise ValueError(f"Unrecognized subject: {value!r}")
    return match[1].upper()


def site_of(subject: str) -> str:
    return "Mayo" if subject.startswith("MAYO") else "UMCU"


def canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["subject"] = frame.subject.map(subject_of)
    frame["channel"] = frame.channel.map(canonical_channel)
    frame["stimpair"] = frame.stimpair.map(canonical_pair)
    return frame


def require_unique(frame: pd.DataFrame, keys: list[str], description: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"Duplicate canonical {description}; refusing a many-to-many join")


def binary_annotation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if (~np.isin(values[np.isfinite(values)], [-1, 0, 1, 2])).any() or np.isinf(values).any():
        raise ValueError("Unknown annotation code")
    return np.where(values == 1, 1., np.where(np.isin(values, [0, 2]), 0., np.nan))


def labeled_matrix(values, n_channels: int, n_pairs: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    expected = (n_channels, n_pairs)
    # scipy simplify_cells squeezes singleton axes; do not silently reshape
    # an actual transposed multi-channel, multi-pair matrix.
    if array.shape != expected and n_channels != 1 and n_pairs != 1:
        raise ValueError(f"Expected channel-by-pair matrix {expected}, got {array.shape}")
    return array.reshape(expected)


def load_annotations(directory: Path, subjects: set[str] | None = None) -> tuple[pd.DataFrame, list[Path]]:
    rows, files = [], []
    for path in sorted(directory.glob("*_annot_*.mat")):
        dataset, rater = path.stem.split("_annot_", 1)
        subject = subject_of(dataset)
        if subjects and subject not in subjects:
            continue
        data = loadmat(path, simplify_cells=True)["annot"]
        channels = np.atleast_1d(data["channels"]).astype(str)
        pairs = np.atleast_1d(data["stimpairs"]).astype(str)
        values = labeled_matrix(data["annotations"], len(channels), len(pairs))
        binary = binary_annotation(values)
        for i, channel in enumerate(channels):
            for j, pair in enumerate(pairs):
                raw = values[i, j]
                rows.append({"subject": subject, "dataset": dataset, "rater": rater,
                             "channel": channel, "stimpair": pair, "raw_annotation": raw,
                             "reference_positive": binary[i, j],
                             "annotation_status": "stimulation_contact" if raw == -1 else "unannotated" if np.isnan(raw) else "labeled",
                             "annotation_file": path.name,
                             "in_published_25_file_list": not (subject == "UMCU21" and rater == "SB")})
        files.append(path)
    frame = canonicalize(pd.DataFrame(rows))
    require_unique(frame, KEY + ["rater"], "annotation records")
    return frame, files


def load_archived(directory: Path, subjects: set[str] | None = None) -> tuple[pd.DataFrame, list[Path], dict]:
    rows, files, configs = [], [], {}
    for path in sorted(directory.glob("*/erdetect_data.mat")):
        subject = subject_of(path.parent.name)
        if subjects and subject not in subjects:
            continue
        data = loadmat(path, simplify_cells=True, variable_names=[
            "channel_labels", "stimpair_labels", "neg_peak_amplitudes", "config"])
        channels = np.atleast_1d(data["channel_labels"]).astype(str)
        pairs = np.atleast_1d(data["stimpair_labels"]).astype(str)
        amplitudes = labeled_matrix(data["neg_peak_amplitudes"], len(channels), len(pairs))
        if np.isinf(amplitudes).any():
            raise ValueError("Infinite archived amplitude is neither a valid call nor the NaN no-call marker")
        for i, channel in enumerate(channels):
            for j, pair in enumerate(pairs):
                rows.append({"subject": subject, "channel": channel, "stimpair": pair,
                             "archived_record_present": True,
                             "archived_detected": bool(np.isfinite(amplitudes[i, j])),
                             "archived_negative_peak_amplitude": amplitudes[i, j]})
        configs[subject] = data["config"]
        files.append(path)
    frame = canonicalize(pd.DataFrame(rows))
    require_unique(frame, KEY, "archived records")
    return frame, files, configs


def parse_boolean(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError(f"Invalid or missing Boolean in {series.name}")
    return mapped.astype(bool)


def load_predictions(directory: Path, subjects: set[str] | None = None) -> tuple[pd.DataFrame, list[Path]]:
    frames, files = [], []
    for path in sorted(directory.glob("*/*.csv")):
        if path.name.endswith(".partial.csv") or (subjects and subject_of(path.parent.name) not in subjects):
            continue
        frame = pd.read_csv(path, dtype={"random_seed": str})
        if frame.empty:
            continue
        frame = canonicalize(frame)
        if set(frame.subject) != {subject_of(path.parent.name)}:
            raise ValueError("Prediction subject disagrees with its source directory")
        for name in ["evaluable", "detected"]:
            frame[name] = parse_boolean(frame[name])
        q = pd.to_numeric(frame.q_joint, errors="coerce")
        if not np.array_equal(frame.evaluable, np.isfinite(q)):
            raise ValueError("Prediction evaluability disagrees with finite archived q_joint")
        if (frame.detected & ~frame.evaluable).any() or not np.array_equal(frame.detected, q.le(.05)):
            raise ValueError("Prediction calls disagree with the prespecified q_joint <= .05 rule")
        if not frame.configuration.isin(CONFIGURATIONS).all():
            raise ValueError("Unrecognized frozen detector configuration")
        frame["prediction_present"] = True
        frames.append(frame)
        files.append(path)
    if not frames:
        raise ValueError("No complete prediction files found")
    result = pd.concat(frames, ignore_index=True)
    require_unique(result, KEY + ["configuration"], "predictions")
    return result, files


def load_source_metadata(directory: Path, data_directory: Path, subjects: set[str]) -> tuple[dict, list[Path]]:
    metadata, files = {}, []
    for subject in sorted(subjects):
        channel_paths = list((directory / f"sub-{subject}").rglob("*_channels.tsv"))
        if len(channel_paths) != 1:
            raise ValueError(f"Expected one source acquisition for {subject}")
        channel_path = channel_paths[0]
        event_path = channel_path.with_name(channel_path.name.replace("_channels.tsv", "_events.tsv"))
        channels, events = pd.read_csv(channel_path, sep="\t"), pd.read_csv(event_path, sep="\t")
        good = set(channels.loc[channels.type.str.upper().eq("ECOG") & channels.status.str.lower().eq("good"), "name"].map(canonical_channel))
        events = events.loc[events.trial_type.eq("electrical_stimulation")]
        if "status" in events:
            events = events.loc[events.status.str.lower().eq("good")]
        counts = events.electrical_stimulation_site.map(canonical_pair).value_counts().to_dict()
        coordinates = None
        if site_of(subject) == "Mayo":
            paths = list((data_directory / f"sub-{subject}").rglob("*_electrodes.tsv"))
            if len(paths) != 1:
                raise ValueError(f"Mayo spatial analysis requires one electrode table for {subject}")
            coord_path = paths[0].with_name(paths[0].name.replace("_electrodes.tsv", "_coordsystem.json"))
            coordinate_info = json.loads(coord_path.read_text())
            if coordinate_info.get("iEEGCoordinateUnits", "").lower() != "mm":
                raise ValueError("Spatial matching requires verified millimetre coordinates")
            table = pd.read_csv(paths[0], sep="\t")
            coordinates = {canonical_channel(row["name"]): np.asarray([row.x, row.y, row.z], dtype=float) for _, row in table.iterrows()}
            files.extend([paths[0], coord_path])
        metadata[subject] = {"good_channels": good, "pair_event_counts": counts, "coordinates": coordinates}
        files.extend([channel_path, event_path])
    return metadata, files


def minimum_stimulation_distance(channel: str, pair: str, coordinates: dict | None) -> float:
    if coordinates is None:
        return np.nan
    points = [coordinates.get(name, np.full(3, np.nan)) for name in [channel, *pair.split("-")]]
    if not np.isfinite(points).all():
        return np.nan
    return float(min(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[0] - points[2])))


def record_universe(labels: pd.DataFrame, predictions: pd.DataFrame, archived: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    expected = [{"subject": subject, "stimpair": pair, "channel": channel}
                for subject, info in metadata.items() for pair in info["pair_event_counts"]
                for channel in sorted(info["good_channels"])]
    keys = pd.concat([labels[KEY], predictions[KEY], archived[KEY], pd.DataFrame(expected)], ignore_index=True).drop_duplicates()
    details = []
    for row in keys.itertuples(index=False):
        info = metadata[row.subject]
        good_recording = row.channel in info["good_channels"]
        good_stimulation = set(row.stimpair.split("-")).issubset(info["good_channels"])
        is_stim = row.channel in row.stimpair.split("-")
        count = int(info["pair_event_counts"].get(row.stimpair, 0))
        distance = minimum_stimulation_distance(row.channel, row.stimpair, info["coordinates"])
        source_eligible = good_recording and good_stimulation and not is_stim and count >= 5
        spatial_eligible = site_of(row.subject) == "UMCU" or (np.isfinite(distance) and distance >= 12.)
        reasons = []
        if not good_recording: reasons.append("recording_contact_not_good_ecog")
        if not good_stimulation: reasons.append("stimulation_contact_not_good_ecog")
        if is_stim: reasons.append("stimulation_contact")
        if count == 0: reasons.append("no_good_source_events_for_pair")
        elif count < 5: reasons.append("fewer_than_five_good_source_events")
        if site_of(row.subject) == "Mayo" and not np.isfinite(distance): reasons.append("missing_mayo_coordinates")
        elif not spatial_eligible: reasons.append("within_12mm_of_stimulation_contact")
        details.append({"site": site_of(row.subject), "source_event_count": count,
                        "source_eligible": source_eligible, "spatial_eligible": spatial_eligible,
                        "scoring_eligible": source_eligible and spatial_eligible,
                        "minimum_stimulation_distance_mm": distance,
                        "eligibility_scope": "mayo_12mm" if site_of(row.subject) == "Mayo" else "umcu_no_coordinates",
                        "eligibility_reasons": ";".join(reasons)})
    keys = pd.concat([keys.reset_index(drop=True), pd.DataFrame(details)], axis=1)
    require_unique(keys, KEY, "universe records")
    return keys


def aggregate_reference(labels: pd.DataFrame) -> pd.DataFrame:
    valid = labels.loc[labels.reference_positive.notna()]
    result = valid.groupby(KEY, as_index=False).agg(
        reference_positive=("reference_positive", "mean"), n_raters=("rater", "size"),
        minimum_vote=("reference_positive", "min"), maximum_vote=("reference_positive", "max"))
    result["strict_consensus"] = result.n_raters.ge(2) & result.minimum_vote.eq(result.maximum_vote)
    result["rater_disagreement"] = result.minimum_vote.ne(result.maximum_vote)
    return result


def join_predictions(universe: pd.DataFrame, predictions: pd.DataFrame, archived: pd.DataFrame) -> pd.DataFrame:
    configured = universe.merge(pd.DataFrame({"configuration": CONFIGURATIONS}), how="cross")
    keep = predictions.drop(columns=["site"], errors="ignore")
    result = configured.merge(keep, on=KEY + ["configuration"], how="left", validate="one_to_one")
    result = result.merge(archived, on=KEY, how="left", validate="many_to_one")
    for name in ["prediction_present", "evaluable", "detected", "archived_record_present", "archived_detected"]:
        result[name] = result[name].astype("boolean").fillna(False).astype(bool)
    result["prediction_status"] = np.where(~result.prediction_present, "no_prediction",
                                           np.where(result.evaluable, "evaluable", "detector_unavailable"))
    result["archived_pair_comparable"] = True
    result["archived_pair_exclusion_reason"] = ""
    result["declared_missing_raw"] = False
    result["missing_raw_reason"] = ""
    return result


def apply_pair_manifest(joined: pd.DataFrame, manifest: pd.DataFrame | None = None, *, allow_incomplete: bool = False) -> pd.DataFrame:
    """Separate documented source loss from incomplete computation.

    Optional manifest: subject, stimpair, archived_pair_comparable,
    declared_missing_raw, reason. The two Boolean fields are independent:
    truncated-but-recovered data may be scored against raters while remaining
    incomparable with the historical detector's different trial set.
    """
    result = joined.copy()
    if manifest is not None and not manifest.empty:
        manifest = manifest.copy()
        if "actual_missing_raw" in manifest and "declared_missing_raw" not in manifest:
            manifest["declared_missing_raw"] = manifest.actual_missing_raw
        manifest["subject"] = manifest.subject.map(subject_of)
        manifest["stimpair"] = manifest.stimpair.map(canonical_pair)
        require_unique(manifest, ["subject", "stimpair"], "pair provenance manifest")
        for _, row in manifest.iterrows():
            mask = result.subject.eq(row.subject) & result.stimpair.eq(row.stimpair)
            if "archived_pair_comparable" in manifest and pd.notna(row.archived_pair_comparable):
                comparable = parse_boolean(pd.Series([row.archived_pair_comparable])).iloc[0]
                result.loc[mask, "archived_pair_comparable"] = comparable
                if not comparable: result.loc[mask, "archived_pair_exclusion_reason"] = str(row.reason)
            if "declared_missing_raw" in manifest and pd.notna(row.declared_missing_raw):
                missing = parse_boolean(pd.Series([row.declared_missing_raw])).iloc[0]
                result.loc[mask, "declared_missing_raw"] = missing
                if missing: result.loc[mask, "missing_raw_reason"] = str(row.reason)
    missing = result.source_eligible & ~result.prediction_present
    incomplete = missing & ~result.declared_missing_raw
    result.loc[missing & result.declared_missing_raw, "prediction_status"] = "missing_raw_no_prediction"
    result.loc[incomplete, "prediction_status"] = "incomplete_processing_no_prediction"
    if incomplete.any() and not allow_incomplete:
        pairs = result.loc[incomplete, ["subject", "stimpair", "configuration"]].drop_duplicates()
        raise ValueError(f"Incomplete predictions for {len(pairs)} source-eligible pair/configuration groups; finish processing or document missing raw data before final scoring")
    return result


def metrics_from_counts(counts: np.ndarray) -> dict[str, np.ndarray]:
    c = np.asarray(counts, dtype=float)
    tp, fn, tn, fp, available, eligible = np.moveaxis(c, -1, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sensitivity, specificity = tp / (tp + fn), tn / (tn + fp)
        total = tp + fn + tn + fp
        accuracy = (tp + tn) / total
        expected = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / total**2
        return {"sensitivity": sensitivity, "specificity": specificity,
                "balanced_accuracy": (sensitivity + specificity) / 2,
                "ppv": tp / (tp + fp), "npv": tn / (tn + fn), "accuracy": accuracy,
                "kappa": np.where(np.isclose(expected, 1., atol=1e-14, rtol=0), np.nan, (accuracy - expected) / (1 - expected)),
                "coverage": available / eligible}


def confusion_contributions(frame: pd.DataFrame, calls: np.ndarray, available: np.ndarray, available_case: bool) -> pd.DataFrame:
    y = frame.reference_positive.to_numpy(dtype=float)
    call, available = np.asarray(calls, bool), np.asarray(available, bool)
    if not np.isfinite(y).all() or ((y < 0) | (y > 1)).any():
        raise ValueError("Reference probabilities must be finite and in [0, 1]")
    scored = available if available_case else np.ones(len(frame), dtype=bool)
    call = call & available
    result = frame[["subject", "site"]].copy()
    result["tp"], result["fn"] = y * call * scored, y * ~call * scored
    result["tn"], result["fp"] = (1 - y) * ~call * scored, (1 - y) * call * scored
    result["available"], result["eligible"] = available.astype(float), 1.
    return result


def subject_bootstrap(contributions: pd.DataFrame, resamples: int = 2000, seed: int = 20260907) -> dict:
    """Resample whole subjects within sites using their complete count vectors."""
    subject = contributions.groupby(["site", "subject"], sort=True)[list(COUNTS)].sum()
    if subject.index.get_level_values("subject").duplicated().any():
        raise ValueError("A subject cannot belong to more than one site")
    if len(subject) < 2:
        return {name: {"low": np.nan, "high": np.nan, "valid_resamples": 0} for name in METRICS}
    rng = np.random.default_rng(seed)
    draws = np.zeros((resamples, len(COUNTS)))
    for _, group in subject.groupby(level="site", sort=True):
        vectors = group.to_numpy()
        indices = rng.integers(0, len(vectors), size=(resamples, len(vectors)))
        draws += vectors[indices].sum(axis=1)
    result = {}
    for name, values in metrics_from_counts(draws).items():
        finite = values[np.isfinite(values)]
        low, high = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
        result[name] = {"low": float(low), "high": float(high), "valid_resamples": len(finite)}
    return result


def summarize_contributions(contributions: pd.DataFrame, identity: dict, resamples: int, seed: int) -> list[dict]:
    groups = [("overall", "all", contributions)]
    groups += [("site", str(key), group) for key, group in contributions.groupby("site", sort=True)]
    groups += [("subject", str(key), group) for key, group in contributions.groupby("subject", sort=True)]
    rows = []
    for level, name, group in groups:
        totals = group[list(COUNTS)].sum().to_numpy()
        point = metrics_from_counts(totals)
        row = dict(identity, level=level, group=name, n_subjects=group.subject.nunique(),
                   subjects=";".join(sorted(group.subject.unique())), n_sites=group.site.nunique(),
                   n_subjects_with_scored_records=int(group.groupby("subject")[list(COUNTS[:4])].sum().sum(axis=1).gt(0).sum()),
                   n_scored_records=float(totals[:4].sum()),
                   eligibility_scope="mixed_site_eligibility" if group.site.nunique() > 1 else "mayo_12mm" if group.site.iloc[0] == "Mayo" else "umcu_no_coordinates")
        row.update(zip(COUNTS, totals))
        row.update({key: float(value) for key, value in point.items()})
        ci = subject_bootstrap(group, resamples, seed) if level != "subject" else {}
        for metric in METRICS:
            for field in ["low", "high", "valid_resamples"]:
                row[f"{metric}_{field}"] = ci.get(metric, {}).get(field, 0 if field == "valid_resamples" else np.nan)
        rows.append(row)
    return rows


def score_reference(frame: pd.DataFrame, identity: dict, resamples: int, seed: int) -> list[dict]:
    rows = []
    for configuration, group in frame.groupby("configuration", sort=True):
        for paired in [False, True]:
            base = group.loc[group.archived_record_present & group.archived_pair_comparable] if paired else group
            if base.empty:
                continue
            for available_case in [True, False]:
                available = base.evaluable.to_numpy()
                contribution = confusion_contributions(base, base.detected.to_numpy(), available, available_case)
                common = dict(identity, configuration=configuration,
                              comparison="paired_archived_intersection" if paired else "erpy_all_source_eligible",
                              missingness_policy="available_case" if available_case else "unavailable_as_no_call")
                rows += summarize_contributions(contribution, dict(common, method="ERPy"), resamples, seed)
                if paired:
                    # Exactly the same evaluated records as the corresponding
                    # ERPy policy. NaN archived peak amplitude already means no call.
                    # Keep zero-scored subjects in the same resampling cohort
                    # as ERPy. Filtering them out would change bootstrap draws.
                    comparison_available = available if available_case else np.ones(len(base), bool)
                    contribution = confusion_contributions(base, base.archived_detected.to_numpy(),
                                                           comparison_available, available_case)
                    if available_case:
                        # Historical coverage is conditional on paired rows;
                        # the joint-selection coverage is shown by the ERPy row.
                        contribution["eligible"] = comparison_available.astype(float)
                    rows += summarize_contributions(contribution, dict(common, method="archived_ER_detect"), resamples, seed)
    return rows


def cohort_frames(frame: pd.DataFrame):
    """Provenance-defined cohorts; never selected from performance or labels."""
    overlap = frame.subject.isin(DEVELOPMENT_OVERLAP_SUBJECTS)
    for scope, mask in [
        ("independent_external_excluding_development_overlap", ~overlap),
        ("all_recovered_replication", np.ones(len(frame), dtype=bool)),
        ("development_overlap_only", overlap),
    ]:
        selected = frame.loc[mask]
        if not selected.empty:
            yield scope, selected


def score_reference_cohorts(frame: pd.DataFrame, identity: dict, resamples: int, seed: int) -> list[dict]:
    rows = []
    for scope, group in cohort_frames(frame):
        rows += score_reference(group, dict(identity, cohort_scope=scope), resamples, seed)
    return rows


def interrater_agreement(labels: pd.DataFrame, universe: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    eligible = labels.merge(universe[KEY + ["site", "scoring_eligible"]], on=KEY, validate="many_to_one")
    eligible = eligible.loc[eligible.scoring_eligible & eligible.reference_positive.notna()]
    rows = []
    for first, second in combinations(sorted(eligible.rater.unique()), 2):
        a = eligible.loc[eligible.rater.eq(first), KEY + ["site", "reference_positive"]]
        b = eligible.loc[eligible.rater.eq(second), KEY + ["reference_positive"]].rename(columns={"reference_positive": "second_vote"})
        common = a.merge(b, on=KEY, validate="one_to_one")
        if common.empty:
            continue
        contribution = confusion_contributions(common, common.second_vote.to_numpy(), np.ones(len(common), bool), False)
        rows += summarize_contributions(contribution, {"rater_a": first, "rater_b": second,
            "reference_scope": "shared_labeled_records_after_source_and_site_specific_spatial_eligibility"}, resamples, seed)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    discard = [column for column in result if any(column == metric or column.startswith(metric + "_")
               for metric in METRICS if metric not in {"accuracy", "kappa"})]
    result = result.drop(columns=discard)
    return result.rename(columns={name: name.replace("accuracy", "agreement") for name in result if name.startswith("accuracy")})


def json_ready(value):
    if isinstance(value, dict): return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.bool_,)): return bool(value)
    return value


def execute(root: Path, output: Path, subjects: set[str] | None = None, resamples: int = 2000, seed: int = 20260907,
            pair_manifest: Path | None = None, allow_incomplete: bool = False) -> dict:
    labels, annotation_files = load_annotations(root / "source_metadata/dataset/derivatives/annots", subjects)
    predictions, prediction_files = load_predictions(root / "predictions", subjects)
    selected = set(labels.subject)
    if not set(predictions.subject).issubset(selected):
        raise ValueError("Prediction subjects lack released annotation files")
    archived, archived_files, archived_config = load_archived(root / "data/ds004774/derivatives/app_detect_output", selected)
    metadata, metadata_files = load_source_metadata(root / "source_metadata/dataset", root / "data/ds004774", selected)
    universe = record_universe(labels, predictions, archived, metadata)
    universe["development_overlap"] = universe.subject.isin(DEVELOPMENT_OVERLAP_SUBJECTS)
    universe["independent_external_subject"] = ~universe.development_overlap
    joined = join_predictions(universe, predictions, archived)
    manifest_frame = pd.read_csv(pair_manifest) if pair_manifest else None
    joined = apply_pair_manifest(joined, manifest_frame, allow_incomplete=allow_incomplete)
    references = aggregate_reference(labels)
    pooled = joined.merge(references, on=KEY, how="left", validate="many_to_one")
    scored = pooled.loc[pooled.scoring_eligible & pooled.reference_positive.notna()]
    summaries = score_reference_cohorts(scored, {"reference_view": "equal_record_available_rater", "rater": ""}, resamples, seed)
    summaries += score_reference_cohorts(scored.loc[scored.strict_consensus.eq(True)],
                                 {"reference_view": "strict_consensus_at_least_two", "rater": ""}, resamples, seed)
    rater_join = labels.merge(joined, on=KEY, how="left", validate="many_to_many")
    for rater, group in rater_join.loc[rater_join.scoring_eligible & rater_join.reference_positive.notna()].groupby("rater", sort=True):
        summaries += score_reference_cohorts(group, {"reference_view": "individual_rater", "rater": str(rater)}, resamples, seed)
    # Preserve the historical 25-file comparison as a clearly secondary view.
    paper_reference = aggregate_reference(labels.loc[labels.in_published_25_file_list])
    paper = joined.merge(paper_reference, on=KEY, how="left", validate="many_to_one")
    summaries += score_reference_cohorts(paper.loc[paper.scoring_eligible & paper.reference_positive.notna()],
                                 {"reference_view": "secondary_published_25_file_list", "rater": ""}, resamples, seed)
    summary = pd.DataFrame(summaries)
    agreement_frames = []
    for scope, group in cohort_frames(labels):
        agreement = interrater_agreement(group, universe, resamples, seed)
        if not agreement.empty:
            agreement["cohort_scope"] = scope
            agreement_frames.append(agreement)
    interrater = pd.concat(agreement_frames, ignore_index=True) if agreement_frames else pd.DataFrame()
    denominator_columns = ["subject", "site", "configuration", "source_eligible", "spatial_eligible",
                           "scoring_eligible", "eligibility_reasons", "prediction_status", "archived_record_present", "archived_pair_comparable"]
    pooled["reference_status"] = np.where(pooled.reference_positive.notna(), "labeled", "no_valid_rater")
    exclusions = pooled.groupby(denominator_columns + ["reference_status"], dropna=False).size().rename("n_unique_records").reset_index()
    label_exclusions = rater_join.groupby(["subject", "rater", "configuration", "annotation_status", "scoring_eligible", "prediction_status"], dropna=False).size().rename("n_rater_records").reset_index()
    availability = scored.groupby(["subject", "site", "configuration", "prediction_status"], dropna=False).agg(
        n_records=("channel", "size"), reference_positive_weight=("reference_positive", "sum"),
        n_distinct_pairs=("stimpair", "nunique")).reset_index()
    availability["reference_negative_weight"] = availability.n_records - availability.reference_positive_weight
    annotation_calls = rater_join.loc[rater_join.scoring_eligible & rater_join.reference_positive.notna()].copy()
    annotation_calls = annotation_calls.merge(references[KEY + ["n_raters"]], on=KEY, validate="many_to_one")
    annotation_calls["record_normalized_rater_weight"] = 1. / annotation_calls.n_raters
    annotation_discordance = annotation_calls.groupby(
        ["subject", "site", "configuration", "rater", "raw_annotation", "prediction_status", "detected"], dropna=False
    ).agg(n_rater_rows=("channel", "size"), record_normalized_rater_weight=("record_normalized_rater_weight", "sum")).reset_index()
    files = annotation_files + prediction_files + archived_files + metadata_files
    if pair_manifest: files.append(pair_manifest)
    overlap_path = root / "source_metadata/development_overlap.json"
    if overlap_path.exists(): files.append(overlap_path)
    manifest = [{"path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                 "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except subprocess.SubprocessError: commit = None
    report = {
        "subjects": sorted(selected), "n_subjects": len(selected), "n_sites": len({site_of(s) for s in selected}),
        "cohorts": {"primary_independent_subjects": sorted(selected - DEVELOPMENT_OVERLAP_SUBJECTS),
                    "n_primary_independent_subjects": len(selected - DEVELOPMENT_OVERLAP_SUBJECTS),
                    "all_recovered_replication_subjects": sorted(selected),
                    "development_overlap_subjects": sorted(selected & DEVELOPMENT_OVERLAP_SUBJECTS)},
        "cohort_amendment": "On 2026-09-07, after the five-subject Mayo pilot had been scored and before UMCU accuracy summaries, provenance checks established that MAYO01 overlaps the earlier public development/worked-example dataset. The primary independent external cohort therefore excludes MAYO01; all-subject replication and the overlap stratum are reported separately. No detector parameters or thresholds changed. This amendment was not prespecified before the Mayo pilot. Final independent cohort: four Mayo and nine UMCU subjects when all14 are processed.",
        "annotation_files_used": len(annotation_files), "includes_extra_umcu21_sb": bool(((labels.subject == "UMCU21") & (labels.rater == "SB")).any()),
        "primary_weighting": "Each eligible labeled channel/pair has total weight one. Its reference-positive weight is the fraction of available raters voting N1. Confusion counts target a uniformly selected available rater within each record; they may be fractional. Pooled metric ratios are not means of per-rater ratios or means of subject metrics.",
        "interpretation": "These metrics quantify concordance with negative early (10-90ms) N1 annotations. Both ERPy configurations are polarity-agnostic; the broad window also admits later responses. A call discordant with an N1-negative annotation is not evidence that no biological evoked response exists. No threshold tuning, superiority test, clinical-utility claim, or equivalence claim is made.",
        "strict_consensus": "At least two available raters and unanimous binary N1 endpoint; single-rater records and disagreements excluded. Code2=P1 is N1-negative.",
        "source_eligibility": "Good ECOG recording and both stimulation contacts; non-stimulation recording contact; at least five good source events before epoch extraction. Missing/bad label codes do not define source eligibility.",
        "spatial_eligibility": "Mayo: finite verified-mm coordinates and minimum distance to either stimulation contact >=12mm. UMCU: coordinates absent, no spatial exclusion. Overall results combine these different eligibility rules and are not exact reproduction of published spatial eligibility.",
        "missingness": "Available-case excludes missing predictions and detector-unavailable results; coverage uses all source/spatial-eligible labeled records. The separate no-call policy includes detector unavailability and explicitly documented missing raw data. Final scoring requires predictions for every recoverable source-eligible record. --allow-incomplete is a preview only: its missing-computation no-calls must not become a final scientific result.",
        "processing_complete": not bool((joined.source_eligible & joined.prediction_status.eq("incomplete_processing_no_prediction")).any()),
        "paired_comparator": "Archived ER-detect calls are finite neg_peak_amplitudes; NaN is a no-call, absent archived rows are unavailable. Both paired policies additionally exclude pairs declared incomparable in the provenance manifest (e.g. different retained trial sets). Paired available-case uses evaluable ERPy; paired no-call policy retains unavailable ERPy. Calls are historical frozen outputs, not a source-package rerun. Comparator coverage=1 is conditional on this intersection.",
        "annotation_category_discordance": "Secondary table keeps annotation0 (no N1) and annotation2 (positive P1) distinct, without reclassifying P1 as N1-positive. Counts per rater are literal rows; record_normalized_rater_weight gives each available rater 1/n_raters so summing categories does not over-weight multiply rated records.",
        "bootstrap": {"unit": "whole subject", "strata": "site", "resamples": resamples, "seed": seed,
                      "interval": "2.5th/97.5th percentile among defined resamples; counts of defined draws retained. No interval for a single subject. All nested contacts, pairs, and raters remain together. Convenience-cohort uncertainty, not population representativeness."},
        "interrater_scope": "Shared valid labels after released source and site-specific spatial eligibility, independent of prediction availability; no duplicate configurations.",
        "bh": "Original q_joint and detected fields retained. No adjustment or threshold fitting after label/spatial joins.",
        "software": {"commit": commit, "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__},
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_manifest": manifest, "archived_detector_configurations": archived_config,
        "overall_metrics": summary.loc[summary.level.eq("overall")].to_dict("records"),
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("metrics", summary), ("record_predictions_and_labels", pooled),
                        ("rater_label_joins", rater_join), ("record_exclusion_denominators", exclusions),
                        ("annotation_exclusion_denominators", label_exclusions),
                        ("annotation_category_discordance", annotation_discordance),
                        ("availability_by_reference", availability), ("interrater_agreement", interrater)]:
        frame.to_csv(output / f"{name}.csv", index=False)
    (output / "summary.json").write_text(json.dumps(json_ready(report), indent=2, allow_nan=False) + "\n")
    return json_ready(report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--subject", action="append")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--pair-manifest", type=Path, help="CSV with pair-level missing-raw and historical trial-comparability decisions")
    parser.add_argument("--allow-incomplete", action="store_true", help="Preview only; never label incomplete-computation no-calls as final results")
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("bootstrap resamples must be positive")
    result = execute(args.root, args.output, {subject_of(s) for s in args.subject} if args.subject else None,
                     args.bootstrap_resamples, args.seed, args.pair_manifest, args.allow_incomplete)
    print(json.dumps({"subjects": result["subjects"], "annotation_files_used": result["annotation_files_used"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
