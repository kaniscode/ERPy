#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ERPy as ep  # noqa: E402
from ERPy.bids import normalize_stim_pair  # noqa: E402
from ERPy.io import parse_brainvision_header  # noqa: E402


@dataclass
class PublicDataset:
    dataset_id: str
    name: str
    events_url: str
    vhdr_url: str
    eeg_url: str
    subject: str
    session: str
    note: str = "open raw/signal BrainVision data"
    electrodes_url: str | None = None


DATASETS = {
    "ds003708": PublicDataset(
        "ds003708",
        "Basis profile curve / Mayo iEEG SPES",
        "https://s3.amazonaws.com/openneuro.org/ds003708/derivatives/preprocessed/sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01_events.tsv",
        "https://s3.amazonaws.com/openneuro.org/ds003708/derivatives/preprocessed/sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01_ieeg.vhdr",
        "https://s3.amazonaws.com/openneuro.org/ds003708/derivatives/preprocessed/sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_task-ccep_run-01_ieeg.eeg",
        "sub-01",
        "ses-ieeg01",
        "open preprocessed BrainVision signal derivative from the published dataset",
        "https://s3.amazonaws.com/openneuro.org/ds003708/derivatives/preprocessed/sub-01/ses-ieeg01/ieeg/sub-01_ses-ieeg01_space-MNI152NLin6Sym_electrodes.tsv",
    ),
    "ds004080": PublicDataset(
        "ds004080",
        "CCEP ECoG across age 4-51",
        "https://s3.amazonaws.com/openneuro.org/ds004080/sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_events.tsv",
        "https://s3.amazonaws.com/openneuro.org/ds004080/sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_ieeg.vhdr",
        "https://s3.amazonaws.com/openneuro.org/ds004080/sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_task-SPESclin_run-021448_ieeg.eeg",
        "sub-ccepAgeUMCU01",
        "ses-1",
        "open raw BrainVision signal from the published CCEP age dataset",
        "https://s3.amazonaws.com/openneuro.org/ds004080/sub-ccepAgeUMCU01/ses-1/ieeg/sub-ccepAgeUMCU01_ses-1_electrodes.tsv",
    ),
}


PUBLIC_CONSENSUS_METHODS = (
    "crp_significance",
    "keller_zscore",
    "kundu_rolston",
    "peak_amplitude",
    "rms_response",
)


def _finalize_public_detection_availability(
    detections: pd.DataFrame,
    *,
    primary_passband_hz: tuple[float, float],
    min_consensus: int = 2,
) -> pd.DataFrame:
    """Apply the public workflow's detector-availability contract.

    The public validation pipeline supplies only its 0.5--80-Hz primary epoch
    stream. That stream cannot support the 70--170-Hz SIGNI computation, so the
    retained ``crowther_gamma`` status row is explicitly unavailable and is not
    allowed to contribute to the historical standalone-method consensus.
    """

    output = detections.copy()
    if output.empty:
        return output

    low_hz, high_hz = (float(value) for value in primary_passband_hz)
    reason = (
        "SIGNI/high gamma unavailable in this public validation: the supplied "
        f"primary epochs were band-pass filtered to {low_hz:g}-{high_hz:g} Hz "
        "and no independent wideband_epochs input retaining 70-170 Hz was supplied"
    )
    gamma_mask = output["method"].astype(str).eq("crowther_gamma")
    output.loc[gamma_mask, "method_available"] = False
    output.loc[gamma_mask, "availability_reason"] = reason
    output.loc[gamma_mask, "notes"] = reason
    output.loc[gamma_mask, "detector_input"] = (
        f"primary_epochs_{low_hz:g}_{high_hz:g}_hz_no_wideband"
    )
    output.loc[gamma_mask, "significant"] = False
    gamma_quantity_columns = [
        column
        for column in ep.DETECTOR_QUANTITY_COLUMNS["crowther_gamma"]
        if column in output.columns
    ]
    scientific_columns = [
        column
        for column in ("score", "p_value", "threshold", *gamma_quantity_columns)
        if column in output.columns
    ]
    output.loc[gamma_mask, scientific_columns] = np.nan

    available = output[output["method"].isin(PUBLIC_CONSENSUS_METHODS)].copy()
    if "method_available" in available.columns:
        available = available[available["method_available"].fillna(False).astype(bool)]
    available = available.drop_duplicates(["channel", "method"])
    significant_counts = (
        available.groupby("channel")["significant"]
        .sum()
        .reindex(output["channel"].drop_duplicates(), fill_value=0)
        .astype(int)
    )
    available_counts = (
        available.groupby("channel")["method"]
        .nunique()
        .reindex(output["channel"].drop_duplicates(), fill_value=0)
        .astype(int)
    )
    output["n_methods_significant"] = output["channel"].map(significant_counts).fillna(0).astype(int)
    output["n_methods_available"] = output["channel"].map(available_counts).fillna(0).astype(int)
    output["consensus_ch"] = output["n_methods_significant"].ge(int(min_consensus))
    output["consensus"] = output["consensus_ch"]
    return output


def _verify_source_bytes(payload: bytes, expected_sha256: str | None, label: str) -> None:
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"Pinned public source checksum mismatch: {label}")


def _pin_dataset(ds: PublicDataset, manifest: dict, settings: dict) -> PublicDataset:
    """Bind the notebook recipe to explicit public object versions and settings."""
    if manifest.get("dataset_id") != ds.dataset_id or manifest.get("settings") != settings:
        raise ValueError("Pinned public recipe dataset/settings mismatch")
    urls = {}
    for key in ("events_url", "vhdr_url", "eeg_url", "electrodes_url"):
        source = manifest["objects"][key]
        if source["url"] != getattr(ds, key) or not source.get("version_id"):
            raise ValueError(f"Missing or inconsistent pinned source identity: {key}")
        urls[key] = source["url"] + "?versionId=" + quote(source["version_id"], safe="")
    return replace(ds, **urls)


def download_events(ds: PublicDataset, expected_sha256: str | None = None) -> pd.DataFrame:
    response = requests.get(ds.events_url, timeout=60)
    response.raise_for_status()
    _verify_source_bytes(response.content, expected_sha256, "events")
    from io import StringIO

    return pd.read_csv(StringIO(response.text), sep="\t")


def download_electrodes(url: str | None, expected_sha256: str | None = None) -> pd.DataFrame:
    if not url:
        return pd.DataFrame()
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    _verify_source_bytes(response.content, expected_sha256, "electrodes")
    from io import StringIO

    return pd.read_csv(StringIO(response.text), sep="\t")


def electrode_metadata_for_channels(
    electrodes: pd.DataFrame,
    channels: list[str],
    patient_id: str,
    session_id: str,
) -> pd.DataFrame:
    if electrodes.empty or "name" not in electrodes.columns:
        return pd.DataFrame(
            [
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "elec_label": ch,
                    "anat_label": "openraw_channel",
                    "bad_channel": False,
                }
                for ch in channels
            ]
        )
    rows = []
    lookup = electrodes.drop_duplicates("name").set_index("name")
    for ch in channels:
        row = lookup.loc[ch] if ch in lookup.index else pd.Series(dtype=object)
        anat = row.get("Destrieux_label_text", row.get("Destrieux_label", "openraw_channel"))
        rows.append(
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "elec_label": ch,
                "anat_label": "" if pd.isna(anat) else anat,
                "mni_x": pd.to_numeric(row.get("x", np.nan), errors="coerce"),
                "mni_y": pd.to_numeric(row.get("y", np.nan), errors="coerce"),
                "mni_z": pd.to_numeric(row.get("z", np.nan), errors="coerce"),
                "hemisphere": row.get("hemisphere", ""),
                "bad_channel": False,
            }
        )
    return pd.DataFrame(rows)


def fetch_range(url: str, start_byte: int, stop_byte_exclusive: int) -> bytes:
    headers = {"Range": f"bytes={start_byte}-{stop_byte_exclusive - 1}"}
    response = requests.get(url, headers=headers, timeout=120)
    response.raise_for_status()
    if response.status_code != 206:
        raise RuntimeError(f"Unexpected HTTP status for range request: {response.status_code}")
    expected_range = f"bytes {start_byte}-{stop_byte_exclusive - 1}/"
    if (not response.headers.get("Content-Range", "").startswith(expected_range)
            or len(response.content) != stop_byte_exclusive - start_byte):
        raise ValueError("Public signal range does not match requested byte interval")
    return response.content


def select_event_blocks(
    events: pd.DataFrame,
    stim_site: str | None = None,
    min_events: int = 4,
    max_sites: int = 1,
) -> list[tuple[str, pd.DataFrame]]:
    df = events.copy()
    if "status" in df.columns:
        df = df[df["status"].fillna("good").astype(str).str.lower() != "bad"]
    if "trial_type" in df.columns:
        mask = df["trial_type"].astype(str).str.contains("stim", case=False, na=False)
        if mask.any():
            df = df[mask]
    if "electrical_stimulation_site" not in df.columns:
        raise ValueError("BIDS events table lacks electrical_stimulation_site")
    df = df.dropna(subset=["onset", "electrical_stimulation_site"]).copy()
    if stim_site:
        df = df[df["electrical_stimulation_site"].astype(str) == stim_site]
    counts = df["electrical_stimulation_site"].value_counts()
    counts = counts[counts >= min_events]
    if counts.empty:
        raise ValueError(f"No stimulation site with at least {min_events} events")
    blocks = []
    for site in counts.head(max_sites).index:
        site = str(site)
        block = df[df["electrical_stimulation_site"].astype(str) == site].sort_values("onset")
        blocks.append((site, block))
    return blocks


def select_event_block(events: pd.DataFrame, stim_site: str | None = None, min_events: int = 4) -> tuple[str, pd.DataFrame]:
    return select_event_blocks(events, stim_site=stim_site, min_events=min_events, max_sites=1)[0]


def create_signal_from_public_raw(
    ds: PublicDataset,
    block: pd.DataFrame,
    out_csv: Path,
    tmin: float = -0.65,
    tmax: float = 0.75,
    max_events: int = 8,
    max_channels: int = 32,
    source_manifest: dict | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, float, list[str], dict]:
    response = requests.get(ds.vhdr_url, timeout=60)
    response.raise_for_status()
    expected_header = source_manifest["objects"]["vhdr_url"]["sha256"] if source_manifest else None
    _verify_source_bytes(response.content, expected_header, "BrainVision header")
    vhdr_text = response.text
    header = parse_brainvision_header(vhdr_text)
    sfreq = float(header["sfreq"])
    n_channels = int(header["n_channels"])
    channels = list(header["channels"])
    resolutions = np.asarray(header["resolutions"], dtype=float)
    block = block.sort_values("onset").head(max_events).copy()
    onsets = pd.to_numeric(block["onset"], errors="coerce").dropna().to_numpy(dtype=float)
    start_s = max(float(onsets.min()) + tmin, 0.0)
    stop_s = float(onsets.max()) + tmax
    start_sample = max(int(np.floor(start_s * sfreq)), 0)
    stop_sample = int(np.ceil(stop_s * sfreq))
    n_samples = stop_sample - start_sample
    bytes_per_sample = 4
    row_width = n_channels * bytes_per_sample
    start_byte = start_sample * row_width
    stop_byte = stop_sample * row_width
    payload = fetch_range(ds.eeg_url, start_byte, stop_byte)
    expected = n_samples * n_channels
    values = np.frombuffer(payload, dtype="<f4", count=expected)
    if values.size != expected:
        raise ValueError(f"Range request returned {values.size} float32 values; expected {expected}")
    data = values.reshape(n_samples, n_channels) * resolutions

    stim_site = str(block["electrical_stimulation_site"].iloc[0])
    stim_contacts = [c for c in re.split(r"[-_]", stim_site) if c]
    keep = []
    for ch in stim_contacts:
        if ch in channels and ch not in keep:
            keep.append(ch)
    for ch in channels:
        if ch not in keep and not re.search(r"ECG|EKG|TRIG|MKR|EMG|thor|abdo|xyz", ch, flags=re.I):
            keep.append(ch)
        if len(keep) >= max_channels:
            break
    keep_idx = [channels.index(ch) for ch in keep]
    base = pd.Timestamp("2026-01-01T00:00:00")
    seconds = np.arange(start_sample, stop_sample) / sfreq
    raw = pd.DataFrame(data[:, keep_idx], columns=keep)
    raw.insert(0, "times", pd.DatetimeIndex(base + pd.to_timedelta(seconds, unit="s")))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_csv, index=False)
    if source_manifest:
        expected_window = source_manifest["window"]
        if [start_sample, stop_sample] != expected_window["sample_range"]:
            raise ValueError("Pinned public signal sample interval changed")
        with out_csv.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_window["csv_sha256"]:
            raise ValueError("Pinned public signal window checksum mismatch; analysis was not run")
    meta = {
        "source_start_second": start_s,
        "source_stop_second": stop_s,
        "source_start_sample": start_sample,
        "source_stop_sample": stop_sample,
        "range_bytes_downloaded": len(payload),
        "source_channels_total": n_channels,
        "channels_kept": keep,
    }
    if source_manifest:
        meta.update(source_identity_verified=True,
                    source_snapshot_commit=source_manifest["snapshot_git_commit"],
                    source_window_expected_sha256=source_manifest["window"]["csv_sha256"],
                    source_range_sha256=hashlib.sha256(payload).hexdigest())
    return (
        base + pd.to_timedelta(start_s, unit="s"),
        base + pd.to_timedelta(stop_s, unit="s"),
        sfreq,
        keep,
        meta,
    )


def choose_representative_channel(detections: pd.DataFrame, epochs) -> str:
    if not detections.empty and "channel" in detections.columns:
        data = detections.copy()
        if "consensus_ch" in data.columns and data["consensus_ch"].any():
            data = data[data["consensus_ch"]]
        if "peak_amplitude_uv" in data.columns and not data.empty:
            channels = data.drop_duplicates("channel").sort_values("peak_amplitude_uv", ascending=False)["channel"].astype(str)
            for ch in channels:
                if ch in epochs.channels:
                    return ch
    return epochs.channels[0]


def _responsive_channels(detections: pd.DataFrame, epochs, n: int = 2) -> list[str]:
    """Pick the most responsive recording contacts for spectral demonstrations."""

    channels: list[str] = []
    if isinstance(detections, pd.DataFrame) and not detections.empty and "channel" in detections.columns:
        data = detections.copy()
        if "consensus_ch" in data.columns and data["consensus_ch"].any():
            data = data[data["consensus_ch"]]
        if "peak_amplitude_uv" in data.columns and not data.empty:
            order = data.drop_duplicates("channel").sort_values("peak_amplitude_uv", ascending=False)["channel"].astype(str)
            channels = [c for c in order if c in epochs.channels]
    if len(channels) < n:
        mean = epochs.get_mean_waveform()
        times = mean.index.to_numpy(dtype=float)
        post = times > 0
        ranked = mean.loc[post].abs().max(axis=0).sort_values(ascending=False).index.astype(str).tolist()
        for ch in ranked:
            if ch in epochs.channels and ch not in channels:
                channels.append(ch)
            if len(channels) >= n:
                break
    return channels[:n]


def write_spectral_figure_suite(fig_dir: Path, epochs, detections: pd.DataFrame) -> dict:
    """Spectral analysis figures: PSD, ERSP, ITPC, PLV, PAC comodulogram, and a connectivity matrix."""

    import matplotlib.pyplot as plt

    from ERPy.spectral import SpectralConfig
    from ERPy.viz.spectral import (
        plot_comodulogram,
        plot_connectivity_matrix,
        plot_psd,
        plot_spectral_summary,
        plot_tfr,
    )
    from ERPy.spectral import compute_psd, compute_tfr, phase_amplitude_comodulogram, plv_matrix

    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    channels = _responsive_channels(detections, epochs, n=2)
    if not channels:
        return figures
    primary = channels[0]
    nyq = 0.45 * float(epochs.sfreq)
    cfg = SpectralConfig(fmin=2.0, fmax=min(150.0, nyq), n_freqs=40, baseline=epochs.baseline or (-0.25, -0.05))

    try:
        fig = plot_spectral_summary(epochs, primary, config=cfg)
        path = fig_dir / "spectral_summary_panel.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["spectral_summary_panel"] = str(path)
    except Exception as exc:
        figures["spectral_summary_panel_error"] = str(exc)

    try:
        psd = compute_psd(epochs, fmin=1.0, fmax=min(150.0, nyq))
        ax = plot_psd(psd, top_n=8)
        path = fig_dir / "power_spectral_density.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["power_spectral_density"] = str(path)
    except Exception as exc:
        figures["power_spectral_density_error"] = str(exc)

    try:
        tfr = compute_tfr(epochs, primary, config=cfg)
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.0), layout="constrained")
        plot_tfr(tfr, ax=axes[0], kind="power")
        plot_tfr(tfr, ax=axes[1], kind="itc")
        path = fig_dir / "time_frequency_ersp_itpc.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["time_frequency_ersp_itpc"] = str(path)
    except Exception as exc:
        figures["time_frequency_ersp_itpc_error"] = str(exc)

    try:
        amp_lo = min(70.0, max(40.0, nyq - 30.0))
        amp_hi = min(150.0, nyq)
        como = phase_amplitude_comodulogram(
            epochs, primary,
            phase_freqs=[4, 6, 8, 10, 12], amp_freqs=list(range(int(amp_lo), int(amp_hi), 15)) or [amp_lo],
            bandwidth_phase=2.0, bandwidth_amp=20.0,
        )
        ax = plot_comodulogram(como)
        path = fig_dir / "phase_amplitude_comodulogram.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["phase_amplitude_comodulogram"] = str(path)
    except Exception as exc:
        figures["phase_amplitude_comodulogram_error"] = str(exc)

    try:
        window = (0.0, min(0.3, float(epochs.tmax)))
        plv = plv_matrix(epochs, band=(8.0, 13.0), time_window=window)
        ax = plot_connectivity_matrix(plv, title="Alpha-band inter-trial PLV", label="PLV")
        path = fig_dir / "plv_connectivity_matrix.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["plv_connectivity_matrix"] = str(path)
    except Exception as exc:
        figures["plv_connectivity_matrix_error"] = str(exc)

    return figures


def write_figure_suite(root: Path, patient, session_id: str, stim_pair: str, epochs, detections: pd.DataFrame, artifact_report) -> dict:
    import matplotlib.pyplot as plt

    from ERPy.viz.networks import (
        edges_from_detections,
        graph_from_edges,
        plot_coordinate_network,
        plot_evoked_response_graph,
        plot_glass_brain_network,
        plot_interactive_connectome,
        plot_network,
        plot_node_metric_glass_brain,
        plot_node_metric_template_brain,
        plot_response_matrix_heatmap,
        plot_template_brain_network,
        response_matrix_from_edges,
        validate_mni_coordinates,
    )
    from ERPy.crp import CRPConfig, run_crp, run_crp_all
    from ERPy.response_metrics import zscore_metric_within_stim
    from ERPy.viz.crp import plot_crp_response_contrast, plot_crp_score_map, plot_crp_summary
    from ERPy.viz.detections import choose_significant_and_nonsignificant_channels, plot_detection_summary
    from ERPy.viz.response_metrics import plot_within_stim_zscore_bars
    from ERPy.viz.theme import apply_erpy_theme
    from ERPy.viz.waveforms import plot_response_map

    apply_erpy_theme()
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}
    try:
        audit_path = root / "mni_coordinate_audit.csv"
        validate_mni_coordinates(patient.elec_meta).to_csv(audit_path, index=False)
        figures["mni_coordinate_audit"] = str(audit_path)
    except Exception as exc:
        figures["mni_coordinate_audit_error"] = str(exc)
    channel = choose_representative_channel(detections, epochs)

    fig = epochs.plot.summary(channel, detections=detections)
    path = fig_dir / "summary_waveform_panel.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures["summary_waveform_panel"] = str(path)

    figures.update(write_spectral_figure_suite(fig_dir, epochs, detections))

    try:
        from ERPy.spectral import SpectralConfig, plv_matrix
        from ERPy.viz.spectral import plot_connectivity_matrix, plot_spectral_summary

        spectral_cfg = SpectralConfig(fmin=4.0, fmax=min(150.0, epochs.sfreq * 0.45), n_freqs=40, baseline=epochs.baseline)
        fig = plot_spectral_summary(epochs, channel, config=spectral_cfg)
        path = fig_dir / "spectral_summary_panel.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["spectral_summary_panel"] = str(path)

        conn_channels = [channel] + [c for c in epochs.channels if c != channel][:11]
        plv = plv_matrix(epochs, band=(8.0, 13.0), channels=conn_channels, time_window=(0.0, min(0.5, epochs.tmax)))
        ax = plot_connectivity_matrix(plv, title=f"{stim_pair} alpha-band inter-trial PLV")
        path = fig_dir / "plv_connectivity_matrix.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["plv_connectivity_matrix"] = str(path)
    except Exception as exc:
        figures["spectral_panel_error"] = str(exc)

    fig, _ = epochs.plot.ranked_grid(detections, metric="peak_amplitude_uv", top_n=16, consensus_only=False)
    path = fig_dir / "response_ranked_grid.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures["response_ranked_grid"] = str(path)

    channel_meta = patient.elec_meta.rename(columns={"elec_label": "channel"})
    if "anat_label" in channel_meta.columns:
        fig, _ = epochs.plot.grouped_grid(channel_meta, group_col="anat_label", ncols=4)
        path = fig_dir / "anatomy_grouped_grid.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["anatomy_grouped_grid"] = str(path)

    ax = plot_response_map(detections, metric="peak_amplitude_uv", top_n=20)
    path = fig_dir / "response_map.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["response_map"] = str(path)

    fig = plot_detection_summary(detections, top_n=24)
    path = fig_dir / "detection_method_summary.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures["detection_method_summary"] = str(path)

    zscore_table = zscore_metric_within_stim(
        detections.assign(stim_pair=stim_pair),
        "peak_amplitude_uv",
        absolute=True,
    )
    zscore_table.to_csv(root / "within_stim_peak_amplitude_zscores.csv", index=False)
    ax = plot_within_stim_zscore_bars(
        zscore_table,
        "peak_amplitude_uv",
        stim_pair=stim_pair,
        top_n=18,
        title=f"{stim_pair} peak-amplitude z-score across recording contacts",
    )
    path = fig_dir / "within_stim_peak_amplitude_zscores.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["within_stim_peak_amplitude_zscores"] = str(path)

    crp_config = CRPConfig(
        response_window=(0.01, min(0.3, epochs.tmax)),
        baseline_window=epochs.baseline or (-0.25, -0.05),
        permutation_n=100,
        random_state=0,
    )
    crp_table = run_crp_all(epochs, config=crp_config)
    crp_table.to_csv(root / "crp_scores.csv", index=False)
    ax = plot_crp_score_map(crp_table, top_n=20)
    path = fig_dir / "crp_score_map.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["crp_score_map"] = str(path)

    crp_channel = choose_representative_channel(
        crp_table.rename(columns={"score": "peak_amplitude_uv"}).assign(consensus_ch=crp_table["significant"]),
        epochs,
    )
    try:
        crp_result = run_crp(epochs, crp_channel, config=crp_config)
        fig = plot_crp_summary(epochs, crp_channel, result=crp_result)
        path = fig_dir / "crp_curve_summary.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["crp_curve_summary"] = str(path)
    except Exception as exc:
        figures["crp_curve_summary_error"] = str(exc)

    sig_channel, nonsig_channel = choose_significant_and_nonsignificant_channels(detections, epochs.channels)
    try:
        fig = plot_crp_response_contrast(epochs, sig_channel, nonsig_channel, config=crp_config)
        path = fig_dir / "significant_vs_nonsignificant_crp_contrast.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["significant_vs_nonsignificant_crp_contrast"] = str(path)
    except Exception as exc:
        figures["significant_vs_nonsignificant_crp_contrast_error"] = str(exc)

    graph_detections = detections.assign(stim_pair=stim_pair)
    edges = edges_from_detections(
        graph_detections,
        elec_meta=patient.elec_meta,
        metric="peak_amplitude_uv",
        consensus_only=False,
    )
    significant_edges = edges_from_detections(
        graph_detections,
        elec_meta=patient.elec_meta,
        metric="peak_amplitude_uv",
        consensus_only=True,
    )
    edges.to_csv(root / "response_edges.csv", index=False)
    significant_edges.to_csv(root / "significant_response_edges.csv", index=False)
    if not edges.empty:
        response_matrix = response_matrix_from_edges(edges, elec_meta=patient.elec_meta)
        response_matrix.to_csv(root / "response_source_matrix.csv")
        ax = plot_response_matrix_heatmap(edges, elec_meta=patient.elec_meta, title=f"{stim_pair} response field")
        path = fig_dir / "response_source_matrix_heatmap.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["response_source_matrix_heatmap"] = str(path)

        network_edges = significant_edges if not significant_edges.empty else edges
        ax = plot_coordinate_network(network_edges, patient.elec_meta, title=f"{stim_pair} significant MNI coordinate response network")
        path = fig_dir / "coordinate_response_graph.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["coordinate_response_graph"] = str(path)

        fig = plot_template_brain_network(network_edges, patient.elec_meta, title=f"{stim_pair} fsaverage cortical response network")
        path = fig_dir / "template_brain_response_network.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["template_brain_response_network"] = str(path)

        try:
            display = plot_glass_brain_network(
                network_edges,
                patient.elec_meta,
                title=None,
                edge_linewidth=1.45,
                edge_alpha=0.82,
            )
            path = fig_dir / "glass_brain_response_network.png"
            display.savefig(path, dpi=300)
            display.close()
            figures["glass_brain_response_network"] = str(path)
        except Exception as exc:
            figures["glass_brain_response_network_error"] = str(exc)

        graph = graph_from_edges(network_edges, threshold=0.0)
        ax = plot_network(graph, title=f"{stim_pair} Kamada-Kawai response topology")
        path = fig_dir / "network_graph.png"
        ax.figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(ax.figure)
        figures["network_graph"] = str(path)

        try:
            html = fig_dir / "interactive_connectome.html"
            plot_interactive_connectome(network_edges, patient.elec_meta, output_html=html, top_n=48, title=f"{stim_pair} significant response connectome")
            figures["interactive_connectome"] = str(html)
        except Exception as exc:
            figures["interactive_connectome_error"] = str(exc)

        try:
            html = fig_dir / "evoked_response_graph.html"
            plot_evoked_response_graph(
                epochs,
                patient.elec_meta,
                stim_pair=stim_pair,
                detections=detections,
                consensus_only=False,
                output_html=html,
                top_n=20,
                title=f"{stim_pair} evoked response graph",
            )
            figures["evoked_response_graph"] = str(html)
        except Exception as exc:
            figures["evoked_response_graph_error"] = str(exc)

    artifact_report.to_csv(root / "artifact_response_report.csv")
    (root / "figure_manifest.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    return figures


def run_site(
    ds: PublicDataset,
    site: str,
    block: pd.DataFrame,
    out_root: Path,
    max_events: int = 8,
    max_channels: int = 32,
    make_figures: bool = False,
    source_manifest: dict | None = None,
) -> dict:
    stim_pair = normalize_stim_pair(site)
    patient_id = ds.dataset_id.replace("ds", "DS")
    session_id = "A"
    root = out_root / ds.dataset_id / stim_pair
    raw_dir = root / "raw"
    meta_dir = root / "metadata"
    proc_dir = root / "procdata"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_dir / f"{ds.dataset_id}_{stim_pair}.csv"
    try:
        stim_start, stim_stop, sfreq, kept_channels, raw_meta_extra = create_signal_from_public_raw(
            ds,
            block,
            raw_csv,
            max_events=max_events,
            max_channels=max_channels,
            source_manifest=source_manifest,
        )
    except requests.RequestException:
        if source_manifest:
            raise  # A pinned recipe must not fall back to unverified caches.
        raw_meta_path = meta_dir / "raw_metadata.csv"
        if not raw_csv.exists() or not raw_meta_path.exists():
            raise
        cached_raw_meta = pd.read_csv(raw_meta_path).iloc[0]
        stim_start = pd.Timestamp(cached_raw_meta["start_time"])
        stim_stop = pd.Timestamp(cached_raw_meta["stop_time"])
        sfreq = float(cached_raw_meta["sampling_freq"])
        kept_channels = [c for c in pd.read_csv(raw_csv, nrows=0).columns if c != "times"]
        summary_path = root / "summary.json"
        raw_meta_extra = {}
        if summary_path.exists():
            try:
                previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                raw_meta_extra = {
                    key: previous_summary[key]
                    for key in previous_summary
                    if key.startswith("source_") and key not in {"source_events_url", "source_vhdr_url", "source_eeg_url", "source_electrodes_url", "source_note", "source_stim_site"}
                }
            except Exception:
                raw_meta_extra = {}
        raw_meta_extra["source_window_cache_used"] = True

    pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "raw_file": raw_csv.name,
                "start_time": stim_start.isoformat(),
                "stop_time": stim_stop.isoformat(),
                "sampling_freq": sfreq,
            }
        ]
    ).to_csv(meta_dir / "raw_metadata.csv", index=False)
    stim_freq = 1.0 / np.median(np.diff(pd.to_numeric(block["onset"], errors="coerce").dropna())) if len(block) > 2 else 0.2
    pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "stim_pair": stim_pair,
                "stim_start": stim_start.isoformat(),
                "stim_stop": stim_stop.isoformat(),
                "stim_freq": float(stim_freq),
                "source_dataset": ds.dataset_id,
                "source_stim_site": site,
            }
        ]
    ).to_csv(meta_dir / "stim_metadata.csv", index=False)
    elec_meta_path = meta_dir / "electrode_metadata.csv"
    try:
        expected_electrodes = source_manifest["objects"]["electrodes_url"]["sha256"] if source_manifest else None
        electrodes = download_electrodes(ds.electrodes_url, expected_electrodes)
        electrode_metadata_for_channels(electrodes, kept_channels, patient_id, session_id).to_csv(
            elec_meta_path,
            index=False,
        )
    except requests.RequestException:
        if source_manifest:
            raise
        if not elec_meta_path.exists():
            raise
    config = {
        "rawdata_path": str(raw_dir),
        "procdata_path": str(proc_dir),
        "rawdata_meta_path": str(meta_dir / "raw_metadata.csv"),
        "stim_meta_path": str(meta_dir / "stim_metadata.csv"),
        "elec_meta_path": str(meta_dir / "electrode_metadata.csv"),
        "session_ids": {patient_id: [session_id]},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    used_block = block.sort_values("onset").head(max_events)
    canonical_events = ep.create_events_from_timestamps(pd.Timestamp("2026-01-01") + pd.to_timedelta(used_block["onset"], unit="s"))
    patient = ep.Patient(patient_id, config_path=str(config_path))
    event_path = patient.dataloader.get_event_path(session_id, stim_pair)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_events.to_csv(event_path, index=False)

    primary_highcut_hz = min(80.0, sfreq / 2 * 0.8)
    steps = [
        ("reject_bad_channels", {"bad_channels": "auto", "method": "drop", "detection_params": {"zscore_threshold": 8.0}}),
        ("artifact_blank", {"width_s": 0.004}),
        ("notch_filter", {"notch_freq": 60.0, "bw": 2.0, "harm": True}),
        ("bandpass_filter", {"lowcut": 0.5, "highcut": primary_highcut_hz, "order": 3}),
    ]
    # Preserve a sample-matched prestimulus baseline for the default
    # 10--300 ms detector response interval, plus filter/epoch edge padding.
    epochs = patient.epoch(
        session_id,
        stim_pair,
        pipeline=steps,
        tmin=-0.5,
        tmax=0.6,
        baseline=(-0.5, -0.03),
        cache=False,
    )
    artifact_report = epochs.flag_artifacts(zscore_threshold=3.0)
    clean = epochs.reject_artifacts(report=artifact_report, mode="nan_response")
    if clean.n_trials() == 0:
        clean = epochs
    zero_report = clean.zero_time_report()
    pd.DataFrame([zero_report]).to_csv(root / "post_artifact_zero_report.csv", index=False)
    clean.post_artifact_anchor_report().to_csv(root / "post_artifact_anchor_report.csv", index=False)
    clean.detect_zero_time_artifacts(zero_time=0.0).to_csv(root / "fixed_zero_artifact_risk_report.csv", index=False)
    detections = clean.detect_erp_all(min_consensus=2)
    detections = _finalize_public_detection_availability(
        detections,
        primary_passband_hz=(0.5, primary_highcut_hz),
        min_consensus=2,
    )
    detections.to_csv(root / "detections.csv", index=False)
    artifact_report.to_csv(root / "artifact_response_report.csv")
    figures = write_figure_suite(root, patient, session_id, stim_pair, clean, detections, artifact_report) if make_figures else {}
    summary = {
        "dataset_id": ds.dataset_id,
        "dataset_name": ds.name,
        "source_events_url": ds.events_url,
        "source_vhdr_url": ds.vhdr_url,
        "source_eeg_url": ds.eeg_url,
        "source_electrodes_url": ds.electrodes_url,
        "source_note": ds.note,
        "source_stim_site": site,
        "stim_pair": stim_pair,
        "n_source_events_for_site": int(len(block)),
        "n_source_events_used": int(len(used_block)),
        "sfreq": float(sfreq),
        **raw_meta_extra,
        "n_erpy_events": int(len(canonical_events)),
        "n_epochs": int(epochs.n_trials()),
        "n_epochs_after_artifact_qc": int(clean.n_trials()),
        "post_artifact_anchor_median_ms": float(zero_report["median_anchor_s"] * 1000.0)
        if zero_report.get("median_anchor_s") is not None
        else None,
        "post_artifact_zero_residual_uv": zero_report.get("max_abs_uv"),
        "n_fixed_zero_artifact_risk_epochs": int(
            pd.read_csv(root / "fixed_zero_artifact_risk_report.csv")["zero_time_is_artifact"].sum()
        )
        if (root / "fixed_zero_artifact_risk_report.csv").exists()
        else 0,
        "n_flagged_artifact_responses": int(artifact_report.table["bad_response"].sum()),
        "n_detection_rows": int(len(detections)),
        "n_consensus_channels": int(detections.drop_duplicates("channel")["consensus_ch"].sum()),
        "standalone_consensus_definition": (
            ">=2 significant available methods among "
            + ", ".join(PUBLIC_CONSENSUS_METHODS)
        ),
        "primary_epoch_passband_hz": [0.5, float(primary_highcut_hz)],
        "crowther_gamma_available": False,
        "crowther_gamma_availability_reason": (
            detections.loc[
                detections["method"].eq("crowther_gamma"),
                "availability_reason",
            ].iloc[0]
            if detections["method"].eq("crowther_gamma").any()
            else "crowther_gamma status row absent"
        ),
        "n_response_edges": int(len(pd.read_csv(root / "response_edges.csv"))) if (root / "response_edges.csv").exists() else 0,
        "n_figures": int(sum(not k.endswith("_error") for k in figures)),
        "workspace": str(root),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_dataset_aggregate_figure_suite(dataset_root: Path, summaries: list[dict]) -> dict:
    import matplotlib.pyplot as plt

    from ERPy.viz.networks import (
        graph_from_edges,
        plot_aggregate_evoked_response_graph,
        plot_coordinate_network,
        plot_glass_brain_network,
        plot_interactive_connectome,
        plot_network,
        plot_node_metric_glass_brain,
        plot_node_metric_template_brain,
        plot_response_matrix_heatmap,
        plot_template_brain_network,
        response_matrix_from_edges,
        resolve_electrode_label,
        stim_pair_to_source_electrode,
        validate_mni_coordinates,
    )
    from ERPy.crp import compare_crp_across_stim_sites
    from ERPy.graph_metrics import compute_dynamic_graph_metrics, evoked_response_edges_over_time, summarize_graph_metric
    from ERPy.response_metrics import zscore_metric_within_stim
    from ERPy.viz.crp import plot_crp_site_comparison
    from ERPy.viz.graph_metrics import plot_graph_metric_heatmap, plot_graph_metric_timecourse
    from ERPy.viz.response_metrics import plot_within_stim_zscore_heatmap
    from ERPy.viz.theme import apply_erpy_theme

    apply_erpy_theme()
    fig_dir = dataset_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    edge_tables = []
    significant_edge_tables = []
    meta_tables = []
    crp_tables = []
    for summary in summaries:
        root = Path(summary["workspace"])
        edges_path = root / "response_edges.csv"
        significant_edges_path = root / "significant_response_edges.csv"
        meta_path = root / "metadata" / "electrode_metadata.csv"
        crp_path = root / "crp_scores.csv"
        if edges_path.exists():
            edge_tables.append(pd.read_csv(edges_path))
        if significant_edges_path.exists():
            significant_edge_tables.append(pd.read_csv(significant_edges_path))
        if meta_path.exists():
            meta_tables.append(pd.read_csv(meta_path))
        if crp_path.exists():
            crp_tables.append(pd.read_csv(crp_path).assign(stim_pair=str(summary.get("stim_pair") or root.name)))
    if not edge_tables:
        return {}

    edges = pd.concat(edge_tables, ignore_index=True)
    significant_edges = pd.concat(significant_edge_tables, ignore_index=True) if significant_edge_tables else pd.DataFrame()
    network_edges = significant_edges if not significant_edges.empty else edges
    elec_meta = pd.concat(meta_tables, ignore_index=True).drop_duplicates("elec_label") if meta_tables else pd.DataFrame()
    edges.to_csv(dataset_root / "aggregate_response_edges.csv", index=False)
    significant_edges.to_csv(dataset_root / "aggregate_significant_response_edges.csv", index=False)
    matrix = response_matrix_from_edges(edges, elec_meta=elec_meta)
    matrix.to_csv(dataset_root / "aggregate_response_matrix.csv")
    peak_zscores = zscore_metric_within_stim(
        edges,
        "peak_amplitude_uv" if "peak_amplitude_uv" in edges.columns else "weight",
        channel_col="record_elec",
        absolute=True,
        output_col="peak_amplitude_within_stim_z",
    )
    peak_zscores.to_csv(dataset_root / "aggregate_peak_amplitude_within_stim_zscores.csv", index=False)

    figures = {}
    if not elec_meta.empty:
        try:
            audit_path = dataset_root / "aggregate_mni_coordinate_audit.csv"
            validate_mni_coordinates(elec_meta).to_csv(audit_path, index=False)
            figures["aggregate_mni_coordinate_audit"] = str(audit_path)
        except Exception as exc:
            figures["aggregate_mni_coordinate_audit_error"] = str(exc)
    ax = plot_response_matrix_heatmap(
        edges,
        elec_meta=elec_meta,
        title="Aggregate stimulation-source by recording-channel response matrix",
    )
    path = fig_dir / "aggregate_response_matrix_heatmap.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["aggregate_response_matrix_heatmap"] = str(path)

    ax = plot_within_stim_zscore_heatmap(
        peak_zscores,
        "peak_amplitude_uv" if "peak_amplitude_uv" in peak_zscores.columns else "weight",
        channel_col="record_elec",
        z_col="peak_amplitude_within_stim_z",
        title="Peak amplitude z-scored across recording contacts within each stimulation site",
    )
    path = fig_dir / "aggregate_peak_amplitude_within_stim_zscores.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["aggregate_peak_amplitude_within_stim_zscores"] = str(path)

    if crp_tables:
        crp_comparison = compare_crp_across_stim_sites(pd.concat(crp_tables, ignore_index=True))
        crp_comparison.to_csv(dataset_root / "aggregate_crp_site_comparison.csv", index=False)
        for metric_name, filename, label in [
            ("within_stim_z", "aggregate_crp_within_stim_z_heatmap.png", "CRP projection z-scored within stimulation site"),
            ("baseline_z", "aggregate_crp_baseline_z_heatmap.png", "CRP canonical-weight baseline z-score"),
            (
                "reconstruction_rms_within_stim_z",
                "aggregate_crp_reconstruction_rms_within_stim_z_heatmap.png",
                "CRP reconstructed RMS z-scored within stimulation site",
            ),
            ("permutation_log10_p", "aggregate_crp_permutation_logp_heatmap.png", "CRP permutation/null -log10 p"),
        ]:
            if metric_name in crp_comparison.columns and crp_comparison[metric_name].notna().any():
                ax = plot_crp_site_comparison(crp_comparison, metric=metric_name, title=label)
                path = fig_dir / filename
                ax.figure.savefig(path, dpi=300, bbox_inches="tight")
                plt.close(ax.figure)
                figures[filename.removesuffix(".png")] = str(path)

    ax = plot_coordinate_network(
        network_edges,
        elec_meta,
        title="Aggregate significant MNI coordinate response network",
        top_n=120,
    )
    path = fig_dir / "aggregate_coordinate_response_network.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["aggregate_coordinate_response_network"] = str(path)

    fig = plot_template_brain_network(
        network_edges,
        elec_meta,
        title="Aggregate fsaverage cortical response network",
        top_n=120,
    )
    path = fig_dir / "aggregate_template_brain_response_network.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures["aggregate_template_brain_response_network"] = str(path)

    evoked_runs = []
    for summary in summaries:
        root = Path(summary["workspace"])
        stim_pair = str(summary.get("stim_pair") or root.name)
        epoch_files = sorted(
            (root / "procdata").glob(f"*/A/epochs/{stim_pair}_A_*_epochs.h5"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        detections_path = root / "detections.csv"
        if not epoch_files or not detections_path.exists():
            continue
        try:
            evoked_runs.append(
                {
                    "epochs": ep.Epochs.from_hdf(epoch_files[0]),
                    "stim_pair": stim_pair,
                    "detections": pd.read_csv(detections_path),
                }
            )
        except Exception:
            continue
    if evoked_runs:
        edge_time_table = pd.DataFrame()
        graph_metric_table = pd.DataFrame()
        try:
            html = fig_dir / "aggregate_evoked_response_graph.html"
            plot_aggregate_evoked_response_graph(
                evoked_runs,
                elec_meta,
                output_html=html,
                top_n_per_stim=14,
                top_n_total=90,
                frame_step_ms=16.0,
                title="Aggregate evoked response graph over time",
            )
            figures["aggregate_evoked_response_graph"] = str(html)
        except Exception as exc:
            figures["aggregate_evoked_response_graph_error"] = str(exc)
        try:
            edge_time_table = evoked_response_edges_over_time(
                evoked_runs,
                elec_meta=elec_meta,
                response_window=(0.0, 0.35),
                frame_step_ms=16.0,
                top_n_per_stim=14,
                top_n_total=90,
                metric_label="Mean evoked voltage (uV)",
            )
            edge_time_table.to_csv(dataset_root / "aggregate_evoked_response_edges_over_time.csv", index=False)
            graph_metric_table = compute_dynamic_graph_metrics(
                edge_time_table,
                metrics=("in_strength", "out_strength", "total_strength", "hub", "authority", "pagerank", "betweenness"),
            )
            graph_metric_table.to_csv(dataset_root / "aggregate_dynamic_graph_metrics.csv", index=False)
            if not graph_metric_table.empty:
                stimulation_nodes = sorted(edge_time_table["source"].dropna().astype(str).unique()) if "source" in edge_time_table.columns else []
                hub_summary = summarize_graph_metric(graph_metric_table, metric="hub", summary="max")
                authority_summary = summarize_graph_metric(graph_metric_table, metric="authority", summary="max")
                hub_focus = str(hub_summary.index[0]) if not hub_summary.empty else str(graph_metric_table["node"].iloc[0])
                authority_focus = str(authority_summary.index[0]) if not authority_summary.empty else str(graph_metric_table["node"].iloc[0])
                fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.2), layout="constrained")
                plot_graph_metric_timecourse(
                    graph_metric_table,
                    node=hub_focus,
                    metric="hub",
                    ax=axes[0],
                    title=f"{hub_focus} hub centrality through response epoch",
                )
                plot_graph_metric_timecourse(
                    graph_metric_table,
                    node=authority_focus,
                    metric="authority",
                    ax=axes[1],
                    color="#0f766e",
                    title=f"{authority_focus} authority through response epoch",
                )
                path = fig_dir / "aggregate_graph_metric_timecourses.png"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                figures["aggregate_graph_metric_timecourses"] = str(path)

                for graph_metric, label in [("hub", "hub centrality"), ("authority", "authority centrality")]:
                    ax = plot_graph_metric_heatmap(
                        graph_metric_table,
                        metric=graph_metric,
                        title=f"Dynamic {label} over the stimulation epoch",
                    )
                    path = fig_dir / f"aggregate_dynamic_{graph_metric}_heatmap.png"
                    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
                    plt.close(ax.figure)
                    figures[f"aggregate_dynamic_{graph_metric}_heatmap"] = str(path)

                    fig = plot_node_metric_template_brain(
                        graph_metric_table,
                        elec_meta,
                        metric=graph_metric,
                        summary="max",
                        stimulation_nodes=stimulation_nodes,
                        title=f"Max {label} over response epoch",
                    )
                    path = fig_dir / f"aggregate_dynamic_{graph_metric}_template_brain.png"
                    fig.savefig(path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                    figures[f"aggregate_dynamic_{graph_metric}_template_brain"] = str(path)

                    try:
                        display = plot_node_metric_glass_brain(
                            graph_metric_table,
                            elec_meta,
                            metric=graph_metric,
                            summary="max",
                            stimulation_nodes=stimulation_nodes,
                            title=None,
                        )
                        path = fig_dir / f"aggregate_dynamic_{graph_metric}_glass_brain.png"
                        display.savefig(path, dpi=300)
                        display.close()
                        figures[f"aggregate_dynamic_{graph_metric}_glass_brain"] = str(path)
                    except Exception as exc:
                        figures[f"aggregate_dynamic_{graph_metric}_glass_brain_error"] = str(exc)
        except Exception as exc:
            figures["aggregate_dynamic_graph_metrics_error"] = str(exc)
        try:
            available_labels = elec_meta["elec_label"].astype(str).tolist() if "elec_label" in elec_meta.columns else []
            static_edge_rows = []
            cumulative_by_node_time: dict[tuple[str, float], float] = {}
            sig_by_node_time: dict[tuple[str, float], int] = {}
            for run in evoked_runs:
                epochs = run["epochs"]
                stim_pair = str(run["stim_pair"])
                detections = run.get("detections")
                mean = epochs.get_mean_waveform()
                times = mean.index.to_numpy(dtype=float)
                response_mask = (times >= 0.0) & (times <= 0.35)
                if not response_mask.any():
                    continue
                stim_node = resolve_electrode_label(stim_pair_to_source_electrode(stim_pair), available_labels)
                if isinstance(detections, pd.DataFrame) and not detections.empty and "channel" in detections.columns:
                    ranked = detections.copy()
                    if "consensus_ch" in ranked.columns and ranked["consensus_ch"].any():
                        ranked = ranked[ranked["consensus_ch"]]
                    if "peak_amplitude_uv" in ranked.columns:
                        ranked["_rank_metric"] = pd.to_numeric(ranked["peak_amplitude_uv"], errors="coerce").abs()
                        ranked = ranked.drop_duplicates("channel").sort_values("_rank_metric", ascending=False)
                    method_counts = (
                        ranked[ranked.get("significant", False).astype(bool)].groupby("channel")["method"].nunique().to_dict()
                        if {"significant", "method"}.issubset(ranked.columns)
                        else {}
                    )
                    channels = ranked["channel"].astype(str).tolist()
                else:
                    method_counts = {}
                    channels = mean.loc[response_mask].abs().max(axis=0).sort_values(ascending=False).index.astype(str).tolist()
                selected = []
                for channel in channels:
                    if channel not in mean.columns:
                        continue
                    record_node = resolve_electrode_label(channel, available_labels)
                    if record_node == stim_node:
                        continue
                    peak = float(mean.loc[response_mask, channel].abs().max())
                    selected.append((channel, record_node, peak, int(method_counts.get(channel, 0))))
                for channel, record_node, peak, sig_count in sorted(selected, key=lambda item: item[2], reverse=True)[:14]:
                    static_edge_rows.append(
                        {
                            "stim_pair": stim_pair,
                            "stim_elec": stim_node,
                            "record_elec": record_node,
                            "weight": peak,
                            "peak_amplitude_uv": peak,
                        }
                    )
                    for time_s, value in mean.loc[response_mask, channel].items():
                        key = (record_node, round(float(time_s), 4))
                        magnitude = abs(float(value))
                        cumulative_by_node_time[key] = cumulative_by_node_time.get(key, 0.0) + magnitude
                        if peak > 0 and magnitude >= 0.20 * peak:
                            sig_by_node_time[key] = sig_by_node_time.get(key, 0) + sig_count
            if static_edge_rows:
                static_edges = pd.DataFrame(static_edge_rows)
                node_size_values = (
                    pd.Series(cumulative_by_node_time)
                    .rename_axis(["node", "time"])
                    .reset_index(name="value")
                    .groupby("node")["value"]
                    .max()
                )
                node_color_values = (
                    pd.Series(sig_by_node_time)
                    .rename_axis(["node", "time"])
                    .reset_index(name="value")
                    .groupby("node")["value"]
                    .max()
                )
                static_edges.to_csv(dataset_root / "aggregate_evoked_response_max_edges.csv", index=False)
                node_size_values.to_csv(dataset_root / "aggregate_evoked_response_max_node_magnitude.csv")
                node_color_values.to_csv(dataset_root / "aggregate_evoked_response_max_node_significance_counts.csv")
                fig = plot_template_brain_network(
                    static_edges,
                    elec_meta,
                    title="Aggregate evoked response max-over-epoch brain network",
                    top_n=90,
                    node_size_values=node_size_values,
                    node_color_values=node_color_values,
                    node_color_label="Max active significance metrics",
                )
                path = fig_dir / "aggregate_evoked_response_max_network.png"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                figures["aggregate_evoked_response_max_network"] = str(path)
        except Exception as exc:
            figures["aggregate_evoked_response_max_network_error"] = str(exc)

    try:
        html = fig_dir / "aggregate_interactive_connectome.html"
        plot_interactive_connectome(
            network_edges,
            elec_meta,
            output_html=html,
            top_n=120,
            title="Aggregate interactive fsaverage stimulation-response connectome",
        )
        figures["aggregate_interactive_connectome"] = str(html)
    except Exception as exc:
        figures["aggregate_interactive_connectome_error"] = str(exc)

    try:
        display = plot_glass_brain_network(
            network_edges,
            elec_meta,
            title=None,
            edge_linewidth=1.35,
            edge_alpha=0.78,
        )
        path = fig_dir / "aggregate_glass_brain_response_network.png"
        display.savefig(path, dpi=300)
        display.close()
        figures["aggregate_glass_brain_response_network"] = str(path)
    except Exception as exc:
        figures["aggregate_glass_brain_response_network_error"] = str(exc)

    graph = graph_from_edges(network_edges, threshold=0.0)
    ax = plot_network(graph, title="Aggregate Kamada-Kawai response topology")
    path = fig_dir / "aggregate_networkx_response_graph.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["aggregate_networkx_response_graph"] = str(path)

    source_counts = network_edges.groupby("stim_pair").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(6.8, 3.4), layout="constrained")
    ax.bar(source_counts.index, source_counts.values, color="#2563eb", alpha=0.86)
    ax.set_ylabel("Response edges")
    ax.set_xlabel("Stimulation pair")
    ax.set_title("Edges per stimulation source")
    ax.tick_params(axis="x", rotation=35)
    path = fig_dir / "aggregate_source_edge_counts.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures["aggregate_source_edge_counts"] = str(path)

    (dataset_root / "aggregate_figure_manifest.json").write_text(json.dumps(figures, indent=2), encoding="utf-8")
    return figures


def _load_cached_summaries(dataset_root: Path, max_sites: int) -> list[dict]:
    summaries = []
    for path in sorted(dataset_root.glob("*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["workspace"] = str(path.parent)
        summaries.append(summary)
    return summaries[:max_sites]


def refresh_cached_network_figures(site_root: Path) -> dict:
    """Regenerate network/brain figures from an existing validation workspace."""

    import matplotlib.pyplot as plt

    from ERPy.viz.networks import (
        graph_from_edges,
        plot_coordinate_network,
        plot_evoked_response_graph,
        plot_glass_brain_network,
        plot_interactive_connectome,
        plot_network,
        plot_response_matrix_heatmap,
        plot_template_brain_network,
        response_matrix_from_edges,
        validate_mni_coordinates,
    )
    from ERPy.crp import CRPConfig, run_crp, run_crp_all
    from ERPy.response_metrics import zscore_metric_within_stim
    from ERPy.viz.crp import plot_crp_response_contrast, plot_crp_score_map, plot_crp_summary
    from ERPy.viz.detections import choose_significant_and_nonsignificant_channels
    from ERPy.viz.response_metrics import plot_within_stim_zscore_bars
    from ERPy.viz.theme import apply_erpy_theme

    apply_erpy_theme()
    meta_path = site_root / "metadata" / "electrode_metadata.csv"
    edges_path = site_root / "response_edges.csv"
    significant_edges_path = site_root / "significant_response_edges.csv"
    if not meta_path.exists() or not edges_path.exists():
        return {}
    elec_meta = pd.read_csv(meta_path)
    edges = pd.read_csv(edges_path)
    significant_edges = pd.read_csv(significant_edges_path) if significant_edges_path.exists() else pd.DataFrame()
    network_edges = significant_edges if not significant_edges.empty else edges
    if network_edges.empty:
        return {}
    stim_pair = str(edges["stim_pair"].dropna().iloc[0]) if "stim_pair" in edges.columns and edges["stim_pair"].notna().any() else site_root.name
    fig_dir = site_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    detections_path = site_root / "detections.csv"
    detections = pd.read_csv(detections_path) if detections_path.exists() else pd.DataFrame()

    if not detections.empty and "peak_amplitude_uv" in detections.columns:
        try:
            zscore_table = zscore_metric_within_stim(
                detections.assign(stim_pair=stim_pair),
                "peak_amplitude_uv",
                absolute=True,
            )
            zscore_table.to_csv(site_root / "within_stim_peak_amplitude_zscores.csv", index=False)
            ax = plot_within_stim_zscore_bars(
                zscore_table,
                "peak_amplitude_uv",
                stim_pair=stim_pair,
                top_n=18,
                title=f"{stim_pair} peak-amplitude z-score across recording contacts",
            )
            path = fig_dir / "within_stim_peak_amplitude_zscores.png"
            ax.figure.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            figures["within_stim_peak_amplitude_zscores"] = str(path)
        except Exception as exc:
            figures["within_stim_peak_amplitude_zscores_error"] = str(exc)

    epoch_files = sorted(
        (site_root / "procdata").glob(f"*/A/epochs/{stim_pair}_A_*_epochs.h5"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if epoch_files:
        try:
            epochs = ep.Epochs.from_hdf(epoch_files[0])
            crp_config = CRPConfig(
                response_window=(0.01, min(0.3, epochs.tmax)),
                baseline_window=epochs.baseline or (-0.25, -0.05),
                permutation_n=100,
                random_state=0,
            )
            crp_table = run_crp_all(epochs, config=crp_config)
            crp_table.to_csv(site_root / "crp_scores.csv", index=False)
            ax = plot_crp_score_map(crp_table, top_n=20)
            path = fig_dir / "crp_score_map.png"
            ax.figure.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(ax.figure)
            figures["crp_score_map"] = str(path)

            crp_channel = choose_representative_channel(
                crp_table.rename(columns={"score": "peak_amplitude_uv"}).assign(consensus_ch=crp_table["significant"]),
                epochs,
            )
            crp_result = run_crp(epochs, crp_channel, config=crp_config)
            fig = plot_crp_summary(epochs, crp_channel, result=crp_result)
            path = fig_dir / "crp_curve_summary.png"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            figures["crp_curve_summary"] = str(path)

            if not detections.empty:
                sig_channel, nonsig_channel = choose_significant_and_nonsignificant_channels(detections, epochs.channels)
                fig = plot_crp_response_contrast(epochs, sig_channel, nonsig_channel, config=crp_config)
                path = fig_dir / "significant_vs_nonsignificant_crp_contrast.png"
                fig.savefig(path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                figures["significant_vs_nonsignificant_crp_contrast"] = str(path)

            figures.update(write_spectral_figure_suite(fig_dir, epochs, detections))
        except Exception as exc:
            figures["crp_refresh_error"] = str(exc)

    try:
        audit_path = site_root / "mni_coordinate_audit.csv"
        validate_mni_coordinates(elec_meta).to_csv(audit_path, index=False)
        figures["mni_coordinate_audit"] = str(audit_path)
    except Exception as exc:
        figures["mni_coordinate_audit_error"] = str(exc)

    response_matrix = response_matrix_from_edges(edges, elec_meta=elec_meta)
    response_matrix.to_csv(site_root / "response_source_matrix.csv")
    ax = plot_response_matrix_heatmap(edges, elec_meta=elec_meta, title=f"{stim_pair} response field")
    path = fig_dir / "response_source_matrix_heatmap.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["response_source_matrix_heatmap"] = str(path)

    ax = plot_coordinate_network(network_edges, elec_meta, title=f"{stim_pair} significant MNI coordinate response network")
    path = fig_dir / "coordinate_response_graph.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["coordinate_response_graph"] = str(path)

    fig = plot_template_brain_network(network_edges, elec_meta, title=f"{stim_pair} fsaverage cortical response network")
    path = fig_dir / "template_brain_response_network.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures["template_brain_response_network"] = str(path)

    try:
        display = plot_glass_brain_network(network_edges, elec_meta, title=None, edge_linewidth=1.45, edge_alpha=0.82)
        path = fig_dir / "glass_brain_response_network.png"
        display.savefig(path, dpi=300)
        display.close()
        figures["glass_brain_response_network"] = str(path)
    except Exception as exc:
        figures["glass_brain_response_network_error"] = str(exc)

    graph = graph_from_edges(network_edges, threshold=0.0)
    ax = plot_network(graph, title=f"{stim_pair} Kamada-Kawai response topology")
    path = fig_dir / "network_graph.png"
    ax.figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)
    figures["network_graph"] = str(path)

    try:
        html = fig_dir / "interactive_connectome.html"
        plot_interactive_connectome(
            network_edges,
            elec_meta,
            output_html=html,
            top_n=48,
            title=f"{stim_pair} significant response connectome",
        )
        figures["interactive_connectome"] = str(html)
    except Exception as exc:
        figures["interactive_connectome_error"] = str(exc)

    if epoch_files and detections_path.exists():
        try:
            html = fig_dir / "evoked_response_graph.html"
            plot_evoked_response_graph(
                ep.Epochs.from_hdf(epoch_files[0]),
                elec_meta,
                stim_pair=stim_pair,
                detections=detections,
                consensus_only=False,
                output_html=html,
                top_n=20,
                title=f"{stim_pair} evoked response graph",
            )
            figures["evoked_response_graph"] = str(html)
        except Exception as exc:
            figures["evoked_response_graph_error"] = str(exc)

    manifest_path = site_root / "figure_manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    else:
        existing = {}
    existing.update(figures)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return figures


def run_dataset(
    ds: PublicDataset,
    out_root: Path,
    min_events: int = 4,
    max_events: int = 8,
    max_sites: int = 1,
    max_channels: int = 32,
    make_figures: bool = False,
    prefer_cache: bool = False,
    source_manifest: dict | None = None,
) -> list[dict]:
    if source_manifest:
        if prefer_cache:
            raise ValueError("Pinned public recipes require verified source retrieval, not summary-cache reuse")
        ds = _pin_dataset(ds, source_manifest, dict(min_events=min_events, max_events=max_events,
                                                  max_sites=max_sites, max_channels=max_channels))
    if prefer_cache:
        summaries = _load_cached_summaries(out_root / ds.dataset_id, max_sites=max_sites)
        if summaries:
            if make_figures:
                for summary in summaries:
                    refresh_cached_network_figures(Path(summary["workspace"]))
                if len(summaries) > 1:
                    write_dataset_aggregate_figure_suite(out_root / ds.dataset_id, summaries)
            return summaries
    try:
        expected_events = source_manifest["objects"]["events_url"]["sha256"] if source_manifest else None
        events = download_events(ds, expected_events)
    except requests.RequestException:
        if source_manifest:
            raise
        summaries = _load_cached_summaries(out_root / ds.dataset_id, max_sites=max_sites)
        if not summaries:
            raise
        if make_figures:
            for summary in summaries:
                refresh_cached_network_figures(Path(summary["workspace"]))
            if len(summaries) > 1:
                write_dataset_aggregate_figure_suite(out_root / ds.dataset_id, summaries)
        return summaries
    blocks = select_event_blocks(events, min_events=min_events, max_sites=max_sites,
                                 stim_site=source_manifest["stim_site"] if source_manifest else None)
    summaries = [
        run_site(
            ds,
            site,
            block,
            out_root=out_root,
            max_events=max_events,
            max_channels=max_channels,
            make_figures=make_figures,
            source_manifest=source_manifest,
        )
        for site, block in blocks
    ]
    if make_figures and len(summaries) > 1:
        write_dataset_aggregate_figure_suite(out_root / ds.dataset_id, summaries)
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ERPy against public open BrainVision SPES signal slices.")
    parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    parser.add_argument("--out", default="validation/public_spes_work")
    parser.add_argument("--min-events", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--sites-per-dataset", type=int, default=1)
    parser.add_argument("--max-channels", type=int, default=32)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args(argv)
    out_root = Path(args.out).resolve()
    selected = DATASETS.values() if args.dataset == "all" else [DATASETS[args.dataset]]
    summaries = []
    for ds in selected:
        summaries.extend(
            run_dataset(
                ds,
                out_root,
                min_events=args.min_events,
                max_events=args.max_events,
                max_sites=args.sites_per_dataset,
                max_channels=args.max_channels,
                make_figures=args.figures,
            )
        )
    summary_df = pd.DataFrame(summaries)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_root / "validation_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
