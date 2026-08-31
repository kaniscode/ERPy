"""Auditable coordinate utilities for intracranial electrode metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd


def complete_contact_coordinates(
    contacts: pd.DataFrame,
    *,
    maximum_fit_rmse_mm: float = 2.5,
    maximum_contact_step_mm: float = 15.0,
) -> pd.DataFrame:
    """Fill an isolated missing contact from a well-fit linear lead trajectory.

    Contacts are grouped by patient, session, and the nonnumeric prefix of
    ``elec_label``. A three-dimensional linear fit is accepted only when at
    least four localized contacts are available, the fit error is within
    ``maximum_fit_rmse_mm``, and the estimated contact spacing is plausible.
    Interpolation or one-contact extrapolation is allowed; an entirely
    unlocalized lead is left unchanged.

    The returned table records ``mni_coordinate_source``, fit RMSE, and the
    number of localized contacts used. These provenance columns distinguish
    inferred positions from measured or imported coordinates.

    Parameters
    ----------
    contacts:
        Electrode metadata containing ``elec_label`` and, when available,
        ``mni_x``, ``mni_y``, and ``mni_z``. ``patient_id`` and ``session_id``
        are used to prevent fits from crossing acquisitions.
    maximum_fit_rmse_mm:
        Largest accepted three-dimensional root-mean-square fit residual.
    maximum_contact_step_mm:
        Largest accepted distance between adjacent fitted contacts. The lower
        bound is fixed at 0.5 mm.

    Returns
    -------
    pandas.DataFrame
        A copy of the input table with conservatively completed coordinates
        and explicit coordinate-provenance columns.
    """

    output = pd.DataFrame(contacts).copy()
    coord_cols = ["mni_x", "mni_y", "mni_z"]
    if "elec_label" not in output.columns:
        return output
    for col in coord_cols:
        if col not in output.columns:
            output[col] = np.nan
        output[col] = pd.to_numeric(output[col], errors="coerce")

    complete = output[coord_cols].notna().all(axis=1)
    if "mni_coordinate_source" not in output.columns:
        output["mni_coordinate_source"] = np.where(
            complete,
            "source_contact_metadata",
            "missing",
        )
    else:
        output["mni_coordinate_source"] = (
            output["mni_coordinate_source"].fillna("").astype(str)
        )
        output.loc[
            complete & output["mni_coordinate_source"].eq(""),
            "mni_coordinate_source",
        ] = "source_contact_metadata"
        output.loc[
            ~complete & output["mni_coordinate_source"].eq(""),
            "mni_coordinate_source",
        ] = "missing"
    for col in ("mni_coordinate_fit_rmse_mm", "mni_coordinate_fit_n_contacts"):
        if col not in output.columns:
            output[col] = np.nan

    parsed = output["elec_label"].astype(str).str.extract(
        r"^(?P<_lead>.*?)(?P<_contact>\d+)$"
    )
    output["_lead"] = parsed["_lead"]
    output["_contact"] = pd.to_numeric(parsed["_contact"], errors="coerce")
    identity_cols = [
        col for col in ("patient_id", "session_id") if col in output.columns
    ]
    group_cols = [*identity_cols, "_lead"]
    grouper: str | list[str] = group_cols[0] if len(group_cols) == 1 else group_cols

    for _, frame in output.dropna(subset=["_lead", "_contact"]).groupby(
        grouper,
        dropna=False,
        sort=False,
    ):
        known = frame[frame[coord_cols].notna().all(axis=1)].copy()
        missing = frame[frame[coord_cols].isna().any(axis=1)].copy()
        if len(known) < 4 or missing.empty:
            continue
        contact_numbers = known["_contact"].to_numpy(dtype=float)
        if np.unique(contact_numbers).size < 4:
            continue
        coefficients = np.vstack(
            [
                np.polyfit(
                    contact_numbers,
                    known[col].to_numpy(dtype=float),
                    deg=1,
                )
                for col in coord_cols
            ]
        )
        fitted = np.column_stack(
            [np.polyval(coefficients[axis], contact_numbers) for axis in range(3)]
        )
        observed = known[coord_cols].to_numpy(dtype=float)
        fit_rmse = float(
            np.sqrt(np.mean(np.sum((observed - fitted) ** 2, axis=1)))
        )
        step_mm = float(np.linalg.norm(coefficients[:, 0]))
        if (
            not np.isfinite(fit_rmse)
            or fit_rmse > float(maximum_fit_rmse_mm)
            or not 0.5 <= step_mm <= float(maximum_contact_step_mm)
        ):
            continue
        minimum_contact = float(np.nanmin(contact_numbers))
        maximum_contact = float(np.nanmax(contact_numbers))
        for index, row in missing.iterrows():
            contact_number = float(row["_contact"])
            if (
                contact_number < minimum_contact - 1
                or contact_number > maximum_contact + 1
            ):
                continue
            predicted = np.array(
                [
                    np.polyval(coefficients[axis], contact_number)
                    for axis in range(3)
                ],
                dtype=float,
            )
            if (
                not np.isfinite(predicted).all()
                or abs(predicted[0]) > 100
                or abs(predicted[1]) > 150
                or abs(predicted[2]) > 120
            ):
                continue
            for col, value in zip(coord_cols, predicted):
                if pd.isna(output.at[index, col]):
                    output.at[index, col] = float(value)
            output.at[index, "mni_coordinate_source"] = "linear_lead_trajectory"
            output.at[index, "mni_coordinate_fit_rmse_mm"] = fit_rmse
            output.at[index, "mni_coordinate_fit_n_contacts"] = int(len(known))

    inferred = output["mni_coordinate_source"].eq("linear_lead_trajectory")
    output.loc[inferred, coord_cols] = output.loc[inferred, coord_cols].round(6)
    output.loc[inferred, "mni_coordinate_fit_rmse_mm"] = pd.to_numeric(
        output.loc[inferred, "mni_coordinate_fit_rmse_mm"],
        errors="coerce",
    ).round(6)
    return output.drop(columns=["_lead", "_contact"])
