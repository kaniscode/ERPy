from __future__ import annotations

import numpy as np
import pandas as pd

import ERPy as ep


def test_continuous_npz_roundtrip_preserves_channels_time_and_metadata(tmp_path):
    sfreq = 2000.0
    index = pd.date_range(
        "2026-01-01 12:00:00",
        periods=4000,
        freq=pd.to_timedelta(1 / sfreq, unit="s"),
    )
    time = np.arange(len(index)) / sfreq
    frame = pd.DataFrame(
        {
            "A": np.sin(2 * np.pi * 10 * time),
            "B": np.cos(2 * np.pi * 30 * time),
            "C": np.sin(2 * np.pi * 70 * time + 0.2),
        },
        index=index,
    )
    frame.attrs.update({"sfreq": sfreq, "signal_units": "uV"})

    resampled = ep.resample_continuous(frame, 250.0)
    path = tmp_path / "continuous_250hz.npz"
    ep.save_continuous_npz(
        resampled,
        path,
        dtype="float32",
        compressed=True,
        metadata={"acquisition_id": "run001"},
    )
    loaded = ep.load_continuous_npz(path)

    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp.npz"))
    assert loaded.columns.tolist() == frame.columns.tolist()
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index[0] == frame.index[0]
    assert loaded.attrs["sfreq"] == 250.0
    assert loaded.attrs["signal_units"] == "uV"
    assert loaded.attrs["acquisition_id"] == "run001"
    assert set(loaded.dtypes.astype(str)) == {"float32"}
    np.testing.assert_allclose(
        loaded.to_numpy(dtype=float),
        resampled.to_numpy(dtype=np.float32).astype(float),
        rtol=0,
        atol=0,
    )
