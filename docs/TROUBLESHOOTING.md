# Troubleshooting

## `No module named ERPy`

Activate the same environment in which ERPy was installed, then verify it:

```text
python -m pip show erpy-neuro
python -c "import ERPy; print(ERPy.__version__)"
```

If `python -m pip show erpy` returns an Erlang-related package, uninstall that
unrelated package and install `erpy-neuro` as described in the getting-started
guide.

```text
python -m pip uninstall erpy
```

Run `python -m pip show erpy-neuro` and `python -c "import ERPy"` in the same
terminal. Using `pip` by itself can target a different Python installation;
the documentation deliberately uses `python -m pip` to keep them together.

## Linux cannot create the environment

Some Linux distributions package the environment module separately. If
`python3 -m venv .venv` reports that `ensurepip` or `venv` is unavailable,
install the `python3-venv` package with your distribution's normal software
manager, then repeat the environment command. On a managed hospital or
university computer, ask the local administrator rather than using a system
Python with elevated permissions.

## PowerShell blocks environment activation

For the current PowerShell session only:

```text
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Alternatively, call `.\.venv\Scripts\python.exe` directly.

## GitHub asks for a username or password

The public ERPy HTTPS clone and release download should not require a GitHub
login. Do not enter your normal GitHub password. Confirm that the URL is
exactly `https://github.com/kaniscode/ERPy` and that the `v1.0.0` release page
opens in a browser. A `404`, `repository not found`, or credential prompt can
mean that the repository or tag has not been published yet, the URL was copied
incorrectly, or your network is intercepting GitHub access.

If Git itself is unavailable, use **Source code (zip)** under **Assets** on the
release page. GitHub does not provide a normal **Download ZIP** action for just
`notebooks/examples`; download the complete release so `_synthetic.py` stays
beside the notebooks.

## Jupyter cannot find the quick-start notebook

Activate the ERPy environment, enter the complete source folder, and run
Jupyter from there:

```text
cd ERPy-source
python -m jupyter lab notebooks/examples/00_quickstart.ipynb
```

If that path does not exist, the release was extracted elsewhere or only one
notebook was downloaded. Locate the folder containing `pyproject.toml`, keep
its complete `notebooks/examples` subfolder, and launch Jupyter with the
matching path.

## Jupyter imports the wrong ERPy version

In the first notebook cell, check the interpreter and package version:

```python
import sys
import ERPy

print(sys.executable)
print(ERPy.__version__)
```

The version should be `1.0.0`, and the interpreter path should point into the
`.venv` folder created for this tutorial. If it does not, close JupyterLab,
activate that environment, and restart it with `python -m jupyter lab`. Avoid
starting Jupyter from an unrelated desktop shortcut.

## Optional file format cannot be read

- EDF, BDF, BrainVision, or BIDS: install the `edf` extra.
- NWB: install the `nwb` extra.
- Plotly/Nilearn brain visualizations: install the `viz` extra.
- MNE/PyVista surfaces: install the `surface` and `viz` extras.

From the source folder containing `pyproject.toml`, use:

```text
python -m pip install --upgrade ".[edf,nwb,viz,surface]"
```

Then rerun `python -c "import ERPy; print(ERPy.__version__)"` in the same
environment.

## `erpy-init` is not recognized

First activate the environment used to install ERPy. The equivalent module
command also works when the console-script folder is not on `PATH`:

```text
python -m ERPy.cli --config ./config.yaml --data-root ./data
python -m ERPy.cli show-config --config ./config.yaml
```

## Update or uninstall ERPy

Activate the intended environment before changing it. To update from an
extracted or cloned release, open a terminal in its source folder and run:

```text
python -m pip install --upgrade ".[edf,nwb,viz]"
python -c "import ERPy; print(ERPy.__version__)"
```

To uninstall the distribution (whose import name is still `ERPy`):

```text
python -m pip uninstall erpy-neuro
```

Uninstalling does not delete recordings, configuration files, cached results,
or exported figures.

## ERPy cannot find a recording

Check the three layers separately:

1. The path in `config.yaml` points to the intended metadata table.
2. `patient_id` and `session_id` match exactly in all tables.
3. `raw_file` exists and is readable from the current computer.

Avoid moving cached output between computers with absolute source paths. A
portable project uses relative paths beneath a common data root.

## No events are detected

Inspect the raw trigger or stimulation channels and choose the most explicit
event source available. Prefer supplied timestamps, a trigger channel, or file
annotations over waveform-artifact inference. If inference is necessary,
compare `auto`, `adjacent`, `consensus`, and `matched_filter` and inspect the
saved event table before epoching.

## Epochs contain a large spike at time zero

Time zero is always stimulation onset, not necessarily the zero-amplitude
anchor. Inspect:

```python
epochs.zero_time_report()
epochs.post_artifact_anchor_report()
```

If the report shows unresolved onset artifact, revise the blanking interval or
anchor settings and rerun preprocessing. Do not interpret peaks that overlap
the excluded artifact interval.

## Too few clean trials remain

Review `epochs.flag_artifacts()` rather than lowering thresholds immediately.
Common causes are incorrect acquisition rail limits, a contaminated reference,
movement across many channels, or an event-time offset. The CRP-energy detector
requires at least eight clean matched trials by default and returns
`insufficient_data` when that requirement is not met.

## `p_crp` does not match the standalone CRP result

The names are retained for file compatibility but refer to different tests.
On a `crp_energy` row, `p_crp` stores the fixed-window, whole-trial sign-flip
reproducibility value \(p_R\), and `p_energy` stores the separately demeaned
matched-energy value \(p_E\). The conjunction is
\(p_{\mathrm{joint}}=\max(p_R,p_E)\), followed by adjustment across eligible
recording contacts in `q_joint`.

The standalone `crp_significance` row instead reports the published
data-selected-duration, balanced-projection extraction test in `p_value` and
its channelwise adjusted value in `crp_q_value`. It is an exploratory
comparator with a different operational definition, so its p-value will
generally differ from `p_crp` on the primary row.

## A plot is blank or labels do not match

Check `epochs.channels`, the stimulation-pair spelling, and the requested time
window. Visualization functions return Matplotlib or Plotly figure objects;
call `show()` only in interactive sessions and use `save_analysis_figure` for
auditable files.

## An older HDF5 cache no longer opens

Version 1.0 stores audit metadata as compressed JSON. Legacy cache metadata may
contain Python pickle data and is rejected by default. Only if the file was
created by you and has remained under your control, load it with the explicit
unsafe legacy option, then save it again in the current format. Never enable
legacy pickle loading for downloaded or untrusted files.

## Still stuck

When opening a GitHub issue, include ERPy/Python versions, the smallest
reproducible example, the full error message, and a synthetic or deidentified
input. Do not attach recordings or tables containing protected information.
