# Getting started with ERPy

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

This guide assumes that you are comfortable opening files and folders but may
be new to Python. ERPy is research software for stimulation-evoked responses in
intracranial recordings. It is not a medical device and its automated labels
must be reviewed alongside the waveforms and quality-control reports.

Start with the [worked cohort recording](../notebooks/examples/00_quickstart.ipynb)
to see actual deidentified trials and executed figures from the CNS/ACC–PAG
analysis cohort. Its source recording is restricted; viewing the stored
figures does not require downloading that input.

## 1. Install Python

Install Python from [python.org](https://www.python.org/downloads/). ERPy
declares Python 3.9 or newer. The current release candidate was checked on
Python 3.12.5; use Python 3.12 to follow its recorded execution environment.
The expanded Python-version matrix has not been verified for these changes.
During Windows installation, select **Add Python to PATH**.

Open Terminal (macOS/Linux) or PowerShell (Windows) and check the installation:

```text
python3 --version
```

On Windows, use `python` instead of `python3` if needed.

## 2. Make a tutorial folder and isolated environment

An environment keeps ERPy separate from other Python projects.

macOS or Linux:

```text
mkdir erpy_tutorial
cd erpy_tutorial
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```text
mkdir erpy_tutorial
cd erpy_tutorial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

When the environment is active, the prompt normally begins with `(.venv)`.

## 3. Download and install ERPy

The import name is `ERPy`; the installable distribution is `erpy-neuro`. Do
not run `pip install erpy`: that name belongs to an unrelated project.

Choose one of the routes below. Routes A and B are best for a first use because
they keep the current notebooks and their adjacent input helpers together.
Do not download an individual notebook by itself. The worked cohort example
and optional analysis APIs are supplied with the current `1.1.0rc1` source.
The routes using `main` retrieve the latest public branch; confirm the version
before continuing. A supplied complete source folder can be installed with the
same command. Record the exact Git revision when reproducing results.

### Route A: clone the current source and worked notebooks with Git

With Git installed, clone current `main` into the tutorial folder:

```text
git clone --branch main --depth 1 https://github.com/kaniscode/ERPy.git ERPy-source
```

No GitHub account, username, or password is needed for a public repository. If
the command succeeds, continue to **Install the downloaded folder** below.

### Route B: download the current source as a ZIP

If Git is unavailable, download the
[current main source ZIP](https://github.com/kaniscode/ERPy/archive/refs/heads/main.zip).
Move the ZIP into the
`erpy_tutorial` folder, extract it, and rename the extracted directory
`ERPy-source`. On Windows, use **Extract All** before renaming it; do not try to
install from inside the unopened ZIP.

### Install the downloaded folder

From the tutorial folder, install the common EDF/BIDS, NWB, and visualization
features:

```text
python -m pip install "./ERPy-source[edf,nwb,viz]"
```

### Route C: install the historical tagged library directly

If you only want the library, have Git installed, and do not need a local copy
of the current notebooks or development APIs, install historical `1.0.0`
directly from the exact tag:

```text
python -m pip install "erpy-neuro[edf,nwb,viz] @ git+https://github.com/kaniscode/ERPy.git@v1.0.0"
```

This route does not create `ERPy-source`; use route A or B before continuing to
the notebook section.

The end-user extras are `edf` for EDF/BDF/BrainVision/BIDS, `nwb` for NWB,
`viz` for Plotly/Nilearn/Seaborn views, `surface` (used with `viz`) for optional
MNE/PyVista surface rendering, and `validation` for the public-data validation
workflows. The current-source `n1-training` extra adds optional scikit-learn
training; N1 runtime prediction does not need it. The `dev` extra adds
contributor test/build tools. A minimal CSV
workflow can omit all extras.

Before executing the public-data cells in notebook 07, add their download and
validation dependencies from the source folder:

```text
python -m pip install "./ERPy-source[validation]"
```

If you used route C, install the validation extra from the same tag instead:

```text
python -m pip install "erpy-neuro[validation] @ git+https://github.com/kaniscode/ERPy.git@v1.0.0"
```

Confirm the installation:

```text
python -m pip show erpy-neuro
python -c "import ERPy; print(ERPy.__version__)"
```

The prepared candidate should print `1.1.0rc1`; route C's historical tag should
print `1.0.0`. A moving `main` checkout may have a later version, so record its
Git commit and package version with your results.

If it prints another version or reports `No module named ERPy`, stop here and
use [TROUBLESHOOTING.md](TROUBLESHOOTING.md). The most common cause is that the
environment was not active in the terminal where the install command ran.

## 4. Try a teaching notebook

If you used route A or B, enter the downloaded source folder, install
JupyterLab, and start the quick-start notebook while the environment is still
active. For reexecution, set `ERPY_COHORT_EXAMPLE_DIR` to the authorized
method-example directory in that environment before starting Jupyter:

```text
cd ERPy-source
python -m pip install jupyterlab
python -m jupyter lab notebooks/examples/00_quickstart.ipynb
```

JupyterLab should open in a web browser. The lead notebook 00 displays an actual
deidentified CNS/ACC–PAG cohort recording. Rerunning it requires authorized
access to the documented trial export and `ERPY_COHORT_EXAMPLE_DIR` set before
starting Jupyter. Its stored figures are available without those restricted
inputs. Teaching notebooks 01 through 06 create deterministic synthetic data;
use **Run > Run All Cells** to reproduce them without participant recordings.
Notebook 07 runs a small public ds003708 recording subset when you execute its
cells. Notebook 08 applies a participant-held-out N1 development model to
actual public ds004774 derived features and needs no raw download. All nine
notebooks include stored tables and
figures for browsing directly on GitHub; raw downloads remain outside version
control. The notebooks cover
waveforms, detection and QC, spectral analysis, networks, interactive brain
views, and figure export. Keep the terminal open while JupyterLab is running;
press `Ctrl+C` in that terminal when you are finished.

Fresh execution of the prepared candidate reports `1.1.0rc1`. Earlier stored
notebook outputs retain the versions under which they were actually executed;
they have not been relabeled as current execution. Because JupyterLab was
started with `python -m jupyter`, its default Python kernel comes from the
active environment. If it differs from your installed version, use **Kernel >
Change Kernel > Python 3** and rerun the first cell; if it remains wrong, close
JupyterLab and follow the environment checks in the troubleshooting guide.

## 5. Create a project for your own data

Stop JupyterLab, run `cd ..` to return to the tutorial folder (or open another
project folder outside `ERPy-source`), and keep the environment active. Then
run:

```text
erpy-init --config ./config.yaml --data-root ./data
erpy-init show-config --config ./config.yaml
```

This creates a configuration file and blank metadata tables. It does not read,
copy, or upload recordings. Fill the tables as follows:

| Table | Required columns | Purpose |
| --- | --- | --- |
| `raw_metadata.csv` | `patient_id`, `session_id`, `raw_file` | Locates each recording. |
| `stim_metadata.csv` | `patient_id`, `session_id`, `stim_pair`, `stim_start`, `stim_stop`, `stim_freq` | Describes stimulation blocks. |
| `electrode_metadata.csv` | `patient_id`, `session_id`, `elec_label` | Describes contacts; anatomy and MNI coordinates are optional. |

Paths may be absolute or relative to the project. Keep raw recordings outside
the Git repository. Never commit identifiers, dates, free-text clinical notes,
credentials, or server paths.

## 6. Run one analysis

Create a file named `first_analysis.py`:

```python
import ERPy as ep

patient = ep.Patient("EXAMPLE_PATIENT", config_path="config.yaml")
result = patient.analyze(
    session_id="EXAMPLE_SESSION",
    stim_pair="STIM_A_1_2",
    pipeline="blank_filt",
)

print(result.primary_significant)
result.save("analysis_tables")
result.clean_epochs.plot.summary(
    "CONTACT_B1",
    detections=result.qc_detections,
)
```

Run it from the same folder:

```text
python first_analysis.py
```

The `result.save("analysis_tables")` line writes CSV tables and a JSON audit
record into the local `analysis_tables` folder. It does not upload anything.
If an analysis uses an opaque label such as `ERPY-RR-<UUID>` to refer to an
approved derivative, that is an author-assigned local governance label, not an
ERPy cloud destination. ERPy does not create or register it automatically. See
[DATA_PRIVACY.md](DATA_PRIVACY.md) for the optional local UUIDv4 convention and
the distinction between release-record labels, the software compatibility
checkpoint, and local cache hashes.

`EXAMPLE_PATIENT`, `EXAMPLE_SESSION`, `STIM_A_1_2`, and `CONTACT_B1` are
fictional placeholders. Replace them with exact values from your metadata. The
primary detector is the CRP-energy conjunction. Its reproducibility component
is the fixed-window whole-trial sign-flip value \(p_R\), stored in the
historical field `p_crp`; its separately demeaned matched-energy component is
\(p_E\), stored in `p_energy`. The standalone `crp_significance` comparator
instead uses a data-selected-duration extraction test. Historical detectors
are reported as exploratory comparisons rather than pooled into an opaque label.

The one-call analysis always evaluates trial-by-channel response artifacts.
The default `blank_filt` pipeline does not automatically screen for channels
that are persistently bad throughout the continuous recording. Use
`pipeline="blank_filt_reject"` (or an explicit `reject_bad_channels` pipeline
step) when that optional screening is appropriate, and review its decisions
before interpreting or excluding a contact.

## 7. Review before interpreting

For every stimulation site:

1. Confirm that detected event times align with stimulation artifacts.
2. Inspect the automatic trial-level artifact report and, when configured, the
   optional persistent-channel report.
3. Check the post-artifact anchor and the number of clean trials.
4. View individual trials, mean/SEM, and detector components.
5. Record the software version, parameters, input provenance, and exclusions.

Useful audit calls are:

```python
result.epochs.zero_time_report()
result.epochs.post_artifact_anchor_report()
artifact_report = result.epochs.flag_artifacts()
```

## 8. Update or uninstall

To update later, download or clone the newer published tag, rename its complete
folder `ERPy-source-new`, activate this same environment, and install from that
new folder:

```text
python -m pip install --upgrade "./ERPy-source-new[edf,nwb,viz]"
python -c "import ERPy; print(ERPy.__version__)"
```

Tags identify fixed releases, so do not expect the contents of `v1.0.0` to
change. To remove ERPy from the active environment:

```text
python -m pip uninstall erpy-neuro
```

This removes the `ERPy` package and `erpy-init` command but does not delete
your recordings, configuration, outputs, source download, or JupyterLab.

The mathematical definitions are in [METHODS.md](METHODS.md), every public
callable is listed in [API_REFERENCE.md](API_REFERENCE.md), and common problems
are covered in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
