# Installing ERPy

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md) · [Documentation index](README.md).

This page is the complete installation reference: which version each command
installs, the optional feature sets, and how to create a project. If you are new
to Python, follow the step-by-step [getting started guide](GETTING_STARTED.md)
instead; it covers the same ground with separate macOS/Linux and Windows
instructions.

## Requirements

- Python 3.9 or newer.
- `pip` and, for the commands that install directly from GitHub, Git.
- An isolated environment (`venv` or conda) is strongly recommended.

ERPy's import name is `ERPy`, while its installable distribution is
`erpy-neuro`. The unrelated name `erpy` on the Python Package Index is an
Erlang communication library; do not use `pip install erpy` for this project.

## Which version am I installing?

This source checkout is `1.1.0rc1`. It includes the current pipeline, public-data
examples and optional negative-N1 classifier. Install the complete source folder:

```bash
python -m pip install ".[edf,nwb,viz]"
git rev-parse HEAD
```

Record the printed Git commit with the results. The source installation reports
`1.1.0rc1`; the tagged installation below retrieves historical `1.0.0`. The
commands using `main` retrieve the latest public branch. Confirm its version
before use, and use an exact recorded commit for reproducing published results.

The package declares Python 3.9 or newer. This release candidate's checks were
run on Python 3.12.5; an expanded Python-version matrix has not been verified
for these changes. Use the recorded environment for numerical reproduction.

## Install the tagged release

For the tagged GitHub release, create an isolated environment and install the
features needed for EDF/BIDS, NWB, and the full visualization gallery. A public
GitHub release does not require a GitHub account or password.

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "erpy-neuro[edf,nwb,viz] @ git+https://github.com/kaniscode/ERPy.git@v1.0.0"
python -c "import ERPy; print(ERPy.__version__)"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "erpy-neuro[edf,nwb,viz] @ git+https://github.com/kaniscode/ERPy.git@v1.0.0"
python -c "import ERPy; print(ERPy.__version__)"
```

The final command should print `1.0.0`. The command above installs directly
from the immutable `v1.0.0` tag and therefore requires Git. It installs the
library but does not create a local copy of the notebooks.

## Install the current source and notebooks

For the current worked notebooks, beginning with the actual deidentified cohort
recording, clone `main`. Those examples postdate the historical `v1.0.0` tag:

```bash
git clone --branch main --depth 1 https://github.com/kaniscode/ERPy.git ERPy-source
python -m pip install "./ERPy-source[edf,nwb,viz]"
python -m pip install jupyterlab
python -m jupyter lab ERPy-source/notebooks/examples/00_quickstart.ipynb
```

The lead notebook's stored figures can be viewed directly. Reexecution requires
the authorized trial export and `ERPY_COHORT_EXAMPLE_DIR` configured before
starting Jupyter. Without Git, download the
[current main source ZIP](https://github.com/kaniscode/ERPy/archive/refs/heads/main.zip),
extract the complete folder, rename it `ERPy-source`, and use the same install
command shown above. A step-by-step guide for readers who are new to Python,
with separate macOS/Linux and Windows commands, is in
[`docs/GETTING_STARTED.md`](GETTING_STARTED.md).

## Development install

For a local source checkout used for development:

```bash
python -m pip install -e ".[dev,edf,nwb,viz]"
```

## Optional features

Install only the extras you need; they can be combined, for example
`".[edf,nwb,viz]"`.

| Extra | Adds | Needed for |
| --- | --- | --- |
| `edf` | MNE | EDF/BDF recordings and BIDS projects that contain them |
| `nwb` | PyNWB | NWB and DANDI recordings |
| `viz` | seaborn, Nilearn, Plotly | Coordinate-based brain graphs and interactive figures |
| `surface` | MNE, PyVista | High-fidelity FreeSurfer surface rendering |
| `validation` | MNE, requests, PyNWB | The public-data validation scripts |
| `n1-training` | scikit-learn | Refitting the optional supervised N1 models |
| `dev` | pytest, ruff, build tools | Running the tests and contributing |

Minimal CSV workflows do not require MNE. EDF/BIDS workflows should install the `edf` extra:

```bash
python -m pip install -e ".[edf]"
```

NWB workflows should install the `nwb` extra:

```bash
python -m pip install -e ".[nwb]"
```

Coordinate-based brain graphs use the `viz` extra. High-fidelity FreeSurfer/MNE/PyVista surface rendering is optional:

```bash
python -m pip install -e ".[viz]"
python -m pip install -e ".[viz,surface]"
```

## Create a project

Bootstrap a portable config and metadata stubs:

```bash
erpy-init --config ./config.yaml --data-root ./data
erpy-init show-config --config ./config.yaml
```

These commands create a project-local configuration file and blank metadata
tables; they never upload recording data.

## Updating an existing installation

Reinstall from the newer source folder or tag into the same environment:

```bash
python -m pip install --upgrade "./ERPy-source[edf,nwb,viz]"
python -c "import ERPy; print(ERPy.__version__)"
```

Epoch caches are bound to the installed source code. After any update, the
first `patient.epoch(...)` or `patient.analyze(...)` call for an acquisition
recomputes its epochs instead of reusing the cache written by the previous
version. Numerical results are unaffected unless the release notes say
otherwise.

## See also

- [Getting started for new Python users](GETTING_STARTED.md)
- [Troubleshooting installation problems](TROUBLESHOOTING.md)
- [Analysis workflow](WORKFLOW.md)
