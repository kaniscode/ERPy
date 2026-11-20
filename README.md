# ERPy

**An auditable pipeline for detecting and analyzing stimulation-evoked responses in intracranial recordings.**

ERPy takes single-pulse electrical stimulation recordings (stereo-EEG or ECoG)
from raw files to reviewable results. Starting from EDF/BDF, BrainVision, NWB,
BIDS, or CSV data, it:

- finds stimulation events and preprocesses the continuous recording;
- extracts event-locked epochs and screens every trial-by-channel response for artifact;
- detects evoked responses with a primary reproducibility-and-energy test, reported alongside published comparator detectors;
- provides waveform, spectral, connectivity, and network analyses with matching figures;
- saves result tables together with an audit record of every setting that produced them.

Each automated decision is kept inspectable (the anchor used to zero each
trial, the artifact reason for each rejected response, each detector's
component statistics) so that a response label can always be traced back to
the waveform evidence behind it.

> ERPy is research software. It does not make clinical decisions and is not a
> medical device. Inspect waveforms, acquisition settings, and quality-control
> reports before interpreting any automated response label.

## Installation

ERPy needs Python 3.9 or newer. Install it into an isolated environment from a
copy of this repository:

```bash
git clone https://github.com/kaniscode/ERPy.git
cd ERPy
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[edf,nwb,viz]"
python -c "import ERPy; print(ERPy.__version__)"
```

The bracketed extras are optional: `edf` reads EDF/BDF and BIDS recordings,
`nwb` reads NWB files, and `viz` adds brain and interactive figures. A plain
`python -m pip install .` is enough for CSV-based projects.

To install without cloning (the example notebooks are then not downloaded):

```bash
python -m pip install "erpy-neuro[edf,nwb,viz] @ git+https://github.com/kaniscode/ERPy.git@main"
```

The package is distributed as `erpy-neuro` and imported as `ERPy`. It is not
published on the Python Package Index; the unrelated PyPI project named `erpy`
is a different library. The `main` branch holds the current release candidate,
and the `v1.0.0` tag the previous release.

More detail: [installation reference](docs/INSTALLATION.md) (versions, all
extras, development install) and a
[step-by-step guide for new Python users](docs/GETTING_STARTED.md).

## A first analysis

```python
import ERPy as ep

patient = ep.Patient("EXAMPLE_PATIENT", config_path="config.yaml")
result = patient.analyze("EXAMPLE_SESSION", "STIM_A_1_2")

result.primary_significant            # responses passing the primary test and artifact QC
result.save("analysis_tables")        # CSV tables and a JSON audit record, written locally
result.clean_epochs.plot.summary("CONTACT_B1", detections=result.qc_detections)
```

`erpy-init --config ./config.yaml --data-root ./data` creates the
configuration file and blank metadata tables for a new project. The
[analysis workflow guide](docs/WORKFLOW.md) walks through this example and the
individual steps behind it.

## Documentation

| Guide | Contents |
| --- | --- |
| [Getting started](docs/GETTING_STARTED.md) | From installing Python to a first analysis, for readers new to Python. |
| [Installation](docs/INSTALLATION.md) | Versions, optional features, development install, creating a project. |
| [Analysis workflow](docs/WORKFLOW.md) | The one-call analysis, its result object, and the step-by-step equivalent. |
| [Data formats](docs/DATA_FORMATS.md) | Project configuration, metadata tables, BIDS/OpenNeuro and NWB import. |
| [Events](docs/EVENTS.md) | Creating and detecting stimulation event times. |
| [Preprocessing and quality control](docs/PREPROCESSING_AND_QC.md) | Pipelines, referencing, bad channels, artifact responses, zero anchoring. |
| [Response detection](docs/DETECTION.md) | The primary detector, comparator detectors, output columns, CRP, N1. |
| [Spectral analysis](docs/SPECTRAL.md) | Power, time-frequency, phase locking, coherence, cross-frequency coupling. |
| [Visualization](docs/VISUALIZATION.md) and [gallery](docs/VISUALIZATION_GALLERY.md) | Plotting functions and the notebooks that demonstrate them. |
| [Network analysis](docs/NETWORKS.md) | Edge tables, graph metrics, brain and connectome figures. |
| [Methods](docs/METHODS.md) | Mathematical definitions of every quantity. |
| [Validation](docs/VALIDATION.md) | Tests, synthetic benchmarks, public-data and expert-label evaluations. |
| [API reference](docs/API_REFERENCE.md) | Every public class and function. |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | Common installation and analysis problems. |

The [documentation index](docs/README.md) lists every guide, and
[`notebooks/examples`](notebooks/examples) contains executed notebooks whose
figures can be read directly on GitHub. Analyses run locally and upload
nothing; see [data privacy](docs/DATA_PRIVACY.md).

## Citation

```bibtex
@software{kanungo_erpy_2026,
  author = {Kanungo, Ishan},
  title = {ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis},
  year = {2026},
  version = {1.1.0rc1},
  url = {https://github.com/kaniscode/ERPy},
  license = {MIT}
}
```

## License and contributing

MIT; see [`LICENSE`](LICENSE). Contributions are welcome; see
[`CONTRIBUTING.md`](CONTRIBUTING.md).
