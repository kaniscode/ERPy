# ERPy documentation

Project: [ERPy: an auditable complete pipeline for intracranial stimulation response detection and analysis](../README.md).

## Start here

| Guide | Read it when |
| --- | --- |
| [Getting started](GETTING_STARTED.md) | You are new to Python or want a guided first run. |
| [Installation](INSTALLATION.md) | You need the details: versions, optional features, development install, project creation. |
| [Analysis workflow](WORKFLOW.md) | You want to see one acquisition go from configuration to saved results. |
| [Troubleshooting](TROUBLESHOOTING.md) | Something does not install, load, or look right. |

## Using ERPy

| Guide | Contents |
| --- | --- |
| [Data formats](DATA_FORMATS.md) | Project configuration, metadata tables, BIDS/OpenNeuro and NWB import. |
| [BIDS, OpenNeuro, and NWB](BIDS_OPENNEURO.md) | What the importers read and write, and their limits. |
| [Events](EVENTS.md) | Creating and detecting stimulation event times. |
| [Preprocessing and quality control](PREPROCESSING_AND_QC.md) | Pipelines, referencing, bad channels, artifact responses, zero anchoring. |
| [Response detection](DETECTION.md) | The primary detector, comparator detectors, output columns, CRP, N1. |
| [Spectral analysis](SPECTRAL.md) | Power, time-frequency, phase locking, coherence, cross-frequency coupling. |
| [Visualization](VISUALIZATION.md) | Waveform views, response maps, saving figures. |
| [Visualization gallery](VISUALIZATION_GALLERY.md) | Every plotting function mapped to an executed notebook. |
| [Network analysis](NETWORKS.md) | Edge tables, graph metrics, brain and connectome figures. |
| [Example notebooks](../notebooks/examples/README.md) | Executed notebooks with stored figures. |
| [API reference](API_REFERENCE.md) | Every public class and function, generated from the source. |
| [Data privacy](DATA_PRIVACY.md) | What stays local, release-record labels, and safe sharing. |

## Methods and evidence

| Guide | Contents |
| --- | --- |
| [Methods](METHODS.md) | Mathematical definitions of every quantity. |
| [Equation reference](EQUATION_REFERENCE.md) | Equation-to-function mapping and confirmed corrections. |
| [Validation](VALIDATION.md) | Unit tests, synthetic benchmarks, public-data and cohort validation. |
| [Benchmark overview](BENCHMARK_OVERVIEW.md) | What each benchmark and public-data check establishes. |
| [External validation](EXTERNAL_VALIDATION.md) | Evaluation against released expert annotations. |
| [Detector-annotation agreement](ANNOTATION_AGREEMENT.md) | How broad-response detection relates to N1 annotations. |
| [Workflow concordance](WORKFLOW_CONCORDANCE.md) | Regression checks on the public interface. |

## Optional and experimental methods

| Guide | Contents |
| --- | --- |
| [Label-free N1 detection](N1_LABEL_FREE.md) | Fixed-rule negative-N1 detection without training labels. |
| [Secondary supervised N1 models](N1_DEVELOPMENT.md) | Developmental negative-N1 classifiers. |
| [Tests across several response windows](MULTISCALE_DEVELOPMENT.md) | The multiple-window experiment and its negative result. |

## Names changed in 1.1

Version 1.1 introduced shorter, conventional names. The earlier names remain
available, so existing scripts and notebooks run unchanged.

| Earlier name | Preferred name |
| --- | --- |
| `epochs.detect_erp_all(...)` | `epochs.detect_all(...)` |
| `epochs.detect_erp(method, ...)` | `epochs.detect(method, ...)` |
| `epochs.get_mean_waveform()`, `get_std_waveform()`, `get_sem_waveform()` | `epochs.mean()`, `epochs.std()`, `epochs.sem()` |
| `recording.events_for(stim_pair)` | `recording.detect_events(stim_pair)` |
| `patient.sessions()` | `patient.session_ids` |
| `dataloader.CONFIG_PATH`, `RAWDATA_PATH`, `PROCDATA_PATH`, `PATIENT_PATH`, `SESSION_PATHS` | `dataloader.config_path`, `rawdata_path`, `procdata_path`, `patient_path`, `session_paths` |
| `dataloader.RAW_META_PATH`, `STIM_META_PATH`, `ELEC_META_PATH`, `EVENT_PATH` | `dataloader.raw_meta_path`, `stim_meta_path`, `elec_meta_path`, `event_path` |
| `dataloader.RAW_META_DF`, `STIM_META_DF`, `ELEC_META_DF` | `dataloader.all_raw_meta`, `all_stim_meta`, `all_elec_meta` |
| `dataloader.RAW_CACHE_FORMAT`, `RAW_CACHE_DTYPE`, `RAW_STAGING_PATH` | `dataloader.raw_cache_format`, `raw_cache_dtype`, `raw_staging_path` |
