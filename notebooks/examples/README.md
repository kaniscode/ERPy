# Example notebooks

Executed notebooks with stored tables and figures that can be read directly on GitHub.

The lead [quick-start notebook](00_quickstart.ipynb) uses an
actual deidentified example cohort recording, with source trial QC,
waveforms, and single-contact reproducibility/energy results. Its stored figures
are viewable directly on GitHub; rerunning requires the authorized local trial
export configured through `ERPY_COHORT_EXAMPLE_DIR`. Restricted signals and
private source locators are not distributed. Compact notebooks 01 through 06
under [`notebooks/examples`](.) cover every public visualization
family using deterministic synthetic data. The gallery index maps every
plotting function to a notebook and identifies optional Nilearn, Plotly, MNE,
or PyVista requirements. Notebook 07 runs a compact public Miller/Hermes
OpenNeuro `ds003708` example and includes its input provenance, detector
tables, and figures. Raw downloads remain outside version control.

[Notebook 08](08_n1_development.ipynb) applies the secondary
negative-N1 development classifier to actual public `ds004774` derived features
and matching stored inference values. It verifies source/model hashes, uses a
model that excluded the example participant, and demonstrates portable model
save/load. The example needs no raw download or scikit-learn. Notebook outputs
retain their actual execution versions; earlier examples are not relabeled as
having been executed under the current release candidate.

| Notebook | What it teaches |
| --- | --- |
| [`00_quickstart.ipynb`](00_quickstart.ipynb) | Actual cohort trials, source QC, waveforms, and freshly computed single-contact reproducibility/energy evidence. |
| [`01_waveform_visualizations.ipynb`](01_waveform_visualizations.ipynb) | Means/SEM, trials, heatmaps, comparisons, ranked and grouped grids. |
| [`02_detection_qc_and_crp.ipynb`](02_detection_qc_and_crp.ipynb) | Artifact reasons, detector matrices, response metrics, and CRP panels. |
| [`03_spectral_visualizations.ipynb`](03_spectral_visualizations.ipynb) | PSD, ERSP/TFR, ITPC, PLV, PAC, and phase coupling. |
| [`04_network_visualizations.ipynb`](04_network_visualizations.ipynb) | Edge tables, adjacency/response matrices, topology, coordinates, and graph metrics. |
| [`05_brain_and_interactive_visualizations.ipynb`](05_brain_and_interactive_visualizations.ipynb) | MNI glass-brain, Plotly, animation, and opt-in surface recipes. |
| [`06_exporting_figures.ipynb`](06_exporting_figures.ipynb) | High quality layout checks and reproducible PNG/PDF export. |
| [`07_public_ds003708_recipe.ipynb`](07_public_ds003708_recipe.ipynb) | Run a public recording subset and view input provenance, detector results, and waveform figures. |
| [`08_n1_development.ipynb`](08_n1_development.ipynb) | Verify an actual public record and held-out N1 model, reproduce its score, round-trip model save/load, and inspect signed logit contributions. |

## Running them

From the repository root:

```bash
python -m pip install jupyterlab
python -m jupyter lab notebooks/examples/00_quickstart.ipynb
```

Keep the complete folder together: the notebooks import its `_cohort.py`,
`_synthetic.py` and `_n1_inputs.py` helpers. The stored outputs are bound to
the notebook source and their recorded execution versions, so the notebooks
still use some older method names (for example `detect_erp_all`), which remain
available. The [visualization gallery](../../docs/VISUALIZATION_GALLERY.md)
maps every plotting function to the notebook that demonstrates it.
