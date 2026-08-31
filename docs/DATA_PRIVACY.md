# Data privacy and safe sharing

ERPy's analysis, caching, export, and project-creation functions run locally
and do not upload data. The repository contains synthetic teaching data and
references to public OpenNeuro data only. Clinical source recordings and
cohort-specific manifests are intentionally excluded.

The optional public-data validation command makes read-only downloads of
public OpenNeuro files or byte ranges. Some optional brain visualizations can
also download public template surfaces through Nilearn. These operations do
not transmit a user's recordings, result tables, metadata, or identifiers.

## Release-record labels are local

A label such as `ERPY-RR-<UUID>` is an optional, opaque audit label used by an
author or institution to refer to a particular permitted local derivative. It
is not an upload destination, account, cloud record, package-installation ID,
or participant ID. ERPy 1.0.0 does not generate, register, or attach these
labels automatically, and `AnalysisResult.save(...)` does not make a network
request. It writes named CSV files and a JSON metadata record to the requested
local folder.

The default saved filename prefix is built from the analysis patient, session,
and stimulation-pair values, and those values can also appear inside the saved
tables and metadata. Assigning an opaque release-record label or renaming a
folder does not deidentify those outputs. Create and inspect a separately
approved derivative before anything is shared.

If a study uses this naming convention, create a random UUIDv4 locally:

```python
from uuid import uuid4

release_record_id = f"ERPY-RR-{str(uuid4()).upper()}"
```

The UUID is random; it must not encode or hash a name, medical-record number,
date, source filename, electrode label, or waveform. Store the private mapping
between this label and the exact approved file set in the governed research
environment. Merely publishing the opaque label does not publish its mapped
files.

Do not confuse an author-assigned release-record label with
`ERPy.CHECKPOINT_COMPATIBILITY_ID`, which is a fixed software-behavior
checkpoint shared by every equivalent ERPy installation. Also do not confuse
it with the short deterministic preprocessing and epoch hashes in local cache
filenames. Neither kind of software label causes an upload.

Before sharing a project, inspect all filenames, tables, notebook outputs,
figures, logs, caches, and document properties. Remove or replace:

- names, medical-record numbers, accession numbers, and free-text notes;
- dates, timestamps, ages, pain states, and rare clinical descriptors;
- institution-specific server paths, usernames, tokens, and credentials;
- mappings between study codes and source-system identifiers;
- raw EDF/BDF/BrainVision/NWB recordings unless their release is authorized;
- notebook outputs that reveal any of the above.

Use generic example labels such as `participant`, `session`, `stim_pair`, and
`recording_contact`. Deidentification is a governance decision, not merely a
string-replacement step: retain only the minimum derived information allowed by
the applicable protocol and data-use agreement.

HDF5 files and serialized Python objects should be treated as data, not benign
documents. Load them only from trusted sources. ERPy rejects legacy pickled
epoch and processed-cache metadata by default because pickle can execute code
while loading.

The public release should pass a final scan for secrets, absolute paths,
protected identifiers, notebook outputs, and document/image metadata. Keep a
separate private provenance record for reproducibility under the governing IRB
or data-use controls.
