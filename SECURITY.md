# Security policy

## Supported release

Security fixes are applied to the latest tagged ERPy release.

## Reporting a vulnerability

Please report vulnerabilities privately to the repository owner through
GitHub's private vulnerability-reporting feature. Do not open a public issue
containing an exploit, credentials, server paths, or protected data.

ERPy reads scientific file formats that may contain embedded metadata. Treat
downloaded HDF5, pickle, notebook, and archive files as untrusted. Legacy
pickled epoch and processed-cache metadata are disabled by default in version
1.0.0.
