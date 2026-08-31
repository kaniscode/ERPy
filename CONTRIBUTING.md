# Contributing

Contributions are welcome when they include a focused explanation, tests, and
documentation. Please work from a branch and run:

```text
python -m pip install -e ".[dev,edf,nwb,viz]"
python -m pytest
```

Scientific changes should state the mathematical definition, defaults,
reference implementation or publication, expected failure modes, and how the
change was validated. Numerical or QC-semantic changes must update the release
notes and compatibility identifier.

Never commit clinical recordings, participant mappings, source-system IDs,
dates, credentials, server paths, or notebook outputs derived from protected
data. Use deterministic synthetic fixtures or clearly licensed public data.

Code style should favor explicit parameters, stable tabular outputs, and useful
error messages. New public callables require complete docstrings and an example
or test. High quality visualizations should be checked at their intended
display size, with readable labels, an accessible palette, and sufficient
padding.
