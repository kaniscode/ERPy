# Complete response-window support

The manuscript candidate rejects an epoch that does not cover the effective declared response window within half a sample (plus a 1e-12-second floating-point tolerance). Such an input receives `insufficient_data`, unavailable p-values and zero randomizations. The result retains the requested window. A deliberately declared shorter window remains usable; the baseline still uses the latest available sample-matched segment within its declared candidate interval.

Before this correction, a 15–300-ms request on a generated epoch ending at 100 ms produced a positive result for the silently shortened 15–100-ms interval. The regression now returns unavailable. This changes input availability, not either numerical component test or the intersection–union formula. Source hashes and before/after results are recorded in `response_window_support_verification.json`.

Four added test cases cover the truncated positive response, a deliberately shorter complete test, complete support and the half-sample boundary. The guard correction passed the then-current 296-test suite with two expected SciPy warnings in existing identical-trial multiscale fixtures. Run `python -m pytest -o addopts= -q` to repeat the suite.

Saved extraction metadata for all 596 available public epoch files begins before 10 ms and ends after 300 ms. Thus both frozen external configurations have full support and never take the new rejection branch. The receipt records each metadata hash and realized interval. Waveform contents were not copied or rerun for this check; source extraction and historical predictions retain their original provenance. The code diff leaves the tested numerical path unchanged for supported inputs.
