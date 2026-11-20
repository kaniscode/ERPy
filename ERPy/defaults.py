"""Shared default settings for the standard ERPy workflow.

These values are the defaults of :meth:`ERPy.Patient.epoch`,
:meth:`ERPy.Patient.analyze`, :class:`ERPy.Pipeline` and cache warm-up, kept in
one place so the entry points cannot drift apart.
"""

from __future__ import annotations

#: Name of the preprocessing preset used when none is requested.
DEFAULT_PIPELINE = "blank_filt"

#: Epoch start and end relative to stimulation onset, in seconds.
DEFAULT_TMIN = -0.5
DEFAULT_TMAX = 1.0

#: Pre-stimulation window subtracted from every trial, in seconds.
DEFAULT_BASELINE = (-0.5, -0.03)

#: Response window scored by the detectors, in seconds.
DEFAULT_RESPONSE_WINDOW = (0.01, 0.3)
