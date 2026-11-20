"""Single source of the package version and numerical-compatibility checkpoint."""

__version__ = "1.1.0rc1"

# This developer-maintained checkpoint is shared by every installation with
# the same numerical and QC semantics. It is not a participant identifier, a
# local release-record label, or a network destination. Bump it only when
# numerical analysis results or QC semantics change.
CHECKPOINT_COMPATIBILITY_ID = "415a636abd5b9a28"
