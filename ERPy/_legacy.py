"""Compatibility shims for interfaces that predate ERPy 1.0."""

from __future__ import annotations


class LegacyScriptHooks:
    """Inert job-script hooks kept so pre-1.0 notebooks continue to run.

    Early releases dispatched work through external Python and shell scripts
    selected with these class attributes. Processing now always runs through
    the built-in code path, so the hooks only reset the attributes.
    """

    PYTHON_SCRIPT: str | None = None
    BASH_SCRIPT: str | None = None

    @classmethod
    def install_standard_scripts(cls) -> None:
        """Do nothing; ERPy's built-in execution path needs no installation."""

        return None

    @classmethod
    def use_standard_scripts(cls, bash_type: str = "simple") -> None:
        """Select ERPy's built-in execution path (the only one available)."""

        cls.PYTHON_SCRIPT = None
        cls.BASH_SCRIPT = None


class legacy_alias:
    """Class attribute that forwards a retired attribute name to its replacement.

    ``DataLoader`` exposed its settings under upper-case names before 1.1.
    Declaring ``RAWDATA_PATH = legacy_alias("rawdata_path")`` keeps reading and
    assigning the old name working while the snake_case attribute holds the
    value.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        self.__doc__ = f"Retained alias of ``{target}``."

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return getattr(instance, self.target)

    def __set__(self, instance, value) -> None:
        setattr(instance, self.target, value)
