from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class JobRunner:
    """Small scheduler adapter used by ERPy queue_job methods.

    The local backend is the reference implementation. ``slurm`` and
    ``submit_job`` are thin adapters that shell out only when the scheduler
    command exists, otherwise they fall back to local execution.
    """

    backend: str | None = None
    queue: str | None = None
    numcpu: int | None = None
    memory: int | None = None
    job_name: str = "erpy_job"
    out_file: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def detect_backend(self) -> str:
        if self.backend:
            return self.backend
        if shutil.which("submit_job"):
            return "submit_job"
        if shutil.which("sbatch"):
            return "slurm"
        return "local"

    @staticmethod
    def env_from_kwargs(**kwargs: Any) -> dict[str, str]:
        env = {}
        for key, value in kwargs.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                import json

                value = json.dumps(value)
            env[str(key).upper()] = str(value)
        return env

    def run(
        self,
        python_script: str | Path,
        env: dict[str, str] | None = None,
        python: str | Path | None = None,
    ) -> tuple["JobRunner", str, str]:
        backend = self.detect_backend()
        python_script = str(python_script)
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        if python is None:
            python = sys.executable

        if backend == "local":
            proc = subprocess.run(
                [str(python), python_script],
                capture_output=True,
                text=True,
                env=merged_env,
                check=False,
            )
            return self, proc.stdout, proc.stderr

        if backend == "slurm" and shutil.which("sbatch"):
            cmd = ["sbatch", "--parsable", "--job-name", self.job_name]
            if self.queue:
                cmd += ["--partition", self.queue]
            if self.numcpu:
                cmd += ["--cpus-per-task", str(self.numcpu)]
            if self.memory:
                cmd += ["--mem", f"{self.memory}G"]
            if self.out_file:
                cmd += ["--output", self.out_file]
            export = ",".join(f"{k}={v}" for k, v in (env or {}).items())
            if export:
                cmd += ["--export", f"ALL,{export}"]
            cmd += ["--wrap", shlex.join([str(python), python_script])]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return self, proc.stdout, proc.stderr

        if backend == "submit_job" and shutil.which("submit_job"):
            cmd = ["submit_job", "-q", self.queue or "normal", "-n", self.job_name]
            if self.numcpu:
                cmd += ["-c", str(self.numcpu)]
            if self.memory:
                cmd += ["-m", str(self.memory)]
            cmd += [str(python), python_script]
            proc = subprocess.run(cmd, capture_output=True, text=True, env=merged_env, check=False)
            return self, proc.stdout, proc.stderr

        fallback = JobRunner(backend="local", job_name=self.job_name)
        return fallback.run(python_script, env=env, python=python)
