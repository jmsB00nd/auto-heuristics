from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import OrchestratorConfig
from .schemas import BackendInfo, BenchmarksInfo, LLMInfo, RunMetadata

FRAMEWORK_VERSION = "0.1.0"


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
    except OSError:
        return None


def _sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _git(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _config_dump(config: OrchestratorConfig) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _feature_flags(config: OrchestratorConfig) -> Dict[str, bool]:
    return {
        "literature_review": bool(getattr(config, "run_stage1_literature_review", False)),
        "evolution": bool(getattr(config, "run_evolution", True)),
        "conversation_mode": bool(getattr(config, "use_conversation_mode", False)),
    }


def _hash_prompts(prompts_dir: str, problem: str) -> Dict[str, str]:
    base = Path(prompts_dir) / problem
    out: Dict[str, str] = {}
    if not base.is_dir():
        return out
    for p in sorted(base.glob("*.txt")):
        h = _sha256_file(p)
        if h:
            out[p.name] = h
    return out


def _backend_info(config: OrchestratorConfig) -> Optional[BackendInfo]:
    name = getattr(config, "backend", None)
    if not name:
        return None
    try:
        from qpu.src.load_backend import BACKEND_FILE_MAP, TOPLOGIES_DIR

        file_path = Path(TOPLOGIES_DIR) / BACKEND_FILE_MAP.get(name, "")
        topology_hash = _sha256_file(file_path) if file_path.exists() else None
        n_qubits: Optional[int] = None
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "num_qubits" in data:
                    n_qubits = int(data["num_qubits"])
                elif isinstance(data, dict) and "coupling_map" in data:
                    # Derive from max node id in the coupling map.
                    cm = data["coupling_map"]
                    if cm:
                        n_qubits = int(max(max(edge) for edge in cm)) + 1
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        return BackendInfo(name=name, n_qubits=n_qubits, topology_hash=topology_hash)
    except Exception:
        return BackendInfo(name=name)


def _benchmarks_info(config: OrchestratorConfig) -> Optional[BenchmarksInfo]:
    directory = getattr(config, "benchmark_dir", None)
    if not directory:
        return None
    base = Path(directory)
    circuits = sorted(p.name for p in base.glob("*.json")) if base.is_dir() else []
    set_hash: Optional[str] = None
    if circuits:
        # Hash the sorted list of (name, sha256) pairs so the set_hash changes
        # if either membership or file contents change.
        parts = []
        for name in circuits:
            ph = _sha256_file(base / name)
            parts.append(f"{name}\t{ph or ''}")
        set_hash = _sha256_bytes("\n".join(parts).encode())
    return BenchmarksInfo(dir=str(base), circuits=circuits, set_hash=set_hash)


def _extract_model(cli_command: str) -> Optional[str]:
    tokens = cli_command.split()
    for i, t in enumerate(tokens):
        if t == "--model" and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith("--model="):
            return t.split("=", 1)[1]
    return None


def build_run_metadata(
    config: OrchestratorConfig,
    run_id: str,
    experiment_name: str,
    seeds: Dict[str, Optional[int]],
) -> RunMetadata:
    cli = getattr(config, "cli_command", "")
    return RunMetadata(
        run_id=run_id,
        experiment_name=experiment_name,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        git_commit=_git(["rev-parse", "--short", "HEAD"]),
        git_dirty=bool(_git(["status", "--porcelain"])),
        framework_version=FRAMEWORK_VERSION,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        llm=LLMInfo(cli_command=cli, model_reported=_extract_model(cli)),
        seeds=seeds,
        config=_config_dump(config),
        feature_flags=_feature_flags(config),
        backend=_backend_info(config),
        benchmarks=_benchmarks_info(config),
        prompts=_hash_prompts(getattr(config, "prompts_dir", ""), getattr(config, "problem", "")),
    )


def write_run_metadata(metadata: RunMetadata, log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "run_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))
    return path
