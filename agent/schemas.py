from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LLMInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    cli_command: str
    model_reported: Optional[str] = None


class BackendInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    n_qubits: Optional[int] = None
    topology_hash: Optional[str] = None


class BenchmarksInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    dir: str
    circuits: List[str] = Field(default_factory=list)
    set_hash: Optional[str] = None


class RunMetadata(BaseModel):
    """Written once per run to ``outputs/logs/{run_id}/run_metadata.json``."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    experiment_name: str
    timestamp_utc: str
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    framework_version: Optional[str] = None
    python_version: str
    platform: str
    llm: LLMInfo
    seeds: Dict[str, Optional[int]] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    feature_flags: Dict[str, bool] = Field(default_factory=dict)
    backend: Optional[BackendInfo] = None
    benchmarks: Optional[BenchmarksInfo] = None
    prompts: Dict[str, str] = Field(default_factory=dict, description="prompt_filename -> sha256 hash")


class RankingEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    rank: int
    idea_name: str
    mean_swaps: float
    mean_depth: float
    error: Optional[str] = None
    code_path: Optional[str] = None


class BestEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    idea_name: str
    mean_swaps: float
    code_path: Optional[str] = None


class RunSummary(BaseModel):
    """Written at end of pipeline to ``outputs/logs/{run_id}/summary.json``."""

    model_config = ConfigDict(extra="allow")

    run_id: str
    status: str
    stage_timings_seconds: Dict[str, float] = Field(default_factory=dict)
    token_totals: Dict[str, float] = Field(default_factory=dict)
    counts: Dict[str, int] = Field(default_factory=dict)
    ranking: List[RankingEntry] = Field(default_factory=list)
    best: Optional[BestEntry] = None


class EvaluationResult(BaseModel):
    """Formalizes ``evaluation_results.json`` emitted per idea / refinement round."""

    model_config = ConfigDict(extra="allow")

    mean_swaps: float
    mean_depth: float = 0.0
    error: Optional[str] = None
    failures: List[Dict[str, Any]] = Field(default_factory=list)


class HistoryEntry(BaseModel):
    """Formalizes entries appended to ``outputs/experiment_history.json`` ``history``."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    code: str
    mean_swaps: float
    mean_depth: float = 0.0
    error: Optional[str] = None
    run_id: Optional[str] = None
