"""
BugAgent configuration.

Provides a structured, hierarchical config with sensible defaults.
Pass a BugAgentConfig (or its to_dict() output) when invoking the graph.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LLMConfig:
    api_key: str = "EMPTY"
    api_base: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    temperature: float = 0.5
    max_tokens: int = 1024
    timeout: int = 120
    max_retries: int = 3


@dataclass
class PreprocessConfig:
    target_fps: float = 4.0
    window_size: int = 8
    window_overlap: int = 0


@dataclass
class ScannerConfig:
    temperature: float = 0.5
    max_tokens: int = 512


@dataclass
class AnalyzerConfig:
    max_iterations: int = 5
    confidence_threshold: float = 0.80
    temperature: float = 0.5
    max_tokens: int = 1024
    # GPU(s) for SAM3 tracker — keep separate from the VLM's GPU (typically 0).
    sam3_gpus: List[int] = field(default_factory=lambda: [1])


@dataclass
class GrounderConfig:
    # NOTE: graph.py overrides this with preprocess.window_size at runtime.
    # Keep in sync with PreprocessConfig.window_size.
    frames_per_window: int = 8
    max_retries: int = 3


@dataclass
class SummarizerConfig:
    fps: float = 4.0
    max_tokens: int = 512


@dataclass
class BugAgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    analyzer: AnalyzerConfig = field(default_factory=AnalyzerConfig)
    grounder: GrounderConfig = field(default_factory=GrounderConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    output_dir: str = "data"
    verbose: bool = True
    save_intermediate: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a plain dict for LangGraph state."""
        return {
            "llm": {
                "api_key": self.llm.api_key,
                "api_base": self.llm.api_base,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "max_retries": self.llm.max_retries,
            },
            "preprocess": {
                "target_fps": self.preprocess.target_fps,
                "window_size": self.preprocess.window_size,
                "window_overlap": self.preprocess.window_overlap,
            },
            "scanner": {
                "temperature": self.scanner.temperature,
                "max_tokens": self.scanner.max_tokens,
            },
            "analyzer": {
                "max_iterations": self.analyzer.max_iterations,
                "confidence_threshold": self.analyzer.confidence_threshold,
                "temperature": self.analyzer.temperature,
                "max_tokens": self.analyzer.max_tokens,
                "sam3_gpus": self.analyzer.sam3_gpus,
            },
            "grounder": {
                "frames_per_window": self.grounder.frames_per_window,
                "max_retries": self.grounder.max_retries,
            },
            "summarizer": {
                "fps": self.summarizer.fps,
                "max_tokens": self.summarizer.max_tokens,
            },
            "output_dir": self.output_dir,
            "verbose": self.verbose,
            "save_intermediate": self.save_intermediate,
        }


def default_config() -> BugAgentConfig:
    """Return default config using a local vLLM server."""
    return BugAgentConfig()


def openai_config(api_key: str, model: str = "gpt-4o") -> BugAgentConfig:
    """Return config for OpenAI API."""
    cfg = BugAgentConfig()
    cfg.llm.api_key = api_key
    cfg.llm.api_base = "https://api.openai.com/v1"
    cfg.llm.model = model
    return cfg
