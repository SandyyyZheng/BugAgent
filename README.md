# BugAgent

A LangGraph-based multimodal LLM pipeline for automated video game glitch detection.

---

## Architecture

```
Video
  │
  ▼
┌─────────────┐
│  Preprocess │  Extract frames · Segment into stitched windows
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Scanner   │  Initial screening of every window
│             │  → has_glitch · category · confidence
│             │  → game_context  ◀ NEW: RAG knowledge base
└──────┬──────┘
       │ has glitch?
       ├──── NO ──────────────────────────────────────────┐
       │                                                   │
       ▼                                                   │
┌─────────────────────────────────────────────────────┐   │
│  Analyzer  (Memory · Planner · Executor · Reflector) │   │
│                                                      │   │
│  Memory holds game_context from Scanner as a RAG     │   │
│  knowledge base for all sub-agents.                  │   │
│                                                      │   │
│  Reflector = adversarial debate:                     │   │
│    Advocate  →  argues this IS a glitch              │   │
│    Skeptic   →  argues this is normal behavior       │   │
│    Judge     →  makes final ruling                   │   │
└──────┬──────────────────────────────────────────────┘   │
       │                                                   │
       ▼                                                   │
┌─────────────┐                                            │
│   Grounder  │  Cluster adjacent glitches · temporal      │
│             │  boundary detection (bidirectional)         │
└──────┬──────┘                                            │
       │                                           ◀───────┘
       ▼
┌─────────────┐
│  Summarizer │  Convert to final report with time-based nodes
└─────────────┘
       │
       ▼
   JSON Report
```

### Key Design Choices

| Component | Original | BugAgent |
|-----------|----------|----------|
| `categorizer` | GlitchCategorizer | **GlitchScanner** — adds `game_context` field |
| `plan_adjust` | PlanAdjustAgent | **GlitchAnalyzer** — reads `game_context` from Memory |
| Orchestration | Sequential scripts | **LangGraph StateGraph** |
| Game context | None | Scanner produces it; Analyzer uses it as a RAG knowledge base |

---

## Project Structure

```
BugAgent/
├── run.py                    # CLI entry point
├── graph.py                  # LangGraph workflow (nodes + edges)
├── state.py                  # BugAgentState TypedDict
├── config.py                 # Hierarchical configuration dataclasses
├── requirements.txt
│
├── llm/
│   └── client.py             # Unified LLM client (OpenAI / Anthropic / vLLM)
│
├── preprocess/
│   └── video_preprocessor.py # Frame extraction + window stitching
│
├── scanner/                  # ── Stage 2 ──
│   ├── scanner.py            # GlitchScanner class
│   └── prompt.txt            # Scanner system prompt (adds game_context)
│
├── analyzer/                 # ── Stage 3 ──
│   ├── agent.py              # GlitchAnalyzer (Memory-Planner-Executor-Reflector)
│   ├── memory.py             # Memory with game_context support
│   ├── tools.py              # VQATool (active) + placeholders
│   └── prompt.txt            # Planner / Advocate / Skeptic / Judge prompts
│
├── grounder/                 # ── Stage 4 ──
│   ├── grounder.py           # TemporalGrounder
│   └── prompt.txt
│
└── summarizer/               # ── Stage 5 ──
    ├── summarizer.py         # Summarizer
    └── prompt.txt
```

---

## game_context: RAG Knowledge Base

The **Scanner** now produces a `game_context` field in every window's output:

```json
{
  "has_glitch": true,
  "category": "Physics",
  "visual_cues": "Red car floating above the road",
  "confidence": 0.82,
  "game_context": "Open-world racing game. Urban road environment with multiple vehicles and city buildings. Physics-based driving mechanics. Player vehicle is a red sports car."
}
```

This string is **aggregated across all windows** and stored in the LangGraph state.
The **Analyzer** injects it into every LLM prompt as a knowledge base section:

```
## Game Context (Knowledge Base)
Open-world racing game. Urban road environment with ...

## Initial Hypothesis (from Scanner)
- Category: Physics
...
```

This gives the Advocate, Skeptic, and Judge agents stable knowledge about the game's
intended physics, art style, and mechanics — helping them distinguish genuine bugs from
intentional design without hallucinating game details.

---

## Tools

Only **VQA** is active. Placeholder classes exist for future tools:

| Tool | Status | Description |
|------|--------|-------------|
| `vqa` | ✅ Active | Visual QA via MLLM — asks questions about stitched window images |
| `object_tracking` | 🔲 Placeholder | Frame-by-frame tracking via SAM3 |
| `math_calculation` | 🔲 Placeholder | Physics/trajectory analysis from tracking data |

To activate a placeholder tool, implement its `execute()` method in
`analyzer/tools.py` and uncomment the registration line in `analyzer/agent.py`.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with a local vLLM server

```bash
# Start vLLM first:
# vllm serve Qwen/Qwen2.5-VL-7B-Instruct --port 8000

python run.py --video data/videos/30p1kv.mp4
```

### 3. Run with OpenAI

```bash
python run.py \
    --video data/videos/haj831.mp4 \
    --api-key $OPENAI_API_KEY \
    --api-base https://api.openai.com/v1 \
    --model gpt-4o \
    --game-name "GTA V"
```

### 4. Programmatic use

```python
from config import BugAgentConfig
from graph import run_pipeline

cfg = BugAgentConfig()
cfg.llm.api_key = "sk-..."
cfg.llm.api_base = "https://api.openai.com/v1"
cfg.llm.model = "gpt-4o"

final_state = run_pipeline(
    video_path="data/videos/haj831.mp4",
    config_dict=cfg.to_dict(),
    game_name="GTA V",
)

report = final_state["final_report"]
print(report["bugs"])
```

---

## Output

The final report is saved to `{output_dir}/results/{video_name}_report.json`:

```json
{
  "video_name": "haj831",
  "game_name": "GTA V",
  "no_bugs": false,
  "bugs": [
    "A red sports car is floating approximately 2 meters above the road surface near the highway overpass, with no visible support or propulsion."
  ],
  "time_nodes": [
    [[12, 15], [23, 24]]
  ]
}
```

`time_nodes[i]` is a list of `[start_sec, end_sec]` intervals for bug `i`.

---

## Configuration Reference

```python
from config import BugAgentConfig

cfg = BugAgentConfig(
    output_dir="data",
    verbose=True,
    save_intermediate=True,   # saves scan/analysis/grounded JSONs to data/intermediate/
)

cfg.llm.api_key    = "EMPTY"
cfg.llm.api_base   = "http://localhost:8000/v1"
cfg.llm.model      = "Qwen/Qwen2.5-VL-7B-Instruct"
cfg.llm.temperature = 0.3
cfg.llm.max_tokens  = 1024
cfg.llm.timeout     = 120

cfg.preprocess.target_fps    = 4.0   # frames/sec to extract
cfg.preprocess.window_size   = 8     # frames per stitched window
cfg.preprocess.window_overlap = 0

cfg.scanner.temperature = 0.3
cfg.scanner.max_tokens  = 512

cfg.analyzer.max_iterations      = 5     # max Planner→Executor→Reflector cycles
cfg.analyzer.confidence_threshold = 0.70 # stop when Judge reaches this confidence

cfg.grounder.frames_per_window = 8  # must match preprocess.window_size

cfg.summarizer.fps = 4.0   # must match preprocess.target_fps
```

---

## LangGraph Flow

```
preprocess_node
      │
scanner_node
      │
      ├── (has glitches) ──► analyzer_node ──► grounder_node ──► summarizer_node ──► END
      │
      └── (no glitches) ──────────────────────────────────────► summarizer_node ──► END
```

The conditional edge `route_after_scanner` skips the analyzer and grounder entirely
when the scanner finds no potential glitches, producing a clean "no bugs" report
without wasting API calls.
