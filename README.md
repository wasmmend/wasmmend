# WasmMend

**Reasoning from Traces: Divergence-Guided Agentic Repair of WebAssembly Discrepancies**

WasmMend is the first system to automatically repair *Native–Wasm functional
discrepancies* — silent runtime divergences that occur when the same C/C++
program produces different observable behavior under a native build (e.g.
`gcc`) versus a WebAssembly build (e.g. `emcc` + Node.js). Such discrepancies
are typically caused by library implementation differences or compiler bugs at
the platform level, so the root cause is hidden beneath the source code and
general-purpose LLM repair agents struggle to localize and fix them.

WasmMend reframes this undirected exploration problem as a focused reasoning
task in two stages:

1. **Differential Trace Analysis** symbolically aligns native and Wasm
   executions and pinpoints the function where their behavior first diverges.
2. **Divergence-Guided Agentic Repair** then hands that diagnostic evidence to
   an LLM-driven analyze/patch loop that synthesizes and validates a fix.


## How it works

```
        +---------+        +-------------------------+        +-------------------------+
Inputs  |  C/C++  |        | Differential Trace      |        | Divergence-Guided       |
        |  repo   |  --->  |   Analysis              |  --->  |   Agentic Repair        |
        | toolch. |        |  (diff_trace_analysis)  |        |  (divergence_guided_    |
        | failure |        |                         |        |   repair)               |
        +---------+        +-------------------------+        +-------------------------+
                                       |                                   |
                                       v                                   v
                              trace_analysis.json                    Final patch
```

### Stage 1 — Differential Trace Analysis

Given a repository with an observed discrepancy, WasmMend:

- Builds a **call graph** from the failing test and identifies the functions
  reachable from the divergence.
- **Serializes types** (arguments, returns, locals) topologically using
  libclang AST traversal, generating type-aware printers for nested
  user-defined structs.
- **Instruments functions** to emit entry/exit events tagged with a unique
  function ID and the values of all input/output state.
- Runs both the native and Wasm builds and collects the two event streams
  (`NativeEvents`, `WasmEvents`).
- Performs a **cross-execution match** (Algorithm 1 in the paper): each
  native event is matched against the Wasm event pool; matched entries are
  pushed onto a shared stack and matched exits pop it. The first unmatched
  event identifies the function in which native and Wasm executions first
  diverge. A `Suspects` list (witnessed via the parent's entry/exit events)
  handles incomplete trace logs caused by *concessions* — places where
  instrumentation had to be omitted because a type could not be safely
  printed.
- During instrumentation, an **LLM Assistance** loop repairs printer code
  that fails to compile (e.g. `std::locale` and friends) and may grant
  *concessions* when a type is intractable.

The output is `trace_analysis.json` (plus `function_types.json` and a merged
`trace_analysis_combined.json`) — the diagnostic context passed to Stage 2.

### Stage 2 — Divergence-Guided Agentic Repair

Grounded in the trace analysis report, two cooperating agents iterate over
the repository:

- **ANALYZE agent.** Inspects source code across the project, can compile and
  re-execute both Wasm and native builds, and may use the LLM-driven
  instrumentation tool to gather additional trace evidence. Produces a
  concrete root-cause hypothesis and patch plan.
- **PATCH agent.** Generates patch code, applies it to the repository, and
  validates that (a) the patch resolves the discrepancy against the native
  baseline and (b) the patch does not pass the test by altering the program's
  original semantics. A patch is accepted only when both validations pass;
  otherwise the agent revises the plan or hands control back to ANALYZE.

The loop terminates when the divergence is eliminated or when the agent
budget (default 50 iterations) is exhausted.


## Repository layout

```
WasmMend/
├── divergence_guided_repair.py     # Stage 2 entry point (agentic repair)
├── diff_trace_analysis.py          # Stage 1 entry point (trace analysis pipeline)
│
├── analysis/                       # Call graph, AST, type extraction, cross-execution match
│   ├── AST_builder.py
│   ├── CallGraphBuilder.py
│   ├── DynamicTraceAnalysis.py     # Algorithm 1: cross-execution match
│   ├── TypeParser.py
│   ├── Data_analyzer.py
│   └── LogFilter.py
│
├── instrumentation/                # Type-aware printers and function instrumentors
│   ├── FunctionInstrumentor.py
│   ├── InstrumentationCoordinator.py
│   ├── CPrintInstrumentor.py       # C  (fprintf / <stdio.h>)
│   ├── OstreamInstrumentor.py      # C++ (std::cout / <iostream>)
│   ├── c_instrumentation_prompts.py
│   └── expand_oneliners.py
│
├── llm/                            # LLM client wrappers and instrumentation-repair loop
│   ├── LLMAgent.py                 # OpenAI / DeepSeek / Gemini / Qwen backends
│   └── LLMInstrumentor.py          # LLM Assistance for tricky printers
│
├── repair/                         # Agent loop, tools, state machine
│   ├── Repairer.py                 # ANALYZE + PATCH iteration driver
│   ├── Toolkit.py                  # Tool implementations (read, edit, compile, run, ...)
│   ├── States.py                   # State transitions and prompts
│   ├── ResponseParser.py
│   ├── WorkflowConfig.py           # Ablation knobs (provide_*, action sets)
│   ├── Models.py
│   └── config.py
│
└── preprocess/                     # Shared preprocessing library used by Stage 1
    └── Preprocess.py
```


## Environment setup

WasmMend orchestrates two compilation toolchains, a Python pipeline, and one
or more LLM backends. This section walks through provisioning each piece in
order.

### 1. System toolchains

The host machine must be able to build a target C/C++ project both natively
and to WebAssembly:

- **Native compiler** — `gcc` (the paper evaluates on Linux x86).
- **Emscripten** — provides `emcc`/`em++` to build the Wasm artifact.
  Install via `emsdk` and source the generated `emsdk_env.sh` so `emcc`
  is on the `PATH`.
- **Node.js** — required to execute the Emscripten-generated JavaScript
  glue code that loads the Wasm binary.
- **libclang** — the native shared library that backs the Python `clang`
  bindings used for AST traversal. On Debian/Ubuntu:

  ```bash
  sudo apt-get install libclang-15-dev
  ```

  On macOS the library shipped with Xcode's command-line tools or a
  Homebrew `llvm` install will work; on other systems install any
  reasonably recent libclang and ensure it is discoverable by the
  Python bindings (e.g. via `LD_LIBRARY_PATH` or
  `clang.cindex.Config.set_library_file(...)`).

### 2. Python environment

Python 3.10 or newer is required. We recommend an isolated environment.

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Using `conda`:

```bash
conda create -n wasmmend python=3.10
conda activate wasmmend
pip install -r requirements.txt
```

The `requirements.txt` shipped with this repository pins the libraries that
were used during the paper's evaluation:

- `clang` (libclang Python bindings — must be paired with a system libclang
  install, see above)
- `google-genai` (modern Gemini SDK, used by the `GeminiAgent` backend)
- `openai` (used by the OpenAI, DeepSeek, and Qwen agents — DeepSeek and
  Qwen are reached via OpenAI-compatible endpoints)
- `tqdm` (progress bars for the parallel instrumentation phase)
- `openpyxl` (optional, only needed when `--collect` is passed to the
  repair driver)

### 3. API keys

LLM credentials are read exclusively from environment variables. Each
variable accepts a single key or a comma-separated list of keys; the agent
rotates to the next key in the list on a 4xx / rate-limit error. Only the
backends you actually plan to use need to be set.

```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-...,sk-..."
export GEMINI_API_KEY="AIza...,AIza..."
export DASHSCOPE_API_KEY="sk-..."   # for Qwen via DashScope
```

### 4. Per-project build scripts

WasmMend expects each target repository to expose two scripts at its root so
the pipeline can produce and run both builds:

- `compile.sh` — builds the project natively (into `build_native/`) and to
  WebAssembly (into `build_wasm/`).
- `run.sh` — executes the failing test under both builds and writes their
  outputs to predictable locations.

A `compile_commands.json` (typically emitted by CMake with
`-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`) is also required as the second
argument to Stage 1.

### 5. Optional environment variables

- `WASM_PREPROCESS_TMP` — base directory used for per-worker repository
  copies during parallel instrumentation. Defaults to the system temp dir
  (use a fast local disk if you are running with many workers).

### 6. Quick sanity check

Once everything is installed, the repository should import cleanly:

```bash
python -c "import diff_trace_analysis, divergence_guided_repair; print('OK')"
```

If this prints `OK`, the Python environment is wired up correctly and you
are ready to run the pipeline (see [Usage](#usage)).


## Usage

The typical workflow runs Stage 1 once per discrepancy to produce
`trace_analysis.json`, then runs Stage 2 to obtain a patch.

### Stage 1 — produce the trace analysis report

```bash
python diff_trace_analysis.py <project_path> <project_path>/build_native/compile_commands.json
```

Useful flags:

| Flag                              | Purpose                                                  |
| --------------------------------- | -------------------------------------------------------- |
| `--backend {gemini,deepseek}`     | LLM backend used for instrumentation repair (default `gemini`). |
| `-j N`, `--num-workers N`         | Parallel worker count for AST instrumentation (default `8`). |
| `--fixed-time`                    | Prepend a fixed-time wrapper to compile/run commands.    |

Outputs are written next to the project: `trace_analysis.json`,
`function_types.json`, and `trace_analysis_combined.json`.

### Stage 2 — divergence-guided repair

```bash
python divergence_guided_repair.py \
    --trace_analysis <project_path>/trace_analysis.json \
    --test_case_name <FailingTestName> \
    --model gemini-3-flash-preview \
    --max_iterations 50
```

Common flags:

| Flag                       | Purpose                                                                     |
| -------------------------- | --------------------------------------------------------------------------- |
| `--trace_analysis PATH`    | Path to the report produced by Stage 1.                                     |
| `--config PATH`            | Alternative: load all repair inputs from a single JSON config.              |
| `--model NAME`             | LLM identifier (default `gemini-3-flash-preview`).                          |
| `--max_iterations N`       | Cap on ANALYZE+PATCH iterations (default `50`).                             |
| `--workflow_config PATH`   | Workflow JSON controlling the available actions and provided information (ablation studies). |
| `--no_trace_analysis`      | Ablation: hide all trace-analysis context from the agent.                   |
| `--no_candidates`          | Ablation: hide candidate functions.                                         |
| `--no_instrumentation`     | Ablation: remove repair-time instrumentation tools.                         |
| `--no_type_deps`           | Ablation: remove type-dependency tools.                                     |
| `--restore`                | Restore source files from the previous run's backup before starting.        |
| `--always_restore`         | After the run, restore every modified source file to its original content.  |
| `--clean_build_dir`        | Force a full clean rebuild of `build_native/` and `build_wasm/`.            |
| `--collect`                | Append a row of summary metrics to `results/{proj}/{test}/results.xlsx`.    |

Repair output is written to:

```
results/<project>/<test_case>/run_<timestamp>_<model>_<ablation_tag>/
    config.json          # snapshot of ablation flags / model / budget
    repair_result.json   # status, fix, history, token accounting
    llm_calls.jsonl      # full LLM transcript
    latest_restore/      # backup of every file the toolkit modified
```

Calling `divergence_guided_repair.py` directly (without `--trace_analysis`) is
also supported via `--root_cause_file / --root_cause_start / --root_cause_end`
or `--config <json>`; this is useful for re-running the agent on a manually
specified function without re-running Stage 1.



