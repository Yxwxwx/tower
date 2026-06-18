# PySCF Agent Redesign Spec

**Date:** 2026-06-18
**Status:** Approved

## 1. Problem

Current `PySCFResult` hardcodes `scf_energy` and `casscf_energy` fields, making it unable to represent arbitrary PySCF calculations (geometry opt, frequency, TD-DFT, MP2, CCSD(T), solvent models, etc.). The agent needs flexible, LLM-driven generation and parsing that adapts to the user's task description.

## 2. Design Principles

- **Maximal detail**: Every prompt instructs the LLM to be exhaustive. Extract all available information from both the task description (for generation) and the log output (for parsing).
- **English prompts**: All LLM prompts are in English.
- **Prompt-as-file**: Prompts live as standalone Markdown files under `prompts/pyscf/`, loaded at runtime — not inlined in Python.
- **No RAG for now**: Domain knowledge comes from LLM weights + prompt engineering.

## 3. Contract Changes

### 3.1 PySCFParams (`contracts/src/contracts/pyscf_task.py`)

```python
class PySCFParams(BaseModel):
    task_description: str = ""     # full natural language description
    fchk_artifact_id: str = ""     # upstream Gaussian checkpoint
    charge: int = 0
    spin: int = 0
    additional_info: dict = Field(default_factory=dict)
```

Supervisor passes `task_description` verbatim from the user. The LLM extracts method, basis, functional, active space, etc. from the text.

### 3.2 PySCFResult (`contracts/src/contracts/pyscf_task.py`)

```python
class PySCFResult(BaseModel):
    mol: dict = Field(default_factory=dict)
    # {"atom": [["N",0,0,0],["N",0,0,1.098]], "basis": "cc-pVDZ", "charge": 0, "spin": 0}
    energy: dict = Field(default_factory=dict)
    # {"scf": -109.42, "mp2_total": -109.74, "tddft_root1": 0.15, ...}
    converged: dict = Field(default_factory=dict)
    # {"scf": True, "casscf": True} — empty dict for methods without convergence concept
    extra: dict = Field(default_factory=dict)
    # Anything else: orbital energies, dipole, frequencies, wall_time, errors, ...
```

## 4. Agent Topology

Two paths (simplified from three):

```
pre:  generate_input → generate_slurm → pre_done
post: read_output → parse_output → [branch]
        ├─ success → register_artifacts (DONE)
        ├─ failure + retries < 5 → fix_input → pre_done
        └─ failure + retries >= 5 → NEEDS_HUMAN
```

### 4.1 New Node: `fix_input`

Analyzes error log + previous attempts → produces corrected script. Retry counter in agent state, max 5.

## 5. Prompt Files

Located at `prompts/pyscf/`:

| File | Purpose |
|------|---------|
| `pyscf_generate.md` | Generate complete PySCF script from task description |
| `pyscf_parse.md` | Parse computation log exhaustively |
| `pyscf_fix.md` | Debug and correct failing scripts |

### 5.1 Key Prompt Constraints

**Generate:**
- Extract geometry from LLM knowledge (experimental bond lengths)
- Extract method/basis/functional EXACTLY from task description
- For CASSCF: use AVAZ for orbital selection, max 15 orbitals
- Script prints JSON Lines output block (`---TOWER_OUTPUT_START---` / `---TOWER_OUTPUT_END---`)
- Output fields cover: SCF, DFT, CASSCF, MP2, CCSD(T), TD-DFT, geom opt, frequencies, properties

**Parse:**
- Extract ALL numerical values at original precision
- Energy dict keys describe the source method
- Never round or fabricate
- Empty log → error entry in extra

**Fix:**
- Never reduce basis set quality (customer requirement is binding)
- Never reduce active space or convergence thresholds
- Allowed: density fitting (RI/df), level_shift, damping, init_guess, max_cycle, DIIS
- Geometry issues: try once, if still failing → report to user
- Each attempt changes ONE variable, documented in `change_description`

## 6. Artifact Changes

`pre_done` now also emits a `run_command` artifact:
```
python -u jobs/{task_id}/script.py > jobs/{task_id}/log 2>&1
```
HPC agent uses this to know how to execute.

## 7. Files Changed

| File | Change |
|------|--------|
| `contracts/src/contracts/pyscf_task.py` | Rewrite Params and Result |
| `agents/pyscf/nodes.py` | Rewrite generate_input, parse_output; add fix_input |
| `agents/pyscf/agent.py` | Update routing for fix_input path |
| `prompts/pyscf/pyscf_generate.md` | New — generation prompt |
| `prompts/pyscf/pyscf_parse.md` | New — parsing prompt |
| `prompts/pyscf/pyscf_fix.md` | New — debugging/fix prompt |
| `agents/supervisor/agent.py` | Update `_build_params` for PySCF |

## 8. Verification

```bash
# Unit: LLM generates script from task description
uv run python -c "
from agents.pyscf.nodes import generate_input
# test with N2 PBE/cc-pVDZ, H2O B3LYP/6-31G*, Cr2 CASSCF(6,6)
"

# Unit: LLM parses a known log
uv run python -c "
from agents.pyscf.nodes import parse_output
# feed known PySCF output, verify energy extraction
"

# Integration: /run through multi-agent
uv run tower chat
> /run calculate N2 PBE/cc-pVDZ energy with pyscf
```
