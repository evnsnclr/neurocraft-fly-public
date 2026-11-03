# Research starter kit

Study a small, source-identified neural model without launching Minecraft.
The three notebooks introduce the graph, apply a synthetic stimulus, and compare
original weights with shuffled weights and no input. The included dataset and
figures can be inspected before installing anything.

Start with the [current model card](MODEL_CARD.md) for the full project's scope.
This tutorial uses **14 neurons and 22 directed edges** from the attributed
MaleCNS flight-power fixture. Its direct pulses and readout pools are declared
experimental choices. It does not reproduce the full network, the Minecraft
sensory encoder, the body programs or biological recordings.

## Choose a starting point

| Notebook | You will do | Question |
| --- | --- | --- |
| [01 · Meet the network](../notebooks/01_meet_the_network.ipynb) | Inspect source IDs, annotations and an adjacency plot. | What is anatomical data, and what is a modeling choice? |
| [02 · Stimulate and observe](../notebooks/02_stimulate_and_observe.ipynb) | Reexecute the experiment and plot inputs and named readouts over modeled time. | How does this fixture respond to a prescribed pulse? |
| [03 · Change and compare](../notebooks/03_change_and_compare.ipynb) | Compare original, seed-7 shuffled weights and no-input trajectories. | Which responses change under this particular control? |

[Example dataset and schema](../research/example-v1/README.md) ·
[Current architecture](ARCHITECTURE.md) · [Results and claim scope](SCIENTIFIC_CLAIMS.md) ·
[Evidence availability](EVIDENCE.md)

## Run the notebooks

Use Python **3.13.18** and uv **0.13.6**, the tested project tool versions; uv must
be exactly 0.13.6 (`pyproject.toml` sets `required-version`), for example
`pip install uv==0.13.6` or the standalone installer for that version at
`https://astral.sh/uv/0.13.6/install.sh`, and `uv python install 3.13.18` fetches
the interpreter if it is absent. Run the following from the source root. Installation downloads the locked Python
packages; the example needs no account, Minecraft, Java, native C build or bulk
data download. It uses the Python reference engine.

```sh
uv sync --frozen --all-packages --group research
uv run --frozen --group research python scripts/research_example.py verify
uv run --frozen --group research python scripts/run_research_notebooks.py \
  --output artifacts/research-notebooks-first-run
```

The runner executes all three notebooks with the current environment's Python,
one fresh local kernel per notebook. It writes executed notebooks, PNG/SVG
figures and a receipt. Output directories must be new; retain failed runs and
choose another directory for a retry. The notebooks themselves verify the
included dataset with the explicitly labeled numerical-tolerance mode, keep
identities and file hashes exact, and run the actual reference engine. Notebook
02 also reports the measured difference and whether numerical equality is exact.

To edit interactively, open the notebooks in a Jupyter-capable editor such as
VS Code, select this project's `.venv/bin/python` kernel, and run all cells from
the top. No persistent kernel registration or notebook web server is required
by the batch command above. Notebook cells execute local Python code; review
third-party changes before running a modified notebook.

The optional `research` dependency group contains the kernel, notebook executor,
notebook format library and Matplotlib. It is separately selected and exactly
locked. Running the Minecraft companion does not require that group. This
starter kit is checked on the development Mac; an independently tested platform
matrix has not been established.

## Regenerate the example dataset

```sh
uv run --frozen --group research python scripts/research_example.py generate \
  --output artifacts/my-research-example
uv run --frozen --group research python scripts/research_example.py verify \
  --dataset artifacts/my-research-example
```

`generate` refuses to overwrite an existing directory. `verify` checks file hashes,
identities and regenerated results. Exact byte equality is the default. The
explicit `--tolerant` option allows the documented floating-point comparison for
another platform while retaining structural and file-integrity checks; a tolerant
pass is reported separately from exact reproduction. See the dataset schema for
its bounds and the values actually compared.

The experiment is fixed before comparison: 100 updates of 10 modeled ms, zero
initial activity, left then right direct pulses, and the same non-input parameters
in every condition. The no-input condition applies zero drive; the requested pulse
schedule remains recorded. The seed-7 shuffle permutes raw-count weights on the
same endpoints and recomputes incoming normalization. It does not rewire edges.

The exported outcomes are model outputs from this experiment. Changes may increase
or decrease a readout, and a shuffle may retain responses. A peak over this finite
window does not establish the eventual peak or settling behavior. One fixed
shuffle gives an example comparison, not a null distribution or population estimate.

## Continue to the full network

The compact dataset teaches inspection and controlled comparison. All full
MaleCNS routes need separately acquired source tables and the derived graph.
Dynamic full-network probes additionally need the compiled native kernel;
annotation/structural census commands do not. Follow the **data acquisition and build** steps in the
[installation guide](INSTALL.md).

Existing headless tools have different scopes:

| Tool | Scope | Additional prerequisites |
| --- | --- | --- |
| `scripts/analyze_readout.py` | Annotation membership and structural connectivity, with no dynamic simulation | Verified full artifact and annotation table |
| `scripts/analyze_readout_candidates.py` | Candidate annotation census; optional `--activity` uses the older escape-v2 diagnostic | Census: artifact/annotations. Optional activity: native kernel, transmitter table and named calibration |
| `scripts/probe_interaction_inputs.py` | Current separate-event encoder under prescribed observations in original/shuffled/no-input conditions; no program policy or body | Native kernel, annotation/transmitter tables and named calibration |
| `scripts/benchmark_interaction_policy.py` | Current policy's issued requests and unpaced compute timing under prescribed observations; no Minecraft application | Same full-model prerequisites as the event probe |

Start by inspecting the interfaces; these commands print help without running a
full-network experiment:

```sh
uv run --frozen python scripts/analyze_readout.py --help
uv run --frozen python scripts/analyze_readout_candidates.py --help
uv run --frozen python scripts/probe_interaction_inputs.py --help
uv run --frozen python scripts/benchmark_interaction_policy.py --help
```

The candidate census's optional legacy activity pass is not a reproduction of
the six-source current policy. These tools are separate experiments; this starter
dataset does not stand in for their execution.
Issued program requests are also distinct from actions applied in Minecraft.
The existing exact Minecraft trace replay route supports fixture/circuit traces;
it is not a general full-network replay API.

For an ML extension, declare a task, held-out inputs, baselines and metrics before
training a readout. For neuroscience, connect source IDs to annotations, record
sign/dynamics assumptions and compare multiple matched perturbations. Preserve
negative results and distinguish modeled responses from biological function.
