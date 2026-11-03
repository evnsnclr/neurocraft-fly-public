# Public source verification. Raw historical session traces and private Git history
# are not distributed. Exactly one archival trace test is excluded below; protocol
# tests using synthetic fixtures and all runtime tests still execute. The private
# repository audit is omitted both as a command and as its pytest wrapper.
.PHONY: companion-ui-test bootstrap sync data-fixture data-sources test test-data lint format-check \
	typecheck audit jvm-audit fabric-sbom experiment-verify benchmark-fixture fabric-test fabric-client-test \
	sbom experiment report demo-prepare demo-run demo replay-demo live-demo live-demo-full \
	live-demo-eem live-demo-eem-v2 live-demo-escape live-demo-escape-v2 \
	live-demo-loom live-demo-program live-demo-interactions live-demo-recording \
	verify verify-local-data clean-generated \
	spmv-build

UV_CACHE_DIR := $(CURDIR)/.uv-cache
UV := UV_CACHE_DIR="$(UV_CACHE_DIR)" uv
UV_RUN := $(UV) run --frozen
UV_RUN_ISOLATED := $(UV) --no-config run --isolated --exact --frozen --no-dev \
	--no-python-downloads \
	--python .venv/bin/python3
DEMO_BUNDLE := artifacts/gametest/external-bridge-gametest-v1
MYPY_TARGETS := packages services \
	scripts/companion_app.py scripts/companion_bootstrap.py scripts/companion_downloads.py \
	scripts/build_companion.py scripts/build_sqlite_runtime.py scripts/java_runtime.py \
	scripts/fetch_companion_runtime.py scripts/process_lifecycle.py \
	scripts/public_alpha.py scripts/installed_mod.py scripts/build_public_release.py \
	scripts/research_example.py scripts/run_research_notebooks.py \
	scripts/action_latency_telemetry.py \
	scripts/analyze_reachability.py scripts/analyze_readout.py \
	scripts/readout_annotations.py scripts/readout_census.py \
	scripts/readout_connectivity.py \
	scripts/analyze_readout_candidates.py scripts/readout_candidates.py \
	scripts/analyze_vision_feasibility.py scripts/vision_columns.py \
	scripts/vision_circuits.py scripts/vision_report.py \
	scripts/audit_repository.py scripts/benchmark_controller.py \
	scripts/benchmark_full_controller.py scripts/calibrate_full_readout.py \
	scripts/benchmark_full_engine.py scripts/build_jvm_inventory.py \
	scripts/build_spmv_kernel.py scripts/extract_flight_fixture.py \
	scripts/flight_fixture_circuit.py scripts/flight_fixture_sources.py \
	scripts/fixture_harness.py scripts/measure_sign_overlay.py \
	scripts/measure_full_sign_overlay.py scripts/measure_escape_arc.py \
	scripts/probe_loom_encoder.py scripts/probe_program_policy.py scripts/probe_entity_loom.py \
	scripts/probe_full_controls.py \
	scripts/probe_event_populations.py scripts/probe_event_contexts.py \
	scripts/probe_program_controls.py scripts/summarize_interactive_session.py \
	scripts/summarize_comparison_session.py scripts/summarize_matched_trials.py \
	scripts/run_interaction_trials.py scripts/run_release_comparisons.py \
	scripts/export_capture_video.py \
	scripts/program_decision_telemetry.py \
	scripts/build_brain_viewer_layout.py scripts/run_brain_viewer.py \
	scripts/run_gauntlet_demo.py \
	scripts/prepare_gametest_demo.py \
	scripts/replay_minecraft_trace.py scripts/run_ab11_java_mutants.py \
	scripts/run_trm_e1.py \
	scripts/run_flygym_embodiment.py \
	scripts/run_connectome_service.py \
	scripts/scan_jvm_advisories.py \
	scripts/run_external_gametest.py scripts/run_fixture_experiment.py \
	scripts/run_program_authority_gametest.py \
	scripts/run_interactive_demo.py scripts/showcase_world.py \
	scripts/verify_fixture_pilot_snapshot.py \
	scripts/verify_paused_window.py

sync:
	$(UV) sync --frozen --all-packages

bootstrap: sync
	cd mod && ./gradlew --no-daemon --version

data-fixture:
	$(UV_RUN) pytest -q tests/data tests/connectome

data-sources:
	$(UV_RUN) python -B scripts/acquire_malecns.py --include-metadata

test: spmv-build
	$(UV_RUN) python -c 'from neurocraft_neural.spmv_kernel import load_c_kernel; print(load_c_kernel().identity.identity_sha256)'
	$(UV_RUN) pytest -q \
		--deselect tests/protocol/test_sensor_contract_v2.py::test_all_seven_historical_sensor_payloads_round_trip_exactly \
		--deselect tests/governance/test_audit_repository.py::test_repository_hygiene_and_g0_identities

test-data:
	$(UV_RUN) python -B -m unittest -v tests.data.test_acquire_malecns tests.data.test_extract_malecns_fixture

lint:
	$(UV_RUN) ruff check .

format-check:
	$(UV_RUN) ruff format --check .

typecheck:
	$(UV_RUN) mypy $(MYPY_TARGETS)

sbom:
	mkdir -p artifacts/sbom
	$(UV_RUN) cyclonedx-py environment .venv --pyproject pyproject.toml \
		--output-reproducible --of JSON -o artifacts/sbom/python.cdx.json

audit: sbom
	# PYSEC-2026-113 is a C++-only PreBufferMetadata path that Apache states is
	# not exposed by Python bindings; SECURITY.md records the narrow exception.
	$(UV_RUN) pip-audit --local --progress-spinner off --ignore-vuln PYSEC-2026-113
	$(UV_RUN) pip-licenses --format=markdown --with-urls

experiment-verify:
	$(UV_RUN) python scripts/verify_fixture_pilot_snapshot.py

experiment: experiment-verify

fabric-sbom:
	cd mod && ./gradlew --no-daemon --offline --dependency-verification strict \
		--stacktrace emitJvmDependencyGraph
	mkdir -p artifacts/sbom
	$(UV_RUN) python -B scripts/build_jvm_inventory.py \
		--graph mod/build/reports/jvm-dependency-graph.json \
		--verification mod/gradle/verification-metadata.xml \
		--lockfile mod/gradle.lockfile \
		--overrides docs/compliance/jvm-license-overrides.json \
		--sbom artifacts/sbom/jvm.cdx.json \
		--notices docs/compliance/JVM_THIRD_PARTY_NOTICES.md

jvm-audit:
	mkdir -p artifacts/audit
	$(UV_RUN) python -B scripts/scan_jvm_advisories.py \
		--lockfile mod/gradle.lockfile \
		--exceptions docs/security/jvm-advisory-exceptions.json \
		--output artifacts/audit/jvm-advisories.json \
		--receipt artifacts/audit/jvm-advisories.receipt.json

benchmark-fixture:
	$(UV_RUN) python scripts/benchmark_controller.py

spmv-build:
	$(UV_RUN) python -B scripts/build_spmv_kernel.py

fabric-test:
	cd mod && ./gradlew --no-daemon --dependency-verification strict --stacktrace clean check build runGameTest

fabric-client-test:
	cd mod && ./gradlew --no-daemon --dependency-verification strict --stacktrace runClientGameTest

demo-prepare:
	$(UV_RUN_ISOLATED) python -I scripts/prepare_gametest_demo.py \
		--output $(DEMO_BUNDLE) \
		--machine-label local-macos-arm64 \
		--os-label macOS

demo-run:
	$(UV_RUN_ISOLATED) python -I scripts/run_external_gametest.py --bundle $(DEMO_BUNDLE)

demo: demo-prepare demo-run

# Opt-in live probe of the ADR 0020 operator lane (PAUSE-gated body programs) against a
# real fixture service and the real mod bridge; writes under artifacts/ (gitignored).
program-authority-gametest:
	$(UV_RUN) python scripts/run_program_authority_gametest.py

replay-demo:
	$(UV_RUN) python scripts/replay_minecraft_trace.py \
		--trace $(DEMO_BUNDLE)/trace.ndjson \
		--stimulation sensor-pulse-v1=77185=1.0

live-demo:
	$(UV_RUN_ISOLATED) python -I scripts/run_interactive_demo.py

live-demo-trm:
	$(UV_RUN_ISOLATED) python -I scripts/run_interactive_demo.py \
		--controller-profile trm_cloned_readout_v1 --brain-viewer

live-demo-gauntlet:
	$(UV_RUN_ISOLATED) python -I scripts/run_gauntlet_demo.py --brain-viewer

# The full-artifact profile needs numpy/pyarrow (array-native backend + sign
# overlay), which the dev group provides and the isolated no-dev interpreter
# deliberately omits; the launcher still runs the service with `python -I` and
# a scrubbed child environment.
live-demo-full:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_wing_v1

# Engineered-Encoder Mode (ADR 0008 Mode 3): the same local-only full
# artifact with Minecraft observations additionally driving the labeled
# LC4/visual/bristle populations through a declared, hashed encoder.
live-demo-eem:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_eem_v1

# EEM v2 (temporal contrast): the same populations and per-capita placement
# as eem v1, but each channel drives its population with the deviation from a
# declared leaky running mean (alpha = 1/8 per game tick) instead of the
# absolute level; touch stays absolute. Refuses to start until the phase-2
# calibration (scripts/calibrate_full_readout.py --profile eem-v2) exists.
live-demo-eem-v2:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_eem_v2

# Escape reflex (EEM escape): the eem v1 sensory boundary (absolute encoder
# over the labeled populations) plus the declared TTMn jump-motoneuron
# readout, whose pooled activity adds one declared term to vertical thrust.
# Refuses to start until the phase-2 escape calibration
# (scripts/calibrate_full_readout.py --profile escape) exists.
live-demo-escape:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_escape_v1

# Escape reflex v2 (EEM escape v2): the same populations and TTMn pool as
# escape v1, with the touch channel alone moving under the declared temporal
# contrast (alpha = 1/8 per game tick) and the jump mapping rectified to the
# non-negative half of its declared codomain. Refuses to start until the
# phase-2 escape-v2 calibration
# (scripts/calibrate_full_readout.py --profile escape-v2) exists.
live-demo-escape-v2:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_escape_v2

# Approach-gated looming (EEM loom v1): the same populations, TTMn pool, and
# rectified jump decoder as escape v2, with the three threat channels driving
# their LC4 populations with the rectified RISE of proximity above a declared
# leaky running mean (alpha = 1/8 per game tick) instead of with the absolute
# level -- an approaching object fires, a standing wall does not. Declared in
# docs/encoders/eem-loom-v1.md. Refuses to start until the loom-v1 calibration
# (scripts/calibrate_full_readout.py --profile loom-v1) exists.
live-demo-loom:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_loom_v2

# The body-program lane: the SAME loom-v1 controller (same artifact, encoder,
# dynamics, decoder, and RunIdentity) plus the declared DNa01 steering readout,
# issuing program_action frames instead of raw motor actions. The mapping from
# the three labeled readouts to a scripted body program is hand-chosen
# engineering (ADR 0019); its identity rides on every frame as policy_hash; the
# programs are scripted presentation (ADR 0020); the connectome is not deciding
# anything. Writes a DIAGNOSTIC program-decisions.ndjson into the run directory
# (not content-addressed, not evidence). Refuses to start until the loom-v1
# calibration (scripts/calibrate_full_readout.py --profile loom-v1) exists.
#
# --action-latency is on for this lane alone, and it is diagnostic, not evidence.
# The first live program-lane sessions (2026-09-03) died on latency bursts -- action
# latencies of 87-99 ms against a 50 ms tick, operator-observed in gitignored local
# logs -- that the mod's own latency log cannot characterise: NeuralBridge spends a
# budget of FIRST_ACTION_LATENCY_SAMPLES = 8 samples per session and reports only
# offer_to_action / send_to_action totals, so it is silent for the rest of the run and
# never says WHERE the time went. This writes every frame's compute_ns /
# serialization_ns / write_ns split to action-latency.ndjson in the run directory.
live-demo-program:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_loom_program_v2 --action-latency

# V4 event/actor inputs and six declared neural gates; candidate until live verified.
live-demo-interactions:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_interactions_program_v2 --comparison-lanes --action-latency

# Human-operated OBS session. No diagnostic PNG capture; render FPS is a ceiling.
live-demo-recording:
	$(UV_RUN) python -I scripts/run_interactive_demo.py \
		--controller-profile full_mcns_interactions_program_v2 \
		--showcase-world --comparison-lanes --control-seed 7 \
		--action-latency --kernel-threads 8 --game-tick-rate 20 \
		--client-width 1280 --client-height 720 --client-max-fps 60

companion-ui-test:
	node --test tests/integration/companion_ui.test.cjs

verify: lint format-check typecheck test companion-ui-test audit jvm-audit experiment-verify fabric-test fabric-sbom fabric-client-test research-verify

verify-local-data:
	$(UV_RUN) python -B scripts/acquire_malecns.py --verify-only --include-metadata

clean-generated:
	@echo "Generated caches are ignored; remove them only with an explicit, narrow command."

.PHONY: verify-source doctor setup prepare run
verify-source:
	python3 -I scripts/build_public_release.py --verify-source .

doctor:
	python3 -I scripts/public_alpha.py doctor

setup:
	python3 -I scripts/public_alpha.py setup

prepare:
	python3 -I scripts/public_alpha.py prepare

run:
	python3 -I scripts/public_alpha.py run

# Headless research example and real notebook execution. Optional tools stay out
# of the companion runtime. Tolerant replay is labeled and checks all identities.
.PHONY: research-verify
research-verify:
	$(UV) sync --frozen --all-packages --group research
	$(UV_RUN) --group research mypy scripts/research_figures.py
	$(UV_RUN) --group research python scripts/research_example.py verify --tolerant
	mkdir -p artifacts
	$(UV_RUN) --group research python scripts/run_research_notebooks.py \
		--output "$$(mktemp -d "$(CURDIR)/artifacts/research-check.XXXXXX")/notebooks"
	$(UV_RUN) --group research pip-audit --local --progress-spinner off --ignore-vuln PYSEC-2026-113
	$(UV_RUN) --group research pip-licenses --format=json --with-urls > artifacts/research-licenses.json
