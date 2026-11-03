#!/usr/bin/env python3
"""Demonstrate AB-11 test power for the Fabric mod by mutating production Java source.

AB-11 requires the acceptance suite to fail whenever the neural computation is
decorative or bypassed; a guard must be shown to discriminate, not merely exist. For
each registered mutant this harness patches the *production* source transiently, runs
a named Gradle invocation, requires that run to fail *and* to report the named test as
failed, then restores the file and compares its bytes against ``git show HEAD:<path>``.
After every mutant it runs the unmutated baseline and requires it to pass, and it
records the concurrently-authored Python AB-11 suite under an enforced collection
floor. Mutated source is never committed: the tree is restored in a ``finally`` block
on every path, and the harness refuses to start unless ``mod/`` is clean. Every count
written to ``artifacts/ab11/`` is parsed from real Gradle, JUnit XML, GameTest, or
pytest output; a mutant that fails to trip its named test is recorded as a failed
mutant run, never defaulted, and a lane that does not report a count records ``null``
rather than a fabricated zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MOD_DIRECTORY: Final = "mod"
OUTPUT_DIRECTORY: Final = "artifacts/ab11"
JUNIT_RESULTS_DIRECTORY: Final = "mod/build/test-results/test"
PYTEST_JUNIT_XML: Final = "artifacts/ab11/pytest-ab11.xml"
AB11_PYTEST_TARGET: Final = "tests/ab11"
# The Python AB-11 lane only counts as evidence when it really collected the suite.
# The floor is the number of AB-11 tests the suite is required to keep carrying; a
# smaller collection means tests were lost, not that the lane passed.
PYTHON_MINIMUM_DISCOVERED: Final = 12

GRADLE_COMMON_ARGUMENTS: Final = (
    "./gradlew",
    "--no-daemon",
    "--dependency-verification",
    "strict",
)
# JUnit-lane runs (every mutant that uses it, and the baseline) must actually execute
# ``:test`` instead of restoring it FROM-CACHE, or the lane would report a previous
# build's verdict about source that is no longer on disk.
GRADLE_NO_BUILD_CACHE: Final = "--no-build-cache"

JUNIT_LANE: Final = "junit"
GAMETEST_LANE: Final = "gametest"

JUNIT_SKIPPED_SOURCE: Final = "junit-xml-skipped-attribute"
# The dedicated-server GameTest summary reports completed/passed/failed counts only.
GAMETEST_SKIPPED_SOURCE: Final = "gametest-summary-does-not-report"

GAMETEST_COMPLETE_PATTERN: Final = re.compile(r"(\d+) GAME TESTS COMPLETE")
GAMETEST_ALL_PASSED_PATTERN: Final = re.compile(r"All (\d+) required tests passed")
GAMETEST_FAILED_COUNT_PATTERN: Final = re.compile(r"(\d+) required tests failed")
# Minecraft prefixes every summary line with a timestamp and logger, so the failed-test
# bullet is matched inside the line rather than anchored to its start.
GAMETEST_FAILURE_LINE_PATTERN: Final = re.compile(r"\s-\s+([a-z0-9_]+:[a-z0-9_]+):\s*(\S.*)$")
CAMEL_BOUNDARY_PATTERN: Final = re.compile(r"(?<!^)(?=[A-Z])")


class HarnessError(RuntimeError):
    """Raised when a precondition, patch, or restore invariant is violated."""


@dataclass(frozen=True)
class Hunk:
    """One exact-match textual replacement that must occur exactly once in the target."""

    old: str
    new: str


@dataclass(frozen=True)
class GradleInvocation:
    """A named Gradle command plus the lane whose real output carries the verdict."""

    lane: str
    arguments: tuple[str, ...]

    def command(self) -> tuple[str, ...]:
        cache_arguments = (GRADLE_NO_BUILD_CACHE,) if self.lane == JUNIT_LANE else ()
        return GRADLE_COMMON_ARGUMENTS + cache_arguments + self.arguments

    def rendered(self) -> str:
        return "cd mod && " + " ".join(_quote(part) for part in self.command())


@dataclass(frozen=True)
class NamedTest:
    """A test the mutant must actually trip, addressed by declaring class and method."""

    class_name: str
    method_name: str

    def identifier(self) -> str:
        return f"{self.class_name}.{self.method_name}"

    def gametest_identifier(self) -> str:
        return f"{_snake_case(self.class_name)}_{_snake_case(self.method_name)}"


@dataclass(frozen=True)
class Mutant:
    """One AB-11 falsification: a transient production patch and the test it must trip."""

    mutant_id: str
    ab11_mutation: str
    summary: str
    target_file: str
    hunks: tuple[Hunk, ...]
    invocation: GradleInvocation
    named_tripped_tests: tuple[NamedTest, ...]
    route_note: str


@dataclass(frozen=True)
class LaneOutcome:
    """Test counts and failed identifiers parsed from one real run.

    ``skipped`` is ``None`` when the lane's own output carries no skip count;
    ``skipped_source`` names where the value came from, or why it is absent.
    """

    source: str
    discovered: int
    passed: int
    failed: int
    skipped: int | None
    skipped_source: str
    failed_test_ids: tuple[str, ...]
    failure_evidence: tuple[str, ...]


@dataclass(frozen=True)
class GradleRun:
    """A completed Gradle invocation and the log file its output was written to."""

    exit_code: int
    stdout: str
    log_path: Path


def _quote(part: str) -> str:
    return f'"{part}"' if " " in part or "*" in part else part


def _snake_case(name: str) -> str:
    return CAMEL_BOUNDARY_PATTERN.sub("_", name).lower()


MUTANTS: Final[tuple[Mutant, ...]] = (
    Mutant(
        mutant_id="MJ2A_STALE_TTL_DISABLED",
        ab11_mutation="stale command",
        summary=(
            "CommandValidator stops enforcing command freshness: the admission-time "
            "age/validThrough/past-target staleness checks are removed and an expired "
            "reservation is actuated at application time instead of failing closed."
        ),
        target_file="mod/src/main/java/dev/neurocraft/fly/control/CommandValidator.java",
        hunks=(
            Hunk(
                old=(
                    "        if (age > MAX_COMMAND_AGE_TICKS "
                    "|| currentGameTick > frame.validThroughTick()) {\n"
                    "            return ControllerFault.STALE_COMMAND;\n"
                    "        }\n"
                    "        if (frame.targetTick() < currentGameTick) {\n"
                    "            return ControllerFault.STALE_COMMAND;\n"
                    "        }\n"
                ),
                new=(
                    "        // AB-11 MUTANT MJ2A: command freshness authority disabled; "
                    "over-TTL frames admit.\n"
                ),
            ),
            Hunk(
                old=(
                    "        if (next.targetTick() < currentGameTick) {\n"
                    "            return reject(ControllerFault.DEADLINE_MISSED, next);\n"
                    "        }\n"
                ),
                new=(
                    "        if (next.targetTick() < currentGameTick) {\n"
                    "            // AB-11 MUTANT MJ2A: an expired reservation actuates "
                    "instead of failing closed.\n"
                    "            pending.removeFirst();\n"
                    "            lastAppliedValidThroughTick = next.validThroughTick();\n"
                    "            return ValidationResult.accepted(next);\n"
                    "        }\n"
                ),
            ),
        ),
        invocation=GradleInvocation(
            lane=JUNIT_LANE,
            arguments=("cleanTest", "test", "--tests", "*CommandValidatorTest"),
        ),
        named_tripped_tests=(
            NamedTest(
                "CommandValidatorTest", "rejectsSequenceStaleFutureWrongBindingAndWrongIdentity"
            ),
        ),
        route_note=(
            "Mutates the sole admission authority rather than substituting a validator "
            "double, so the existing named rejection test is the detector."
        ),
    ),
    Mutant(
        mutant_id="MJ2B_HOLD_LAST_ACTUATION",
        ab11_mutation="stale command",
        summary=(
            "CommandValidator keeps a copy of the last accepted frame and replays it when "
            "the queue is empty past the declared window, instead of latching "
            "DEADLINE_MISSED. This is the over-TTL hold-last actuation AB-03 forbids."
        ),
        target_file="mod/src/main/java/dev/neurocraft/fly/control/CommandValidator.java",
        hunks=(
            Hunk(
                old="    private long resetBoundaryTick = -1;\n",
                new=(
                    "    private long resetBoundaryTick = -1;\n"
                    "    // AB-11 MUTANT MJ2B: retains the last accepted frame for replay.\n"
                    "    private ActionFrame lastAcceptedFrame;\n"
                ),
            ),
            Hunk(
                old=(
                    "            return reject(ControllerFault.DEADLINE_MISSED);\n"
                    "        }\n"
                    "        if (next.targetTick() < currentGameTick) {\n"
                ),
                new=(
                    "            if (lastAcceptedFrame != null) {\n"
                    "                // AB-11 MUTANT MJ2B: hold-last actuation replaces "
                    "the missed-deadline fault.\n"
                    "                return ValidationResult.accepted(lastAcceptedFrame);\n"
                    "            }\n"
                    "            return reject(ControllerFault.DEADLINE_MISSED);\n"
                    "        }\n"
                    "        if (next.targetTick() < currentGameTick) {\n"
                ),
            ),
            Hunk(
                old=(
                    "        pending.removeFirst();\n"
                    "        lastAppliedValidThroughTick = next.validThroughTick();\n"
                    "        return ValidationResult.accepted(next);\n"
                ),
                new=(
                    "        pending.removeFirst();\n"
                    "        lastAppliedValidThroughTick = next.validThroughTick();\n"
                    "        lastAcceptedFrame = next;\n"
                    "        return ValidationResult.accepted(next);\n"
                ),
            ),
        ),
        invocation=GradleInvocation(
            lane=JUNIT_LANE,
            arguments=("cleanTest", "test", "--tests", "*CommandValidatorTest"),
        ),
        named_tripped_tests=(
            NamedTest(
                "CommandValidatorTest", "missingCommandAfterStreamBeginsLatchesDeadlineFault"
            ),
        ),
        route_note=(
            "ValidationResult's constructor invariant makes an in-record hold-last bypass "
            "impossible, so the validator itself is the only place the mutation can live."
        ),
    ),
    Mutant(
        mutant_id="MJ1B_ORBIT_SCRIPT",
        ab11_mutation="hidden goal",
        summary=(
            "FlyEntity.tick gains an unconditional scripted horizontal orbit applied before "
            "passive physics, independent of the neural bridge and of any accepted command."
        ),
        target_file="mod/src/main/java/dev/neurocraft/fly/entity/FlyEntity.java",
        hunks=(
            Hunk(
                old=(
                    "        FlyControllerRuntime.recordActuation(snapshot, decision, "
                    "lastActuation);\n"
                    "        integratePassivePhysics();\n"
                ),
                new=(
                    "        FlyControllerRuntime.recordActuation(snapshot, decision, "
                    "lastActuation);\n"
                    "        // AB-11 MUTANT MJ1B: scripted orbit unrelated to the "
                    "connectome runtime.\n"
                    "        setDeltaMovement(getDeltaMovement().add(\n"
                    "                0.02 * Math.cos(tickCount * 0.3),\n"
                    "                0.0,\n"
                    "                0.02 * Math.sin(tickCount * 0.3)));\n"
                    "        integratePassivePhysics();\n"
                ),
            ),
        ),
        invocation=GradleInvocation(lane=GAMETEST_LANE, arguments=("runGameTest",)),
        named_tripped_tests=(
            NamedTest("FlyEntityGameTests", "missingBridgeIsNeutralAndHasNoHorizontalIntent"),
        ),
        route_note=(
            "The GameTest lane spawns the registered entity on a dedicated server, so the "
            "scripted motion is observed on the real authoritative world transition."
        ),
    ),
    Mutant(
        mutant_id="MJ1A_MOB_INHERITANCE_SWAP",
        ab11_mutation="hidden goal",
        summary=(
            "FlyEntity is re-parented onto net.minecraft.world.entity.Mob, which makes the "
            "vanilla goal selector, target selector, and navigation reachable from the "
            "registered entity's inheritance path."
        ),
        target_file="mod/src/main/java/dev/neurocraft/fly/entity/FlyEntity.java",
        hunks=(
            Hunk(
                old="public final class FlyEntity extends Entity {\n",
                new=(
                    "// AB-11 MUTANT MJ1A: vanilla mob inheritance restored on the "
                    "registered entity.\n"
                    "public final class FlyEntity extends net.minecraft.world.entity.Mob {\n"
                ),
            ),
            Hunk(
                old="    protected void readAdditionalSaveData(CompoundTag tag) {\n",
                new="    public void readAdditionalSaveData(CompoundTag tag) {\n",
            ),
            Hunk(
                old="    protected void addAdditionalSaveData(CompoundTag tag) {\n",
                new="    public void addAdditionalSaveData(CompoundTag tag) {\n",
            ),
        ),
        invocation=GradleInvocation(
            lane=JUNIT_LANE,
            arguments=(
                "cleanTest",
                "test",
                "--tests",
                "*FlyEntityArchitectureTest",
                "-x",
                "runGameTest",
            ),
        ),
        named_tripped_tests=(
            NamedTest(
                "FlyEntityArchitectureTest",
                "directEntityInheritanceMakesMobGoalsAndNavigationUnavailable",
            ),
        ),
        route_note=(
            "The planned registry swap onto a gametest-source-set Mob double is compile "
            "infeasible: the gametest source set depends on main, never the reverse, so a "
            "main-source EntityType factory cannot name it (verified: 'package "
            "dev.neurocraft.fly.gametest does not exist'). Re-parenting the real "
            "production class has the same test power against the real registered entity "
            "and needs no committed test double. Only the JUnit lane is run because the "
            "GameTest lane is a task dependency of 'test': a Mob-parented FlyEntity cannot "
            "define LivingEntity synched data, so runGameTest would abort before the "
            "architecture assertion is ever evaluated."
        ),
    ),
)

BASELINE_INVOCATION: Final = GradleInvocation(lane=JUNIT_LANE, arguments=("cleanTest", "test"))


def _git(repo_root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def require_clean_mod_tree(repo_root: Path, target_files: Sequence[str]) -> None:
    """Refuse to run unless mod/ is clean and every target file matches HEAD exactly."""
    status = _git(repo_root, "status", "--porcelain", "--", MOD_DIRECTORY)
    if status.returncode != 0:
        raise HarnessError(f"git status failed: {status.stderr.strip()}")
    if status.stdout.strip():
        raise HarnessError(
            "refusing to mutate: mod/ has uncommitted changes:\n" + status.stdout.rstrip()
        )
    for target_file in target_files:
        require_matches_head(repo_root, target_file)


def require_matches_head(repo_root: Path, target_file: str) -> None:
    """Fail unless the working-tree copy of one target file is byte-identical to HEAD."""
    diff = _git(repo_root, "diff", "--quiet", "HEAD", "--", target_file)
    if diff.returncode != 0:
        raise HarnessError(f"target file differs from HEAD: {target_file}")


def matches_head_bytes(repo_root: Path, target_file: str) -> bool:
    """Observe whether one file's bytes equal ``git show HEAD:<path>`` exactly.

    This is the value recorded as ``source_restored_to_head``: an observation made
    after the restore, not an assumption that the restore worked.
    """
    completed = subprocess.run(
        ("git", "show", f"HEAD:{target_file}"),
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise HarnessError(
            f"git show HEAD:{target_file} failed: "
            f"{completed.stderr.decode('utf-8', 'replace').strip()}"
        )
    return (repo_root / target_file).read_bytes() == completed.stdout


def apply_hunks(path: Path, hunks: Sequence[Hunk]) -> None:
    """Apply every hunk, requiring each old snippet to occur exactly once."""
    source = path.read_text(encoding="utf-8")
    for index, hunk in enumerate(hunks, start=1):
        occurrences = source.count(hunk.old)
        if occurrences != 1:
            raise HarnessError(
                f"hunk {index} for {path} matched {occurrences} times; exactly one required"
            )
        source = source.replace(hunk.old, hunk.new)
    path.write_text(source, encoding="utf-8")


def restore_file(repo_root: Path, target_file: str) -> None:
    """Return one target file to its HEAD bytes and check that the restore landed."""
    checkout = _git(repo_root, "checkout", "--", target_file)
    if checkout.returncode != 0:
        raise HarnessError(f"failed to restore {target_file}: {checkout.stderr.strip()}")
    require_matches_head(repo_root, target_file)


@contextmanager
def mutated_source(repo_root: Path, mutant: Mutant) -> Iterator[None]:
    """Apply a mutant for the duration of the block and always restore afterwards."""
    target_path = repo_root / mutant.target_file
    try:
        apply_hunks(target_path, mutant.hunks)
        yield
    finally:
        restore_file(repo_root, mutant.target_file)


def unified_patch_text(mutant: Mutant) -> str:
    """Render a deterministic old->new description; its sha256 identifies the mutation."""
    lines = [f"--- a/{mutant.target_file}", f"+++ b/{mutant.target_file}"]
    for index, hunk in enumerate(mutant.hunks, start=1):
        lines.append(f"@@ hunk {index} @@")
        lines.extend(f"-{line}" for line in hunk.old.splitlines())
        lines.extend(f"+{line}" for line in hunk.new.splitlines())
    return "\n".join(lines) + "\n"


def patch_sha256(mutant: Mutant) -> str:
    return hashlib.sha256(unified_patch_text(mutant).encode("utf-8")).hexdigest()


def run_gradle(repo_root: Path, invocation: GradleInvocation, log_path: Path) -> GradleRun:
    """Run one Gradle invocation sequentially and persist its combined output."""
    completed = subprocess.run(
        invocation.command(),
        cwd=repo_root / MOD_DIRECTORY,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    return GradleRun(exit_code=completed.returncode, stdout=completed.stdout, log_path=log_path)


def junit_outcome(roots: Iterable[ET.Element], *, short_names: bool) -> LaneOutcome:
    """Fold one or more JUnit XML documents into a single lane outcome.

    ``short_names`` renders Gradle's fully-qualified class names as the bare
    ``Class.method`` identifiers ``NamedTest`` is matched against; pytest identifiers
    are kept whole because nothing is matched against them.
    """
    discovered = 0
    failed = 0
    skipped = 0
    failed_ids: list[str] = []
    evidence: list[str] = []
    for root in roots:
        for suite in root.iter("testsuite"):
            discovered += int(suite.get("tests", "0"))
            skipped += int(suite.get("skipped", "0"))
        for case in root.iter("testcase"):
            problems = list(case.findall("failure")) + list(case.findall("error"))
            if not problems:
                continue
            failed += 1
            class_name = case.get("classname") or ""
            method_name = case.get("name") or ""
            if short_names:
                class_name = class_name.rsplit(".", 1)[-1]
                method_name = method_name.removesuffix("()")
            identifier = f"{class_name}.{method_name}"
            failed_ids.append(identifier)
            for problem in problems:
                headline = (problem.get("message") or problem.get("type") or "").strip()
                evidence.append(f"{identifier}: {headline}")
    return LaneOutcome(
        source="junit-xml",
        discovered=discovered,
        passed=discovered - failed - skipped,
        failed=failed,
        skipped=skipped,
        skipped_source=JUNIT_SKIPPED_SOURCE,
        failed_test_ids=tuple(failed_ids),
        failure_evidence=tuple(evidence),
    )


def parse_junit_results(results_directory: Path) -> LaneOutcome:
    """Read every JUnit XML report Gradle wrote for the run."""
    reports = sorted(results_directory.glob("TEST-*.xml"))
    return junit_outcome((ET.parse(report).getroot() for report in reports), short_names=True)


def parse_junit_results_file(report: Path) -> LaneOutcome:
    """Read one pytest-written JUnit XML report (its root may be a testsuites element)."""
    return junit_outcome((ET.parse(report).getroot(),), short_names=False)


def parse_gametest_summary(output: str) -> LaneOutcome:
    """Read the dedicated-server GameTest summary block from real run output."""
    complete = GAMETEST_COMPLETE_PATTERN.search(output)
    if complete is None:
        raise HarnessError("GameTest output carried no 'GAME TESTS COMPLETE' summary line")
    discovered = int(complete.group(1))
    failed_ids: list[str] = []
    evidence: list[str] = []
    for line in output.splitlines():
        match = GAMETEST_FAILURE_LINE_PATTERN.search(line)
        if match is None:
            continue
        failed_ids.append(match.group(1))
        evidence.append(f"{match.group(1)}: {match.group(2).strip()}")
    reported_failures = GAMETEST_FAILED_COUNT_PATTERN.search(output)
    all_passed = GAMETEST_ALL_PASSED_PATTERN.search(output)
    if reported_failures is not None:
        failed = int(reported_failures.group(1))
    elif all_passed is not None:
        failed = 0
    else:
        raise HarnessError("GameTest output carried no required-test verdict line")
    return LaneOutcome(
        source="gametest-summary",
        discovered=discovered,
        passed=discovered - failed,
        failed=failed,
        # The GameTest summary block reports completed, passed, and failed required
        # tests only; there is no skip count to read, so none is invented.
        skipped=None,
        skipped_source=GAMETEST_SKIPPED_SOURCE,
        failed_test_ids=tuple(failed_ids),
        failure_evidence=tuple(evidence),
    )


def collect_outcome(repo_root: Path, lane: str, run: GradleRun) -> LaneOutcome:
    if lane == GAMETEST_LANE:
        return parse_gametest_summary(run.stdout)
    return parse_junit_results(repo_root / JUNIT_RESULTS_DIRECTORY)


def observed_failure(named: NamedTest, lane: str, outcome: LaneOutcome) -> bool:
    """Report whether a named test really appears as failed in the parsed run output."""
    if lane == GAMETEST_LANE:
        wanted = named.gametest_identifier()
        return any(failed.split(":")[-1] == wanted for failed in outcome.failed_test_ids)
    return named.identifier() in outcome.failed_test_ids


def evidence_for(named: NamedTest, lane: str, outcome: LaneOutcome) -> tuple[str, ...]:
    key = named.gametest_identifier() if lane == GAMETEST_LANE else named.identifier()
    return tuple(line for line in outcome.failure_evidence if key in line)


def outcome_record(outcome: LaneOutcome) -> dict[str, object]:
    return {
        "counts_source": outcome.source,
        "discovered": outcome.discovered,
        "passed": outcome.passed,
        "failed": outcome.failed,
        "skipped": outcome.skipped,
        "skipped_source": outcome.skipped_source,
        "failed_test_ids": list(outcome.failed_test_ids),
    }


def run_mutant_cycle(repo_root: Path, mutant: Mutant, output_directory: Path) -> dict[str, object]:
    """Patch, run, verify, restore, and describe exactly one mutant."""
    log_path = output_directory / "logs" / f"{mutant.mutant_id}.log"
    with mutated_source(repo_root, mutant):
        run = run_gradle(repo_root, mutant.invocation, log_path)
        outcome = collect_outcome(repo_root, mutant.invocation.lane, run)
    require_matches_head(repo_root, mutant.target_file)
    restored_to_head = matches_head_bytes(repo_root, mutant.target_file)

    named_ids = [named.identifier() for named in mutant.named_tripped_tests]
    tripped = {
        named.identifier(): observed_failure(named, mutant.invocation.lane, outcome)
        for named in mutant.named_tripped_tests
    }
    evidence: dict[str, list[str]] = {
        named.identifier(): list(evidence_for(named, mutant.invocation.lane, outcome))
        for named in mutant.named_tripped_tests
    }
    failures: list[str] = []
    if run.exit_code == 0:
        failures.append("gradle invocation succeeded; the mutant was not detected")
    if outcome.discovered == 0:
        failures.append("zero tests were discovered, so the lane demonstrated nothing")
    if not restored_to_head:
        failures.append(f"restored source does not match HEAD bytes: {mutant.target_file}")
    failures.extend(
        f"named test did not fail: {identifier}"
        for identifier, observed in tripped.items()
        if not observed
    )
    return {
        "mutant_id": mutant.mutant_id,
        "ab11_mutation": mutant.ab11_mutation,
        "summary": mutant.summary,
        "route_note": mutant.route_note,
        "target_file": mutant.target_file,
        "patch_sha256": patch_sha256(mutant),
        "patch_unified": unified_patch_text(mutant),
        "gradle_invocation": mutant.invocation.rendered(),
        "gradle_lane": mutant.invocation.lane,
        "exit_code": run.exit_code,
        "named_tripped_tests": named_ids,
        "named_tripped_observed": tripped,
        "named_failure_evidence": evidence,
        "source_restored_to_head": restored_to_head,
        "status": "tripped" if not failures else "not_tripped",
        "problems": failures,
        **outcome_record(outcome),
    }


def run_baseline(repo_root: Path, output_directory: Path) -> dict[str, object]:
    """Run the unmutated tree once and require a clean pass across both lanes."""
    log_path = output_directory / "logs" / "BASELINE.log"
    run = run_gradle(repo_root, BASELINE_INVOCATION, log_path)
    junit = parse_junit_results(repo_root / JUNIT_RESULTS_DIRECTORY)
    gametest = parse_gametest_summary(run.stdout)
    problems: list[str] = []
    if run.exit_code != 0:
        problems.append("baseline gradle invocation failed")
    if junit.failed or gametest.failed:
        problems.append("baseline reported failing tests")
    if junit.discovered == 0 or gametest.discovered == 0:
        problems.append("baseline discovered no tests in one lane")
    return {
        "gradle_invocation": BASELINE_INVOCATION.rendered(),
        "exit_code": run.exit_code,
        "junit_lane": outcome_record(junit),
        "gametest_lane": outcome_record(gametest),
        "status": "passed" if not problems else "failed",
        "problems": problems,
    }


def run_python_ab11_suite(repo_root: Path) -> dict[str, object]:
    """Record the concurrently-authored Python AB-11 suite, or note it as unavailable."""
    suite_directory = repo_root / AB11_PYTEST_TARGET
    junit_path = repo_root / PYTEST_JUNIT_XML
    invocation = (
        "UV_CACHE_DIR=.uv-cache uv run --frozen pytest -q "
        f"{AB11_PYTEST_TARGET} --junitxml {PYTEST_JUNIT_XML}"
    )

    def unavailable(detail: str, **extra: object) -> dict[str, object]:
        return {
            "record": "PYTHON_AB11_SUITE",
            "invocation": invocation,
            "status": "unavailable",
            "detail": detail,
            **extra,
        }

    if not suite_directory.is_dir():
        return unavailable(f"{AB11_PYTEST_TARGET} does not exist yet")
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["UV_CACHE_DIR"] = ".uv-cache"
    completed = subprocess.run(
        (
            "uv",
            "run",
            "--frozen",
            "pytest",
            "-q",
            AB11_PYTEST_TARGET,
            "--junitxml",
            PYTEST_JUNIT_XML,
        ),
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if not junit_path.is_file():
        return unavailable(
            "pytest wrote no JUnit XML report",
            exit_code=completed.returncode,
            tail=completed.stdout.strip().splitlines()[-10:],
        )
    outcome = parse_junit_results_file(junit_path)
    return {
        "record": "PYTHON_AB11_SUITE",
        "invocation": invocation,
        "status": "passed" if completed.returncode == 0 else "failed",
        "exit_code": completed.returncode,
        "junit_xml": PYTEST_JUNIT_XML,
        **outcome_record(outcome),
    }


def python_suite_problems(record: dict[str, object]) -> list[str]:
    """Gate the Python AB-11 lane: a real, complete, passing collection or nothing.

    An ``unavailable`` record, a collection below ``PYTHON_MINIMUM_DISCOVERED``, or any
    failed or skipped test is a harness failure. Without this gate the Java lanes could
    report a pass while the Python half of AB-11 never ran.
    """
    problems: list[str] = []
    status = record.get("status")
    if status != "passed":
        problems.append(f"python AB-11 suite status is {status!r}, not 'passed'")
    discovered = record.get("discovered")
    if not isinstance(discovered, int) or discovered < PYTHON_MINIMUM_DISCOVERED:
        problems.append(
            f"python AB-11 suite discovered {discovered!r} tests, below the enforced "
            f"floor of {PYTHON_MINIMUM_DISCOVERED}"
        )
    failed = record.get("failed")
    if failed != 0:
        problems.append(f"python AB-11 suite reported {failed!r} failed tests")
    skipped = record.get("skipped")
    if skipped != 0:
        problems.append(f"python AB-11 suite reported {skipped!r} skipped tests")
    return problems


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def selected_mutants(requested: Sequence[str]) -> tuple[Mutant, ...]:
    if not requested:
        return MUTANTS
    known = {mutant.mutant_id: mutant for mutant in MUTANTS}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise HarnessError(f"unknown mutant id(s): {', '.join(unknown)}")
    return tuple(known[name] for name in requested)


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing mod/ (default: this script's repository)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"evidence directory (default: <repo-root>/{OUTPUT_DIRECTORY})",
    )
    parser.add_argument(
        "--mutant",
        action="append",
        default=[],
        metavar="MUTANT_ID",
        help="run only the named mutant (repeatable); the baseline always runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    arguments = parse_arguments(argv)
    repo_root: Path = arguments.repo_root.resolve()
    output_directory: Path = (
        arguments.output_dir.resolve()
        if arguments.output_dir is not None
        else repo_root / OUTPUT_DIRECTORY
    )
    try:
        mutants = selected_mutants(arguments.mutant)
        target_files = sorted({mutant.target_file for mutant in mutants})
        require_clean_mod_tree(repo_root, target_files)
    except HarnessError as error:
        print(f"AB-11 harness refused to start: {error}", file=sys.stderr)
        return 2

    records: list[dict[str, object]] = []
    exit_code = 0
    try:
        for mutant in mutants:
            print(
                f"[AB-11] mutant {mutant.mutant_id} -> {mutant.invocation.rendered()}", flush=True
            )
            record = run_mutant_cycle(repo_root, mutant, output_directory)
            records.append(record)
            write_json(output_directory / f"java-mutant-{mutant.mutant_id}.json", record)
            print(f"[AB-11] {mutant.mutant_id}: {record['status']}", flush=True)
        print(f"[AB-11] baseline -> {BASELINE_INVOCATION.rendered()}", flush=True)
        baseline = run_baseline(repo_root, output_directory)
        print(f"[AB-11] baseline: {baseline['status']}", flush=True)
        python_record = run_python_ab11_suite(repo_root)
        print(f"[AB-11] python suite: {python_record['status']}", flush=True)
    except HarnessError as error:
        print(f"AB-11 harness aborted: {error}", file=sys.stderr)
        return 2
    finally:
        for target_file in sorted({mutant.target_file for mutant in MUTANTS}):
            restore_file(repo_root, target_file)

    tripped = all(record["status"] == "tripped" for record in records)
    python_problems = python_suite_problems(python_record)
    for problem in python_problems:
        print(f"[AB-11] python lane problem: {problem}", file=sys.stderr)
    if not tripped or baseline["status"] != "passed" or python_problems:
        exit_code = 1
    summary: dict[str, object] = {
        "suite": "AB-11 Java mutation power",
        "charter_target": "AB-11: acceptance tests must fail when neural computation is bypassed",
        "mutants": records,
        # ``all_mutants_tripped`` is only interpretable next to what was selected: the
        # full registry, or the subset named by --mutant.
        "mutant_selection": "requested-subset" if arguments.mutant else "full-registry",
        "selected_mutant_ids": [mutant.mutant_id for mutant in mutants],
        "registry_mutant_ids": [mutant.mutant_id for mutant in MUTANTS],
        "all_mutants_tripped": tripped,
        "baseline_pass": baseline,
        "python_ab11_suite": python_record,
        "python_minimum_discovered": PYTHON_MINIMUM_DISCOVERED,
        "python_ab11_suite_problems": python_problems,
        "python_ab11_suite_pass": not python_problems,
        "harness_status": "pass" if exit_code == 0 else "fail",
    }
    write_json(output_directory / "java-mutant-summary.json", summary)
    print(f"[AB-11] summary -> {output_directory / 'java-mutant-summary.json'}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
