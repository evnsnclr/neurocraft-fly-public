"""Render existing offline Circuit Playground measurements as a standalone HTML report.

No experiment is run. Only summary, plan and optional comparison-mask descriptors
are read; raw traces are not reverified. Output must be outside the evidence
directory and is created exclusively, so measured artifacts cannot be overwritten.
Uses only the Python standard library; plots are inline SVG and contain no script.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA = "circuit-playground-offline-v1"
MAX_JSON_BYTES = 32 * 1024 * 1024
EVENTS = (
    ("player_approach", "Player approach"),
    ("mob_approach", "Mob approach"),
    ("attack", "Attack attempt"),
    ("brush", "Brush contact"),
    ("food", "Sweet food"),
    ("light_off", "Light decrease"),
)
CASE_LABELS = {
    **dict(EVENTS),
    "slow_approach": "Slower approach",
    "two_approaches": "Two approach pulses",
}
COLORS = {
    "original": "#087f8c",
    "dnp01_silenced": "#c65a16",
    "sham": "#3a4853",
    "restored": "#7854a3",
}
CSS = """
:root{color-scheme:light;
font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;

color:#1c303d;
background:#f4f7fa;
font-size:15px}
*{box-sizing:border-box}
body{margin:0}

main{max-width:1480px;
margin:auto;
padding:42px 28px 70px}
h1{font-size:clamp(30px,4vw,48px);

letter-spacing:-.045em;
margin:8px 0 15px}
h2{font-size:25px;
margin:0 0 14px}
h3{font-size:19px}

p{line-height:1.65;
max-width:100ch}
section{background:white;
border:1px solid #dae3e9;

border-radius:14px;
padding:24px;
margin:24px 0}
header p{max-width:95ch}
.eyebrow{letter-spacing:.12em;

font-size:12px;
font-weight:750;
color:#087f8c}
.notice{background:#fff1db;
border-left:5px solid #bd6818;

padding:16px 20px;
border-radius:6px;
line-height:1.6}
.muted,small{color:#556d7a}
code{font-size:12px;

overflow-wrap:anywhere}
.hash{display:block;
margin:6px 0}
.scroll{overflow:auto}
table{width:100%;

border-collapse:collapse;
font-size:13px}
th,td{text-align:left;
vertical-align:top;
padding:12px;

border-bottom:1px solid #e2e9ee}
th{background:#edf3f7;
color:#304b5d;
font-size:12px}
th:first-child{
min-width:140px}
.matrix td{min-width:180px}
.program{display:block;
white-space:nowrap;
padding:2px 0}

.program b{display:inline-block;
min-width:24px;
color:#087f8c}
.source{display:block;
padding:5px 0 8px;

font-weight:650}
.number{white-space:nowrap;
font-variant-numeric:tabular-nums}
.badge{display:inline-block;

padding:4px 8px;
border-radius:20px;
background:#e3f0f2;
font-size:12px;
font-weight:650}
.warning{
background:#fff1db;
color:#81551f}
.plots{display:grid;
grid-template-columns:repeat(3,minmax(260px,1fr));

gap:16px}
.plot{margin:0;
border:1px solid #e2e9ee;
border-radius:9px;
padding:10px}
.plot svg{width:100%;

height:auto;
display:block}
.plot figcaption{font-size:12px;
line-height:1.5;
color:#556d7a}
.legend{
display:flex;
gap:20px;
flex-wrap:wrap;
margin:14px 0}
.legend span:before{content:'';
display:inline-block;

width:22px;
height:3px;
margin:0 6px 4px 0;
background:var(--swatch)}
details{margin:15px 0}
summary{
cursor:pointer;
font-weight:650;
padding:9px 0}
.metric{font-weight:650}
footer{font-size:12px;
line-height:1.7}

@media(max-width:1000px){.plots{grid-template-columns:1fr}
main{padding:24px 14px}
section{padding:17px}
}

@media print{body{background:white}
main{padding:0}
section{break-inside:avoid;
border-radius:0}
.scroll{
overflow:visible}
details{display:block}
table{font-size:10px}
.matrix td{min-width:0}
th,td{padding:6px}
}

"""


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("report metrics must be finite numbers or null")
    return f"{value:.4g}"


def label(name: str) -> str:
    if name.startswith("shuffle_"):
        return "Shuffled weights · seed " + name.removeprefix("shuffle_")
    if name.startswith("comparison_"):
        return "Comparison mask " + name.removeprefix("comparison_")
    return {
        "original": "Original weights",
        "sham": "Sham operation",
        "restored": "Restored original",
        "dnp01_silenced": "DNp01 pair silenced",
        "no_input": "No sensory input",
    }.get(name, name)


def canonical_digest(value: Any) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as stream:
        payload = stream.read(MAX_JSON_BYTES + 1)
    if len(payload) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds {MAX_JSON_BYTES} bytes: {path.name}")
    result = json.loads(payload, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return result, sha256(payload).hexdigest()


def program_counts(result: Mapping[str, Any]) -> str:
    counts = result.get("issued_program_counts")
    if not isinstance(counts, dict):
        return '<span class="muted">Issued program counts unavailable</span>'
    total = 0
    rows = []
    for program, count in sorted(counts.items()):
        if type(count) is not int or count < 0:
            raise ValueError("issued program counts must be nonnegative integers")
        total += count
        rows.append(f'<span class="program"><b>{count}</b> {escape(program)}</span>')
    return "".join(rows) + f"<small>{total} issued observations; no Minecraft applications</small>"


def fingerprint_table(fingerprints: Mapping[str, Any]) -> str:
    columns = ["original", "shuffle_7", "shuffle_11", "shuffle_23"]
    columns.extend(
        sorted(name for name in fingerprints if name.startswith("shuffle_") and name not in columns)
    )
    columns.append("no_input")
    rows = ["<div class=scroll><table class=matrix><thead><tr><th>Prescribed event</th>"]
    rows.extend(f"<th>{escape(label(name))}</th>" for name in columns)
    rows.append("</tr></thead><tbody>")
    for case, title in EVENTS:
        rows.append(f"<tr><th>{escape(title)}</th>")
        for condition in columns:
            result = fingerprints.get(condition, {}).get(case)
            if not isinstance(result, dict):
                rows.append('<td class="muted">Unavailable</td>')
                continue
            sources = result.get("selected_sources")
            if not isinstance(sources, list) or any(not isinstance(item, str) for item in sources):
                raise ValueError("selected_sources must be an array of strings")
            selected = ", ".join(sources) if sources else "None recorded"
            rows.append(
                f'<td><small>Selected sources</small><span class="source">{escape(selected)}</span>'
            )
            rows.append(program_counts(result) + "</td>")
        rows.append("</tr>")
    return "".join(rows) + "</tbody></table></div>"


def trace(result: Mapping[str, Any], channel: int | None) -> list[float] | None:
    values = result.get("ttmn_signed_evoked_mean" if channel is None else "ttmn_signed_evoked_lr")
    if values is None:
        return None
    if not isinstance(values, list) or not values:
        raise ValueError("TTMn traces must be nonempty arrays")
    selected = []
    for entry in values:
        value = entry
        if channel is not None:
            if not isinstance(entry, list) or len(entry) != 2:
                raise ValueError("TTMn L/R traces require two values per observation")
            value = entry[channel]
        number(value)
        if value is None:
            raise ValueError("TTMn trace samples must be finite, not null")
        selected.append(float(value))
    return selected


def plot(pilot: Mapping[str, Any], case: str, channel: int | None) -> str:
    names = ["original", "dnp01_silenced", "sham", "restored"]
    names.extend(sorted(name for name in pilot if name.startswith("comparison_")))
    series = []
    for name in names:
        values = trace(pilot.get(name, {}).get(case, {}), channel)
        if values is not None:
            series.append((name, values))
    title = (
        "Signed evoked pair mean"
        if channel is None
        else f"Signed evoked TTMn {'left' if channel == 0 else 'right'}"
    )
    if not series:
        return f'<figure class="plot"><p>{escape(title)}: unavailable</p></figure>'
    length = len(series[0][1])
    if any(len(values) != length for _, values in series):
        raise ValueError("plotted conditions must have equal observation counts")
    limit = max(abs(value) for _, values in series for value in values)
    limit = limit * 1.12 if limit else 1e-12
    if not math.isfinite(limit):
        raise ValueError("TTMn values exceed finite plotting range")
    width, height, left, right, top, bottom = 440, 250, 66, 15, 24, 43
    inner_width, inner_height = width - left - right, height - top - bottom

    def point(index: int, value: float) -> str:
        x = left + inner_width * index / max(1, length - 1)
        y = top + inner_height * (1 - value / limit) / 2
        return f"{x:.2f},{y:.2f}"

    svg = [
        f'<figure class="plot"><svg viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{escape(title)}">'
    ]
    svg.append(f"<title>{escape(CASE_LABELS.get(case, case))}: {escape(title)}</title>")
    for fraction, value in ((0, limit), (0.5, 0.0), (1, -limit)):
        y = top + fraction * inner_height
        svg.append(f'<line x1="{left}" y1="{y}" x2="{width - right}" y2="{y}" stroke="#dce5eb"/>')
        svg.append(
            f'<text x="{left - 6}" y="{y + 4}" text-anchor="end" font-size="10" '
            f'fill="#536b79">{escape(number(value))}</text>'
        )
    for index in sorted({0, (length - 1) // 2, length - 1}):
        x = left + inner_width * index / max(1, length - 1)
        svg.append(
            f'<text x="{x}" y="{height - 23}" text-anchor="middle" '
            f'font-size="11" fill="#536b79">{index}</text>'
        )
    # Comparison masks are deliberately faint and drawn first, retaining every curve.
    for name, values in sorted(series, key=lambda item: not item[0].startswith("comparison_")):
        comparison = name.startswith("comparison_")
        color = COLORS.get(name, "#7c8992")
        dash = ' stroke-dasharray="5 3"' if name in {"sham", "restored"} else ""
        points = " ".join(point(index, value) for index, value in enumerate(values))
        svg.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="{1 if comparison else 2}" '
            f'opacity="{0.4 if comparison else 0.95}"{dash}>'
            f"<title>{escape(label(name))}</title></polyline>"
        )
    svg.append(
        f'<text x="{left + inner_width / 2}" y="{height - 5}" text-anchor="middle" '
        'font-size="11" fill="#536b79">Observation tick (5 neural steps each)</text>'
    )
    svg.append(
        f"</svg><figcaption>{escape(title)}. Dimensionless activity minus the "
        "same-condition bright-sham trajectory.</figcaption></figure>"
    )
    return "".join(svg)


def exact_label(value: Any) -> str:
    if value is True:
        return "exact"
    return "not exact" if value is False else "unavailable"


def pilot_section(
    pilot: Mapping[str, Any], restoration: Mapping[str, Any], plan: Mapping[str, Any]
) -> str:
    cases = list(pilot.get("original", {}))
    cases.sort(key=lambda name: (name != "player_approach", name != "slow_approach", name))
    names = [name for name in ("original", "sham", "dnp01_silenced") if name in pilot]
    names.extend(sorted(name for name in pilot if name.startswith("comparison_")))
    names.extend(name for name in pilot if name not in names and name != "restored")
    if "restored" in pilot:
        names.append("restored")
    result = [
        "<section><h2>Internal-population intervention pilot</h2>",
        "<p>These curves subtract each condition's own no-event trajectory "
        "from its raw signed TTMn activity. "
        "The pair mean can hide opposing left/right changes; all three views are shown. "
        "Silencing effects here describe the prescribed model experiment, "
        "not biological necessity or Minecraft movement.</p>",
    ]
    result.append('<div class="legend">')
    result.extend(
        f'<span style="--swatch:{color}">{escape(label(name))}</span>'
        for name, color in COLORS.items()
    )
    result.append('<span style="--swatch:#7c8992">All comparison masks</span></div>')
    for case in cases:
        result.append(f"<h3>{escape(CASE_LABELS.get(case, case))}</h3><div class=plots>")
        result.extend(plot(pilot, case, channel) for channel in (None, 0, 1))
        result.append(
            "</div><div class=scroll><table><thead><tr><th>Condition</th><th>Peak |mean|</th>"
            "<th>|Δ peak|</th><th>Signed Δ peak</th><th>Peak / original</th>"
            "<th>Sum |mean| x tick</th><th>Onset tick</th>"
            "<th>Actual issued programs</th></tr></thead><tbody>"
        )
        for name in names:
            row = pilot[name].get(case)
            if not isinstance(row, dict):
                result.append(
                    f"<tr><th>{escape(label(name))}</th><td colspan=7>Unavailable</td></tr>"
                )
                continue
            difference = row.get("peak_difference_from_original")
            absolute = None if difference is None else abs(float(number_value(difference)))
            metrics = (
                row.get("peak_abs_mean"),
                absolute,
                difference,
            )
            result.append(f"<tr><th>{escape(label(name))}</th>")
            result.extend(f'<td class="number">{escape(number(value))}</td>' for value in metrics)
            ratio = row.get("peak_ratio_to_original")
            ratio_text = (
                "Undefined" if ratio is None and "peak_ratio_to_original" in row else number(ratio)
            )
            result.append(f'<td class="number">{escape(ratio_text)}</td>')
            result.append(
                f'<td class="number">{escape(number(row.get("integral_abs_mean_tick")))}</td>'
            )
            onset = row.get("onset_tick")
            onset_text = number(onset) if onset is not None else "No crossing / undefined"
            result.append(f'<td class="number">{escape(onset_text)}</td>')
            result.append(f"<td>{program_counts(row)}</td></tr>")
        result.append("</tbody></table></div>")
    result.append(
        f"<p><b>Onset rule:</b> {escape(plan.get('onset', 'Unavailable'))}. "
        "Ratios are undefined when the original peak is at or below "
        f"{escape(number(plan.get('ratio_floor')))}. "
        "Missing metrics remain unavailable; null onset is not a measured zero latency.</p>"
    )
    result.append(
        "<p><b>Exact replay checks:</b> "
        + "; ".join(
            f"{escape(label(name))}: {escape(exact_label(value))}"
            for name, value in restoration.items()
        )
        + ". These compare recorded numerical/semantic rows after fresh resets, "
        "not world restoration.</p></section>"
    )
    return "".join(result)


def number_value(value: Any) -> float:
    number(value)
    if value is None:
        raise ValueError("a numerical effect cannot be null")
    return float(value)


def matching_section(plan: Mapping[str, Any], masks: Mapping[str, Any] | None) -> str:
    result = [
        "<section><h2>Comparison-mask matching</h2>",
        f"<p class=notice>{escape(plan.get('matching_scope', 'Matching scope unavailable'))}. "
        "Anatomical size/side/sign restrictions are distinct from activity/strength ranking. "
        "These masks are a descriptive comparison sample, not a biological null distribution. "
        "Do not interpret poorly matched masks as equivalent circuit controls.</p>",
    ]
    if masks is None:
        return "".join(result) + "<p>Comparison-mask descriptors unavailable.</p></section>"
    result.append(
        "<div class=scroll><table><thead><tr><th>Mask</th><th>Cell type / body IDs</th>"
        "<th>Pair RMS feature distance</th><th>Original descriptive calipers</th>"
        "<th>Selection</th></tr></thead><tbody>"
    )
    for index, pair in enumerate(masks.get("pairs", [])):
        within = pair.get("within_original_activity_calipers")
        status = (
            "Within descriptive calipers"
            if within is True
            else "Outside descriptive calipers"
            if within is False
            else "Unavailable"
        )
        ids = ", ".join(str(value) for value in pair.get("body_ids", []))
        badge_class = "badge warning" if within is False else "badge"
        result.append(
            f"<tr><th>Comparison {index:02d}</th><td>{escape(pair.get('type', 'Unavailable'))}"
            f"<br><small>{escape(ids)}</small></td>"
            f"<td>{escape(number(pair.get('pair_rms_feature_distance')))}</td>"
            f'<td><span class="{badge_class}">{escape(status)}</span></td>'
            f"<td>Rank {escape(pair.get('matching_rank', 'Unavailable'))}; "
            f"seed {escape(pair.get('seed', 'Unavailable'))}</td></tr>"
        )
    return "".join(result) + "</tbody></table></div></section>"


def render(
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    masks: Mapping[str, Any] | None,
    summary_sha256: str,
    directory_name: str,
) -> str:
    if summary.get("status") != "complete" or plan.get("schema") != SCHEMA:
        raise ValueError(
            "only completed circuit-playground-offline-v1 measurements can be rendered"
        )
    fingerprints, pilot = summary.get("fingerprints"), summary.get("pilot")
    if not isinstance(fingerprints, dict) or not isinstance(pilot, dict):
        raise ValueError("summary requires fingerprint and pilot mappings")
    origin = plan.get("origin", {})
    source_commit = origin.get("source_commit", "Unavailable")
    parts = [
        "<!doctype html><html lang=en><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        "<title>Circuit Playground · offline evidence</title><style>",
        CSS,
        "</style><main><header>",
        "<div class=eyebrow>NEUROCRAFT FLY · OFFLINE MODEL EXPERIMENT</div>",
        "<h1>What changes when the wiring changes?</h1>",
        f"<p>Recorded result: <b>{escape(directory_name)}</b>. "
        "This report displays existing measurements; it runs no model.</p>",
        "<p class=notice><b>Prescribed offline inputs and issued scripted programs.</b> "
        "No Minecraft applications, human play, biological necessity, cognition "
        "or recovered natural behavior are established. "
        "Program counts below are issued observations, including baseline CRUISE; "
        "these are not applied actions or independent trials.</p>",
        f"<code class=hash>Result SHA-256 · {escape(summary_sha256)}</code>",
        "<code class=hash>Plan identity · "
        f"{escape(summary.get('plan_sha256', 'Unavailable'))}</code>",
        f"<code class=hash>Recorded base source commit · {escape(source_commit)}</code></header>",
        "<section><h2>Six-event response fingerprint</h2>"
        "<p>Each cell reports selected source names "
        "and every issued program count in the saved summary. An empty selected-source "
        "list is not the same as no issued movement: CRUISE can remain. "
        "These observations share a prescribed history within each "
        "condition; their count is not a number of independent behavioral repetitions.</p>",
        fingerprint_table(fingerprints),
        "</section>",
        pilot_section(pilot, summary.get("restoration_exact", {}), plan),
        matching_section(plan, masks),
        "<section><h2>Provenance and limits</h2>",
        f"<p>{escape(summary.get('claim_scope', 'Scope unavailable'))}. "
        f"Timing resolution: {escape(plan.get('timing_resolution', 'Unavailable'))}. "
        "The model uses assigned dynamics, engineered senses and hand-chosen readouts "
        "selecting scripted body programs. A deterministic restoration/rerun establishes "
        "numerical reproducibility within this experiment, not an independent sample.</p>",
        "<p>Recorded Git HEAD identifies the base checkout. The source hashes below "
        "identify measured files; this report does not establish a clean tree or "
        "publication readiness. The renderer verifies the plan and optional matching "
        "descriptor files it reads against the result manifest; it does not reverify "
        "all neural traces or rerun the experiment.</p><details>"
        "<summary>Recorded source-file hashes</summary>",
        "<div class=scroll><table><thead><tr><th>Source</th><th>SHA-256</th></tr></thead><tbody>",
    ]
    for name, item in sorted(plan.get("sources", {}).items()):
        parts.append(
            f"<tr><td>{escape(name)}</td>"
            f"<td><code>{escape(item.get('sha256', 'Unavailable'))}</code></td></tr>"
        )
    parts.append(
        "</tbody></table></div></details></section><footer>Private/local evidence view. "
        "Charts show signed raw TTMn differences from a same-condition no-event trajectory, "
        "not spikes or calibrated biological firing rates. No network access, external scripts "
        "or external assets are used by this report.</footer></main></html>"
    )
    return "".join(parts)


def write_report(input_path: Path, output_path: Path) -> Path:
    summary_path = input_path / "summary.json" if input_path.is_dir() else input_path
    directory = summary_path.resolve().parent
    destination = output_path.resolve()
    if destination.is_relative_to(directory):
        raise ValueError("report output must be outside the frozen evidence directory")
    summary, summary_hash = read_json(summary_path)
    plan, plan_hash = read_json(directory / "plan.json")
    manifest = summary.get("files", {})
    if manifest.get("plan.json") != plan_hash or summary.get("plan_sha256") != canonical_digest(
        plan
    ):
        raise ValueError("plan bytes or semantic identity differ from the recorded summary")
    masks = None
    masks_path = directory / "frozen-comparison-masks.json"
    if masks_path.exists():
        masks, masks_hash = read_json(masks_path)
        if manifest.get(masks_path.name) != masks_hash:
            raise ValueError("comparison-mask descriptor differs from the recorded summary")
    document = render(summary, plan, masks, summary_hash, directory.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(document)
        stream.write("\n")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="evidence directory or summary.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new HTML file outside evidence directory"
    )
    args = parser.parse_args(argv)
    try:
        result = write_report(args.input, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Report refused: {exc}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
