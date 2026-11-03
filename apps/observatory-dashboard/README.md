# Trace-only observatory

Open `index.html` in a browser and choose a local causal `.ndjson` trace or bounded
event `.json` file. The page never opens a WebSocket, sends a request, uploads data,
or exposes a controller input. It renders only fields present in the selected file.

For canonical traces, the browser recomputes every SHA-256 entry hash, checks the
previous-hash chain, and requires the footer. This is an inspection aid, not the
authoritative verifier; use `neurocraft_protocol.verify_trace` for acceptance work.

The activity panel deliberately says **dimensionless model activity**. Mode, data
tier, and maximum claim are read from the manifest and are never inferred from the
appearance of the plot.
