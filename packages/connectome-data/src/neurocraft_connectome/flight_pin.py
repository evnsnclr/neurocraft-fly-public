"""Independent trusted identity for the MaleCNS flight-steering fixture.

This pin lives in its own module on purpose. ``loader.py`` and ``model.py`` are
hashed by ``_loader_implementation_sha256`` into every fixture graph identity, so
a single byte added to either file rotates ``runtime_identity_sha256`` and every
downstream graph, adapter, decoder, and trace hash. A second fixture therefore
declares its identity here and passes it to the existing
``load_malecns_fixture(..., expected_pin=..., expected_file_bytes=...)``
parameters instead of editing the loader.

``mcns-v1.0-flight-steering-v1`` coexists with ``mcns-v1.0-flight-power-v1``. It
does not supersede it: the flight-power fixture remains the sole authority for
``BOUNDARY_ROLE_BODY_IDS`` and for the G1 full-artifact boundary overlay.

The fixture is produced by ``scripts/extract_flight_fixture.py`` together with
``scripts/flight_fixture_circuit.py`` and ``scripts/flight_fixture_sources.py``
from the same three generation-qualified source objects the flight-power fixture
uses.
"""

from __future__ import annotations

from neurocraft_connectome.model import FixturePin, SourceArtifact

MALECNS_FLIGHT_FIXTURE_V1_PIN = FixturePin(
    dataset_id="male-cns:v1.0",
    fixture_id="mcns-v1.0-flight-steering-v1",
    schema_version=1,
    file_sha256="71d9a56c81914d02af8286745a23f320e37531d040b8249ff50719a26871b05b",
    payload_sha256="bd1484024bb4fa891ebf59e1067df2d064e198d67c3557d94a0fd9bda99f9fe5",
    semantic_sha256="f57e4a416a9673f34a120b807d79e0060eebcbea32156cc27d5cbea7f1be620f",
    source_manifest_sha256=("9412105f54884a57521bbdf7fe46659caab559ef7877e6a62af95f50e1ee3768"),
    source_artifacts=(
        SourceArtifact(
            source_id="annotations",
            filename="body-annotations-male-cns-v1.0-minconf-0.5.feather",
            byte_length=14_483_314,
            sha256="2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
            generation="1780494878811468",
            url=(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
                "?generation=1780494878811468"
            ),
        ),
        SourceArtifact(
            source_id="neurotransmitters",
            filename="body-neurotransmitters-male-cns-v1.0.feather",
            byte_length=43_282_834,
            sha256="95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
            generation="1780894899156750",
            url=(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                "flat-connectome/body-neurotransmitters-male-cns-v1.0.feather"
                "?generation=1780894899156750"
            ),
        ),
        SourceArtifact(
            source_id="weights",
            filename="connectome-weights-male-cns-v1.0-minconf-0.5.feather",
            byte_length=1_051_241_946,
            sha256="e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
            generation="1780494887545976",
            url=(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
                "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
                "?generation=1780494887545976"
            ),
        ),
    ),
)
MALECNS_FLIGHT_FIXTURE_V1_FILE_BYTES = 284_949
"""Exact byte length of the pinned fixture, gated before any byte is read."""

MALECNS_FLIGHT_FIXTURE_V1_PATH = "data/fixtures/mcns-v1.0-flight-steering-v1.json"
"""Repository-relative location of the pinned bytes."""

MALECNS_FLIGHT_FIXTURE_V1_NODE_COUNT = 128
"""Emitted source neurons; asserted by the fixture tests, never inferred."""

MALECNS_FLIGHT_FIXTURE_V1_EDGE_COUNT = 887
"""Emitted directed source edges of weight >= 5 among the selected bodies."""
