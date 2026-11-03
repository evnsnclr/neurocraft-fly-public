from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import acquire_malecns as acq  # noqa: E402


class FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        url: str,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._body = io.BytesIO(body)
        self._url = url
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self._body.close()


class FakeOpener:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[urllib.request.Request] = []

    def open(
        self, request: urllib.request.Request, timeout: float
    ) -> FakeResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected mock HTTP request")
        return self.responses.pop(0)


class NoNetworkOpener:
    def open(self, request: urllib.request.Request, timeout: float) -> FakeResponse:
        raise AssertionError("verify-only attempted network access")


def make_bulk_resource(data: bytes) -> acq.ResourceSpec:
    filename = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    generation = "1780494878811468"
    return acq.ResourceSpec(
        resource_id="annotations",
        kind="bulk",
        filename=filename,
        url=(
            "https://storage.googleapis.com/flyem-male-cns/v1.0/"
            "connectome-data/flat-connectome/"
            f"{filename}?generation={generation}"
        ),
        generation=generation,
        expected_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        purpose="offline test fixture",
    )


def make_metadata_resource(data: bytes) -> acq.ResourceSpec:
    return acq.ResourceSpec(
        resource_id="release_notes",
        kind="metadata",
        filename="male-cns-v1.0-release-notes.html",
        url="https://male-cns.janelia.org/release/",
        generation=None,
        expected_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        purpose="offline metadata fixture",
    )


def make_manifest(
    source: acq.ResourceSpec, metadata: acq.ResourceSpec | None = None
) -> acq.DatasetManifest:
    return acq.DatasetManifest(
        schema_version=1,
        dataset_id="male-cns:v1.0",
        default_cache_dir="data/cache/male-cns-v1.0",
        sources=(source,),
        metadata_documents=(metadata,) if metadata is not None else (),
    )


class ManifestTests(unittest.TestCase):
    def test_manifest_contains_exact_pins_and_optional_metadata(self) -> None:
        manifest = acq.load_manifest()
        self.assertEqual(manifest.dataset_id, "male-cns:v1.0")
        self.assertEqual(
            [resource.resource_id for resource in manifest.sources],
            ["annotations", "neurotransmitters", "weights"],
        )
        self.assertEqual(
            [resource.expected_bytes for resource in manifest.sources],
            [14483314, 43282834, 1051241946],
        )
        self.assertEqual(
            [resource.generation for resource in manifest.sources],
            ["1780494878811468", "1780894899156750", "1780494887545976"],
        )
        self.assertEqual(
            [resource.sha256 for resource in manifest.sources],
            [
                "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2",
                "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621",
                "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1",
            ],
        )
        self.assertEqual(
            [resource.resource_id for resource in manifest.metadata_documents],
            ["download_index", "release_notes", "license_legalcode"],
        )
        self.assertEqual(
            [resource.expected_bytes for resource in manifest.metadata_documents],
            [44208, 21431, 48970],
        )
        self.assertEqual(
            [resource.sha256 for resource in manifest.metadata_documents],
            [
                "51144922891ccdc5cda2490d55b39ef69222134ed6944bbff2766379e403ad3c",
                "b2a4d40f0ae376f8c887cfbdbbc3dd10a6a165380b658fce8a20fd153a0a16aa",
                "6d55b998ed5c54f43426d059a8c549ed58a3321e5463e6a6af1c6b56ab78c333",
            ],
        )
        self.assertEqual(len(manifest.selected_resources(False)), 3)
        self.assertEqual(len(manifest.selected_resources(True)), 6)
        self.assertEqual(
            manifest.manifest_sha256,
            acq.CANONICAL_MANIFEST_SHA256,
        )

    def test_manifest_pin_mutation_fails_before_field_validation(self) -> None:
        canonical = acq.DEFAULT_MANIFEST_PATH.read_bytes()
        mutated = canonical.replace(
            b"1780494878811468",
            b"1780494878811469",
            1,
        )
        self.assertNotEqual(canonical, mutated)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.json"
            path.write_bytes(mutated)
            with self.assertRaisesRegex(
                acq.ManifestError, "canonical manifest SHA-256 mismatch"
            ):
                acq.load_manifest(path)

    def test_allowlist_rejects_wrong_scheme_host_path_and_query(self) -> None:
        safe = make_bulk_resource(b"safe")
        with self.assertRaises(acq.UnsafeSourceError):
            acq.validate_source_url(safe.url.replace("https://", "http://"))
        with self.assertRaises(acq.UnsafeSourceError):
            acq.validate_source_url(safe.url.replace("storage.googleapis.com", "example.org"))
        with self.assertRaises(acq.UnsafeSourceError):
            acq.validate_source_url(
                "https://storage.googleapis.com/flyem-male-cns/v1.0/other.feather"
            )
        with self.assertRaises(acq.UnsafeSourceError):
            acq.validate_source_url(safe.url + "&alt=media")
        self.assertEqual(
            acq.validate_source_url("https://male-cns.janelia.org/download/"),
            "download_index",
        )

    def test_redirect_handler_rejects_off_host_target(self) -> None:
        source = make_bulk_resource(b"safe")
        request = urllib.request.Request(source.url)
        handler = acq.SameResourceRedirectHandler()
        with self.assertRaises(acq.UnsafeSourceError):
            handler.redirect_request(
                request,
                io.BytesIO(),
                302,
                "Found",
                {},
                "https://example.org/payload",
            )

    def test_redirect_handler_rejects_generation_change(self) -> None:
        source = make_bulk_resource(b"safe")
        request = urllib.request.Request(source.url)
        handler = acq.SameResourceRedirectHandler()
        with self.assertRaises(acq.UnsafeSourceError):
            handler.redirect_request(
                request,
                io.BytesIO(),
                302,
                "Found",
                {},
                source.url.replace("1780494878811468", "1780494878811469"),
            )


class AcquisitionTests(unittest.TestCase):
    def test_fresh_download_verifies_and_promotes_atomically(self) -> None:
        data = b"deterministic feather stand-in"
        resource = make_bulk_resource(data)
        response = FakeResponse(
            data,
            url=resource.url,
            headers={"Content-Length": str(len(data))},
        )
        opener = FakeOpener([response])

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            result = acq.acquire_source(
                cache, resource, opener=opener, timeout=1.0
            )
            target = cache / resource.filename
            partial = cache / f"{resource.filename}.part"
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(partial.exists())

        request_headers = {
            key.lower(): value for key, value in opener.requests[0].header_items()
        }
        self.assertEqual(request_headers["accept-encoding"], "identity")
        self.assertNotIn("authorization", request_headers)
        self.assertNotIn("cookie", request_headers)

    def test_partial_download_resumes_with_validated_range(self) -> None:
        data = b"0123456789abcdef"
        prefix = data[:6]
        suffix = data[6:]
        resource = make_bulk_resource(data)
        response = FakeResponse(
            suffix,
            url=resource.url,
            status=206,
            headers={
                "Content-Length": str(len(suffix)),
                "Content-Range": f"bytes {len(prefix)}-{len(data) - 1}/{len(data)}",
            },
        )
        opener = FakeOpener([response])

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            partial = cache / f"{resource.filename}.part"
            partial.write_bytes(prefix)
            result = acq.acquire_source(
                cache, resource, opener=opener, timeout=1.0
            )
            self.assertEqual(result.status, "resumed")
            self.assertEqual(result.resumed_from, len(prefix))
            self.assertEqual((cache / resource.filename).read_bytes(), data)
            self.assertFalse(partial.exists())

        request_headers = {
            key.lower(): value for key, value in opener.requests[0].header_items()
        }
        self.assertEqual(request_headers["range"], f"bytes={len(prefix)}-")

    def test_ignored_range_restarts_partial_without_duplication(self) -> None:
        data = b"complete replacement"
        resource = make_bulk_resource(data)
        response = FakeResponse(
            data,
            url=resource.url,
            status=200,
            headers={"Content-Length": str(len(data))},
        )
        opener = FakeOpener([response])

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            partial = cache / f"{resource.filename}.part"
            partial.write_bytes(data[:4])
            result = acq.acquire_source(
                cache, resource, opener=opener, timeout=1.0
            )
            self.assertEqual(result.status, "restarted")
            self.assertEqual((cache / resource.filename).read_bytes(), data)

    def test_hash_mismatch_keeps_partial_and_never_promotes(self) -> None:
        expected = b"expected bytes"
        received = b"tampered bytes"
        self.assertEqual(len(expected), len(received))
        resource = make_bulk_resource(expected)
        opener = FakeOpener(
            [
                FakeResponse(
                    received,
                    url=resource.url,
                    headers={"Content-Length": str(len(received))},
                )
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with self.assertRaises(acq.VerificationError):
                acq.acquire_source(
                    cache, resource, opener=opener, timeout=1.0
                )
            self.assertFalse((cache / resource.filename).exists())
            self.assertEqual(
                (cache / f"{resource.filename}.part").read_bytes(), received
            )

    def test_final_response_url_cannot_leave_allowlisted_host(self) -> None:
        data = b"redirected"
        resource = make_bulk_resource(data)
        opener = FakeOpener(
            [
                FakeResponse(
                    data,
                    url="https://example.org/redirected",
                    headers={"Content-Length": str(len(data))},
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(acq.UnsafeSourceError):
                acq.acquire_source(
                    Path(directory), resource, opener=opener, timeout=1.0
                )

    def test_verify_rejects_hard_linked_artifact(self) -> None:
        data = b"hard-linked cache artifact"
        resource = make_bulk_resource(data)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            original = cache / "original"
            target = cache / resource.filename
            original.write_bytes(data)
            os.link(original, target)
            with self.assertRaisesRegex(
                acq.VerificationError, "single-link regular file"
            ):
                acq.verify_source(cache, resource)


class CliTests(unittest.TestCase):
    def test_verify_only_uses_no_network_and_emits_success_receipt(self) -> None:
        data = b"already cached"
        resource = make_bulk_resource(data)
        manifest = make_manifest(resource)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / resource.filename).write_bytes(data)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(acq, "load_manifest", return_value=manifest):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = acq.main(
                        [
                            "--cache-dir",
                            str(cache),
                            "--verify-only",
                        ],
                        opener=NoNetworkOpener(),
                    )
            self.assertEqual(code, 0)
            self.assertEqual(stderr.getvalue(), "")
            receipts = [json.loads(line) for line in stdout.getvalue().splitlines()]
            self.assertEqual(receipts[0]["status"], "verified")
            self.assertEqual(receipts[0]["expected_sha256"], resource.sha256)
            self.assertEqual(receipts[0]["observed_sha256"], resource.sha256)
            self.assertEqual(receipts[0]["url"], resource.url)
            self.assertEqual(receipts[-1]["mode"], "verify-only")

    def test_verify_only_missing_file_is_clear_error_and_creates_nothing(self) -> None:
        resource = make_bulk_resource(b"missing")
        manifest = make_manifest(resource)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "absent-cache"
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(acq, "load_manifest", return_value=manifest):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = acq.main(
                        ["--cache-dir", str(cache), "--verify-only"],
                        opener=NoNetworkOpener(),
                    )
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertFalse(cache.exists())
            receipts = [json.loads(line) for line in stderr.getvalue().splitlines()]
            self.assertEqual(receipts[0]["status"], "error")
            self.assertEqual(receipts[0]["error_type"], "verification")
            self.assertIn("missing", receipts[0]["message"])
            self.assertEqual(receipts[-1]["failed"], 1)

    def test_metadata_verification_is_opt_in(self) -> None:
        source_data = b"source"
        metadata_data = b"metadata"
        source = make_bulk_resource(source_data)
        metadata = make_metadata_resource(metadata_data)
        manifest = make_manifest(source, metadata)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            (cache / source.filename).write_bytes(source_data)
            first, first_failures = acq.process_resources(
                manifest,
                cache,
                verify_only=True,
                include_metadata=False,
                opener=NoNetworkOpener(),
                timeout=1.0,
            )
            self.assertEqual(first_failures, 0)
            self.assertEqual([item["resource_id"] for item in first], ["annotations"])

            second, second_failures = acq.process_resources(
                manifest,
                cache,
                verify_only=True,
                include_metadata=True,
                opener=NoNetworkOpener(),
                timeout=1.0,
            )
            self.assertEqual(second_failures, 1)
            self.assertEqual(
                [item["resource_id"] for item in second],
                ["annotations", "release_notes"],
            )
            self.assertEqual(second[1]["status"], "error")


if __name__ == "__main__":
    unittest.main()
