# Third-party software, data, and assets

The private companion runtime inventory, packaging necessity and license handling
are in [COMPANION_RUNTIME_NOTICES.md](docs/compliance/COMPANION_RUNTIME_NOTICES.md).
Bundled CPython and existing production wheels retain their original notices;
Java, Minecraft and bulk data are acquired separately under their own terms.

**Inventory update:** 2026-09-06, research starter kit and public local alpha.
Project software and associated documentation use the [MIT License](LICENSE).
Third-party software, data and assets retain their separate terms; this code
license does not grant rights to other original assets. Historical evidence
entries below retain their recorded scope.
The current objective is an interactive demo and its distribution (ADR 0019),
not advancement through the retired research-gate ladder.

## Local-alpha installation tooling

The optional `scripts/installed_mod.py` command uses Fabric Installer **1.1.2**
to obtain an isolated server, avoiding a second implementation of Fabric's
installation logic. Fabric's official version metadata listed it as the stable
installer when checked on 2026-09-05. Its source is
[Apache-2.0 licensed](https://github.com/FabricMC/fabric-installer/blob/master/LICENSE).
The installer is downloaded separately, not vendored or placed in the mod JAR.

| Download | SHA-256 | Use |
| --- | --- | --- |
| Fabric Installer 1.1.2 from `maven.fabricmc.net` | `61e035bf7bf70153e127440ce34de47c9036f0a2d0c65d1529454bd35ceefe4f` | Official installation of Minecraft 1.21.5 / Loader 0.16.10 into a new private directory. |
| Fabric API 0.128.2+1.21.5 from `maven.fabricmc.net` | `a82fd00827206e911936ed1e0ceaec6eb55d061ca5d3c5d63c7f0031426d29ae` | The already pinned API, installed separately beside the project mod. |

The helper enforces those upstream checksum pins before use. The official
installer resolves its transitive files; a local inventory detects changes to
the resulting runtime before launch. This inventory is not a signature or a
claim that every upstream dependency is vulnerability-free. The resulting
directory includes Minecraft and is never part of the public source/mod package.
Project dependency audit exceptions remain explicit in SECURITY.md.

## Current incorporated material

The repository contains attributed MaleCNS-derived fixtures, a compact research example and project lock/build metadata. It contains no vendored third-party source, game binary, third-party model, Minecraft asset, or bulk connectome object.

| Material | Exact identity | License / role | Evidence boundary |
|---|---|---|---|
| MaleCNS flight-power CI fixture | `data/fixtures/mcns-v1.0-flight-power-v1.json`; 36,707 bytes; SHA-256 `3076c22fd0c87041c601b865c9fc8e913466200491f80775060e28412a087ff8` | CC BY 4.0 derived data; structural CI fixture | Embedded attribution, citation, source generations/hashes, source-row lineage, change notice, and redistribution decision; not a full graph or executable model. |
| MaleCNS flight-steering fixture | `data/fixtures/mcns-v1.0-flight-steering-v1.json`; 284,949 bytes; SHA-256 `71d9a56c81914d02af8286745a23f320e37531d040b8249ff50719a26871b05b` | CC BY 4.0 derived data; 128-neuron, 887-edge structural fixture | Embedded attribution, citation, source hashes, source-row lineage, change notice and redistribution decision (`FIXTURE-MCNS-02`); not a full graph or executable model. |
| Python dependency lock | `uv.lock`; Python `>=3.11,<3.14`; uv `0.13.6` | Exact resolved Python build/runtime/test/tool graph | Frozen by `uv run --frozen`; component licenses are audited from resolved packages rather than inferred from this lock alone. |
| Gradle dependency controls | `mod/gradle.lockfile` and `mod/gradle/verification-metadata.xml` | Exact Java/Fabric coordinates and SHA-256 verification | External artifacts and metadata remain checksum verified. The cross-platform lock excludes only the presence of `java-objc-bridge` and Netty native epoll because Minecraft selects those modules by host OS; their exact artifacts remain SHA-256 verified. The metadata also records SHA-256 for nine exact Linux-native Minecraft/LWJGL JAR entries after each JAR's SHA-1 matched Mojang's immutable [Minecraft 1.21.5 version manifest](https://piston-meta.mojang.com/v1/packages/5fd126ffe444d6f323157903d91d3a82471c84e2/1.21.5.json). Four narrow trusted-artifact rules cover only the pinned Loom mapping JAR, its mapping-identity-specific remapped Fabric JAR group, and two exact split Minecraft JARs because Loom embeds cache-specific ZIP timestamps in those locally derived outputs. Fresh-cache strict local server and client builds passed, followed by all three jobs in private Actions run `32056567435`; this is not a repository-wide or remote-group waiver. |

## Incorporated Python runtime and evidence environment

The Python environment is frozen in `uv.lock`. External runtime dependencies are
`websockets==17.0.1`, `numpy==2.2.5` and `pyarrow==21.0.0`. NumPy and PyArrow are
declared by the packages that use them so a production-only environment can load
the full controller. Their versions and numerical source are unchanged. Remaining
direct packages are build, test, type, lint, audit, license and SBOM tools.

| Tool | Locked identity | License / role | Evidence boundary |
|---|---|---|---|
| `websockets` | `17.0.1` | BSD-3-Clause; loopback neural-service transport | Authentication, size/schema limits, bounded queues, and shutdown are integration-tested. |
| NumPy | `2.2.5` | BSD-3-Clause; deterministic array/data verification | Locked for source/artifact work; full runtime arrays are ignored derived data, not redistributed by this inventory. |
| PyArrow | `21.0.0` | Apache-2.0; Feather source inspection/extraction | Locked to preserve hash-bound evidence. `pip-audit` reports `PYSEC-2026-113` / `CVE-2026-25087`; the only exception is documented in `SECURITY.md` because Apache states the C++ `RecordBatchFileReader::PreBufferMetadata` path is not exposed by Python bindings and this repository has no such call. This is not a broad Arrow exception. |
| Hatchling | `1.31.0` | MIT; package build backend | Exact build-system version in every project package. |
| Development/audit tools | `cyclonedx-bom==7.3.1`, `mypy==2.3.0`, `pip-audit==2.10.1`, `pip-licenses==5.5.5`, `pytest==9.1.1`, `pytest-cov==7.1.0`, `ruff==0.16.0` | Build/test/audit tooling; licenses come from the resolved inventory | Not imported by the controller service; frozen in the development dependency group. |

`make sbom` reproducibly generated the ignored `artifacts/sbom/python.cdx.json`, CycloneDX 1.6 JSON, 86,911 bytes, SHA-256 `d3f8c14256651d939b62c9c774c0785081a523da67eed1c3b2fffb0a7f7ab105` (73 components in this resolution). `make audit` depends on this target, runs the exact PyArrow exception above, and emits the resolved license inventory. The ignored SBOM is a local audit product, not a committed release artifact; regenerate and re-hash it after any lock change.

No candidate below is an actual dependency merely because it was audited or recommended. When implementation adds one, record the resolved artifact/version/hash and transitive license report and include every required notice.

No third-party source is approved for vendoring. Use pinned package-manager artifacts or clearly separated tools unless a later ADR documents why vendoring is necessary.

## Optional research notebook environment

The `research` dependency group is used only for executing and plotting the
starter notebooks. Versions and transitive hashes are pinned in `uv.lock`;
all pre-existing dependency versions were retained. These tools are downloaded
by the user, not vendored into the source or Minecraft mod. No notebook server
or hosted service is started by the batch runner.

| Tool | Version | License and necessity |
| --- | --- | --- |
| [Matplotlib](https://pypi.org/project/matplotlib/3.10.8/) | 3.10.8 | Matplotlib license (PSF-based); numerical plots and PNG/SVG exports. |
| [nbformat](https://pypi.org/project/nbformat/5.10.4/) | 5.10.4 | BSD-3-Clause; standard notebook parsing and validation. |
| [nbclient](https://pypi.org/project/nbclient/0.10.4/) | 0.10.4 | BSD-3-Clause; actual cell execution with failure propagation. |
| [ipykernel](https://pypi.org/project/ipykernel/6.30.1/) | 6.30.1 | BSD-3-Clause; local Python kernel from the selected environment. |

These are established upstream projects; the pinned package metadata and project
links were checked on 2026-09-06. Research verification audits the installed group
for known dependency advisories and records its license inventory. Existing
exceptions in SECURITY.md remain explicit; a passing scan is not a guarantee
against unknown vulnerabilities. The dataset's copied anatomical fields retain
MaleCNS attribution and CC BY 4.0; generated model outputs and code are described
separately in [the dataset README](research/example-v1/README.md).

## Incorporated platform and Java/Fabric build tuple

The following tuple is implemented, locked, and exercised by automated builds/tests. Minecraft and Mojang materials are retrieved into ignored user/build caches under their terms; they are not incorporated into Git or approved for redistribution.

| Component | Exact identity | License posture | Implemented role | Current boundary |
|---|---|---|---|---|
| Minecraft Java Edition | `1.21.5` | Proprietary; Minecraft EULA and Usage Guidelines apply | Automated server/client/GameTest runtime plus one local headless development-server smoke | The user accepted the ignored local normal-server EULA and `LIVE-CIRCUIT-01` observed a headless development run. Production-JAR/manual-client observation remains open; never redistribute game files, worlds, account material, or EULA state. |
| Java platform | Java `21`; built-in `java.net.http.WebSocket` | JDK distribution terms apply | Mod runtime and bridge client | Built and exercised; avoids a third-party Java WebSocket library. Exact JDK distribution inventory remains part of the final reproduction receipt. |
| Fabric Loader | `0.16.10`; [`FabricMC/fabric-loader@fda9a7b84f0898f57b46fbbfda58795e7fa31cef`](https://github.com/FabricMC/fabric-loader/tree/fda9a7b84f0898f57b46fbbfda58795e7fa31cef) | Apache-2.0 | Mod loader and JUnit integration | Locked/checksum-verified; server/client tests exercised. |
| Fabric API | `0.128.2+1.21.5`; branch snapshot [`FabricMC/fabric-api@6a5e77458fab21477aaca2aa587230cb32c3cd32`](https://github.com/FabricMC/fabric-api/tree/6a5e77458fab21477aaca2aa587230cb32c3cd32) | Apache-2.0 | Registration, events, rendering, and GameTest APIs | Locked/checksum-verified; server/client tests exercised. |
| Fabric Loom | Maven artifact `net.fabricmc:fabric-loom:1.10.5` | MIT | Gradle build plugin | Locked and strict-verified in the current cache; fresh generated mapping archives require exact reviewed checksum refresh. |
| Gradle | Wrapper `8.12`; official distribution SHA-256 `7a00d51fb93147819aab76024feece20b6b84e420694101f276be952e08bef03` | Apache-2.0 distribution | Reproducible build launcher | Wrapper committed, distribution checksum verified, deterministic archive settings enabled. |
| Mojang mappings | `loom.officialMojangMappings()` | Mojang terms apply; not project-owned source | Development mappings | Retrieved through Loom; no decompiled Mojang source or generated proprietary artifacts are committed. |
| JUnit Jupiter | BOM `5.11.4` | EPL-2.0 | Pure-Java unit tests | Test-only, locked and checksum-verified. |
| Fabric example mod | 1.21.5 branch [`FabricMC/fabric-example-mod@4484e3f688a627d5c374cdd148cf6b135eaa34dc`](https://github.com/FabricMC/fabric-example-mod/tree/4484e3f688a627d5c374cdd148cf6b135eaa34dc) | CC0-1.0 | Historical configuration reference only | Not a runtime dependency; project identifiers, logic, and assets are original. |

## Superseded or rejected candidates, not incorporated

These records remain only to prevent an older report from silently reviving them.

| Candidate | Inspected revision | Disposition |
|---|---|---|
| Fabric API for Minecraft 1.21.1, artifact `0.116.15+1.21.1` | [`FabricMC/fabric-api@a4064d45e19e314f5a5407e496433fd03a073123`](https://github.com/FabricMC/fabric-api/tree/a4064d45e19e314f5a5407e496433fd03a073123) | **Superseded** by the selected 1.21.5 tuple. Must not appear in the implementation lock or build. |
| Fabric example mod 1.21.1 | [`FabricMC/fabric-example-mod@a4c6556aeab4eb100f9f0e3c11d44175384796e6`](https://github.com/FabricMC/fabric-example-mod/tree/a4c6556aeab4eb100f9f0e3c11d44175384796e6) | **Superseded** by the pinned 1.21.5 reference snapshot. |
| Java-WebSocket `v1.6.0` | [`TooTallNate/Java-WebSocket@da3cf2a777aed862f2f5b5cf060cae7969958667`](https://github.com/TooTallNate/Java-WebSocket/tree/da3cf2a777aed862f2f5b5cf060cae7969958667), MIT | **Rejected for v1.** Java 21's built-in `java.net.http.WebSocket` removes this runtime dependency. |
| gRPC Java `v1.82.4` | [`grpc/grpc-java@a144c66788dd42eb098b446ca505eebe73430197`](https://github.com/grpc/grpc-java/tree/a144c66788dd42eb098b446ca505eebe73430197), Apache-2.0 | **Rejected for v1.** It adds Protobuf and Netty surface without a demonstrated need. A later change requires an ADR and a fresh dependency review. |

## Conditional candidates, not incorporated

| Component | Immutable source inspected | SPDX license | Boundary before any use |
|---|---|---|---|
| `neuprint-python` | [`connectome-neuprint/neuprint-python@d89e624b445340f3deb0ace7998b219af57a75ad`](https://github.com/connectome-neuprint/neuprint-python/tree/d89e624b445340f3deb0ace7998b219af57a75ad), release line 0.6.3 | BSD-3-Clause | Optional offline audit only; canonical MaleCNS bulk files do not require a token. Pin package/hash, retain notices, and never use a live API in controller runtime. |
| CloudVolume | [`seung-lab/cloud-volume@aa15c3d02ab90440c137e52feaf95acd88fcfedc`](https://github.com/seung-lab/cloud-volume/tree/aa15c3d02ab90440c137e52feaf95acd88fcfedc) | BSD-3-Clause | Optional preprocessing only. Select a concrete release, pin hashes, retain notices, allow-list sources, and cap downloads. |
| `mc-runtime-test` | Release `4.5.1`; [`headlesshq/mc-runtime-test@fbafe02f89f4f13682e7b16dbcee6b6a43f4faca`](https://github.com/headlesshq/mc-runtime-test/tree/fbafe02f89f4f13682e7b16dbcee6b6a43f4faca) | MIT | Optional CI runner. Pin the action by commit, review permissions, retain notices, and prove minimum test count plus failure propagation. |
| HeadlessMC | Release `2.10.0`; [`headlesshq/headlessmc@d079294cd15fe6a6ed64ed6dc374b0ef073e0607`](https://github.com/headlesshq/headlessmc/tree/d079294cd15fe6a6ed64ed6dc374b0ef073e0607) | MIT | Optional launcher. Pin artifact/hash and do not redistribute or publicly cache game/account material. |

## Acquired and fallback connectome data

MaleCNS and its extracted fixture are data inputs, not software dependencies. Their provenance never implies a biological or behavioral result.

| Target | License / attribution | Current evidence state | Gate before use or redistribution |
|---|---|---|---|
| Janelia MaleCNS `male-cns:v1.0`, released 2026-06-08 | Official release and committed manifest record CC BY 4.0 plus the Berg et al. preprint citation | Three canonical generation-qualified Feather objects acquired, hashed, parsed, and retained only in the ignored cache | `ACQ-MCNS-01`; bulk objects remain excluded from Git; every future derivative needs a distinct lineage/rights receipt. |
| MaleCNS flight-power fixture `mcns-v1.0-flight-power-v1` | CC BY 4.0 derived data with embedded attribution, license link, citation, and change notice | **Incorporated:** 14 source neurons, 22 source-row-traceable edges; two fresh-cache builds byte-identical; deterministic fixture dynamics and automated bridge candidate exercised | `FIXTURE-MCNS-01`; approved for commit with notices; use does not imply full-MaleCNS execution, physiology, G3 promotion, or G4. |
| MaleCNS flight-steering fixture `mcns-v1.0-flight-steering-v1` | CC BY 4.0 derived data with embedded attribution, license link, citation, and change notice | **Incorporated:** 128 source neurons, 887 source-row-traceable edges; the graph behind the historical trained-readout result (ADR 0038) | `FIXTURE-MCNS-02`; approved for commit with notices; use does not imply full-MaleCNS execution. |
| Full artifact `male-cns-v1.0-superclass-v1` | CC BY 4.0-derived local data under the source attribution/citation | **Ignored local derivative:** 166,700 nodes, 25,582,938 directed edges; two byte-identical builds | `RUNTIME-MCNS-01`; G1 only for manifest `b33bdda8…f67c98c`; not committed or approved for redistribution by that receipt. |
| BANC v888 / paper-version v2 edgelist | Reported CC BY 4.0; exact file terms still require manifest verification | Fallback candidate only; no data acquired or incorporated | Requires a separate decision, exact materialization/file hashes, manifest, and non-merging rule before use. |

## Optional components requiring a new decision

| Component | Inspected source | License | Status / boundary |
|---|---|---|---|
| GeckoLib | [`bernie-g/geckolib@05048071ce4801969237ffdca0b73223e986d94d`](https://github.com/bernie-g/geckolib/tree/05048071ce4801969237ffdca0b73223e986d94d) | MIT | Deferred. Start with native Fabric rendering. Select and audit an exact Minecraft-1.21.5-compatible artifact before use. |
| `itzg/docker-minecraft-server` | [`itzg/docker-minecraft-server@f5d94230e8c13fe42901f30c84830b313ecba233`](https://github.com/itzg/docker-minecraft-server/tree/f5d94230e8c13fe42901f30c84830b313ecba233) | Apache-2.0 | Optional dedicated-server smoke environment. Pin image digest; user must explicitly accept Minecraft EULA; disposable world and loopback ports. |
| Mineflayer | [`PrismarineJS/mineflayer@d0189a88cc8b4adf56e18cdf1a635da74482c3ba`](https://github.com/PrismarineJS/mineflayer/tree/d0189a88cc8b4adf56e18cdf1a635da74482c3ba) | MIT | Optional external test observer or visibly labeled scripted baseline only. Never a Faithful Mode policy. Use an offline local identity, not stored Microsoft credentials. |
| ReplayMod | [`ReplayMod/ReplayMod@4361ec953a53d5bf7f1bf663f6a406273d44a146`](https://github.com/ReplayMod/ReplayMod/tree/4361ec953a53d5bf7f1bf663f6a406273d44a146) | GPL-3.0 | Optional separately installed demo recorder only. Do not link, bundle, copy, or treat replay as scientific telemetry. Sanitize recordings before sharing. |
| Neuroglancer | [`google/neuroglancer@c0f79b10d04647c6fff6ff030cb2dc4054526cf4`](https://github.com/google/neuroglancer/tree/c0f79b10d04647c6fff6ff030cb2dc4054526cf4) | Apache-2.0 | Optional external connectome viewer; not part of the first dashboard or control loop. Reassess browser/data-source privacy before deployment. |
| FlyGym / NeuroMechFly v2 | [`NeLy-EPFL/flygym@38c8ec61034cd59bc5ba0de20688d4a3c0000d60`](https://github.com/NeLy-EPFL/flygym/tree/38c8ec61034cd59bc5ba0de20688d4a3c0000d60) | Apache-2.0 | Optional separate offline comparison environment. It is not Minecraft physics and must not be bundled merely to support “biomechanical” wording. |

## Reference-only prior art

The following repositories informed architecture/research but are not approved runtime dependencies or copied sources:

| Project | Inspected revision | License / reason for reference-only status |
|---|---|---|
| Drosophila whole-brain LIF model | [`philshiu/Drosophila_brain_model@91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960) | MIT code; research notebook, older maintenance, data rights separate. Cite and independently implement disclosed equations/tests. |
| Neurokernel | [`neurokernel/neurokernel@066320ea8314c4c7bb583ac35434181784fbd29d`](https://github.com/neurokernel/neurokernel/tree/066320ea8314c4c7bb583ac35434181784fbd29d) | BSD-3-Clause text; CUDA/MPI architecture too heavy for initial runtime. |
| FlyBrainLab umbrella repository | [`FlyBrainLab/FlyBrainLab@5dd079afd0ac0a39659082d5ac95c330d0fa6bde`](https://github.com/FlyBrainLab/FlyBrainLab/tree/5dd079afd0ac0a39659082d5ac95c330d0fa6bde) | BSD-3-Clause; multi-service installer/security surface unsuitable as a dependency. |
| Legacy NeuroMechFly | [`NeLy-EPFL/NeuroMechFly@6fdc6212bcf5b71fcd47bca119876c92092c2cd6`](https://github.com/NeLy-EPFL/NeuroMechFly/tree/6fdc6212bcf5b71fcd47bca119876c92092c2cd6) | Apache-2.0; archived and superseded by FlyGym. |
| Project Malmo | [`microsoft/malmo@2af08005411e78c79a44528e9da2f17a167e1f6b`](https://github.com/microsoft/malmo/tree/2af08005411e78c79a44528e9da2f17a167e1f6b) | MIT; last release 2018 and obsolete Minecraft/build stack. |
| NeoForge | [`neoforged/NeoForge@9a147e8c009abdc8baf8ab7e4a60adf4f42b6955`](https://github.com/neoforged/NeoForge/tree/9a147e8c009abdc8baf8ab7e4a60adf4f42b6955) | LGPL-2.1; technically viable but redundant with selected Fabric target. |
| Paper | [`PaperMC/Paper@c9e894d3cc03f21f80de4f4061a795e11941e89a`](https://github.com/PaperMC/Paper/tree/c9e894d3cc03f21f80de4f4061a795e11941e89a) | Primarily GPL-3.0 with named MIT exceptions; server plugin cannot alone provide required custom client rendering/overlay. |
| MockBukkit | [`MockBukkit/MockBukkit@a5144c01e84a906130220fc0ad646e55b5783c4d`](https://github.com/MockBukkit/MockBukkit/tree/a5144c01e84a906130220fc0ad646e55b5783c4d) | MIT; Paper/Bukkit-only mocks, not applicable to Fabric or physics/rendering verification. |

### Copyleft analyst tools

`navis-org/fafbseg-py@d0da95123ee606e204ae2c702e7bc78538646fbd` and `navis-org/navis@ae9f0422eb4aab8045803f1ba93dcc8ed684ab18` are GPL-3.0. They may be used only as clearly separated analyst tools unless a later licensing decision establishes that importing/distributing them with the project is acceptable. Do not copy their source into this repository. Neutral artifacts exported from them still require independent dataset provenance and redistribution review.

## Data and model licenses are separate

Software permission never grants permission to redistribute connectome tables, neuron meshes, annotations, model assets, or paper supplements.

- `flyconnectome/flywire_annotations` was inspected at release `v3.1.0`, commit [`8587524c1748ce5ef2080822a2fc890fc03bf597`](https://github.com/flyconnectome/flywire_annotations/tree/8587524c1748ce5ef2080822a2fc890fc03bf597). No repository license was detected. **Do not copy it into the repository** until the exact file terms and citation/redistribution requirements are documented.
- The connectivity and completeness files in the MIT-licensed `Drosophila_brain_model` repository must not be assumed MIT-licensed merely because the surrounding code is. Resolve their origin and terms independently.
- Any accepted connectome must have a manifest containing source URL, access date, release/materialization identifier, file size, cryptographic hash, scope, citation, license/terms URL, redistribution status, and transformation lineage.
- CI should use a small original synthetic network or an explicitly redistributable fixture. Bulk or credential-gated data belongs in ignored local caches.

## Minecraft and Mojang/Microsoft material

Minecraft is proprietary and is not a third-party open-source dependency that can be redistributed under the licenses above. Follow the current official [Minecraft Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines) and applicable EULA.

Do not commit or publish:

- Minecraft client/server jars or a preassembled modded game;
- decompiled or copied Mojang source;
- Mojang textures, models, sounds, fonts, or branding as project assets;
- account refresh/access tokens, launcher profiles, cookies, player data, or private worlds;
- CI caches containing proprietary game files in public storage.

Distribute only original mod code/assets and reproducible instructions that obtain a legitimate game installation at build/run time. Any public-facing material must make clear that NeuroCraft Fly is not an official Minecraft product and is not approved by or associated with Mojang or Microsoft.

## Historical primitive-only Faithful Mode exclusion list

This list applies to the earlier primitive-only controller lane. The current
interactive profile uses the declared scripted-program interface described in
[MODEL_CARD.md](docs/MODEL_CARD.md) and [ARCHITECTURE.md](docs/ARCHITECTURE.md).
For the historical primitive-only lane, the following may not be linked, called,
or consulted to select movement:

- Baritone (`cabaletta/baritone@ad627c83cc0ac6059d7a1b29eca4e54214584b5e`, LGPL-3.0);
- Mineflayer pathfinding, behavior-tree, state-machine, or autonomous-control plugins;
- vanilla Minecraft mob goals/navigation or another mod's mob AI;
- Malmo policies, command scripts, an LLM, or a learned conventional policy hidden behind the transport;
- replayed action scripts except in an explicitly labeled scripted-baseline mode.

Physics, collision handling, command clamping, connection timeouts, and a fail-closed paused state are permitted only when documented as safety/environment constraints and tested not to choose goals or actions.

## Maintainer checklist for every addition

Before merging a third-party addition:

1. Record package name, exact version, immutable source revision, artifact URL, SHA-256, direct/transitive status, SPDX license, copyright/NOTICE path, purpose, and removal path.
2. Verify the license text from the exact artifact, not only a registry badge. Escalate missing/custom/conflicting licenses.
3. Produce/update the language lockfile and SBOM; review transitive repositories and package-install scripts.
4. Scan the resolved artifact for known vulnerabilities and document unresolved findings or compensating controls. “No finding” is not a guarantee of security.
5. Test on the frozen Java/Python/Minecraft/OS tuple. For a mod dependency, test both dedicated server and client to catch client-class loading failures.
6. Bind services to loopback, authenticate sessions, cap messages/downloads/caches, redact credentials, and fail closed.
7. Add required license and NOTICE files to the distributed artifact. Preserve source offers or relinking/installation rights if a future LGPL/GPL component is intentionally distributed.
8. For datasets/assets, complete the separate provenance and redistribution manifest.
9. Update this file from “candidate” to “incorporated” only after the resolved artifact exists in a manifest/lockfile and verification evidence is recorded.
