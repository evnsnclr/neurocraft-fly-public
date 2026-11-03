# JVM third-party notices (build-time inventory)

This inventory covers the **entire Gradle build graph** of the Fabric mod: the Minecraft/Fabric runtime the game itself loads, the Loom plugin classpath, mappings and decompilers, and the JUnit/GameTest classpaths.

**Nothing listed here is redistributed.** The project mod JAR bundles zero third-party components: `mod/build.gradle` declares no `include`/shadow dependency and Loom's `includeInternal` bundling configuration resolves empty in `mod/gradle.lockfile`. Every component below is compiled against, remapped, run in a developer/test JVM, or fetched by the user's own Gradle and Minecraft caches under their own terms.

Regenerate with:

```bash
cd mod && ./gradlew --no-daemon --dependency-verification strict \
    emitJvmDependencyGraph
python3 scripts/build_jvm_inventory.py \
    --graph mod/build/reports/jvm-dependency-graph.json \
    --verification mod/gradle/verification-metadata.xml \
    --lockfile mod/gradle.lockfile \
    --overrides docs/compliance/jvm-license-overrides.json \
    --sbom artifacts/sbom/jvm.cdx.json \
    --notices docs/compliance/JVM_THIRD_PARTY_NOTICES.md
```

SHA-256 values in the machine-readable SBOM come only from the tracked `mod/gradle/verification-metadata.xml`; this tool never downloads anything. A purl carries a `repository_url` qualifier only for groups that are absent from Maven Central (`com.mojang` on libraries.minecraft.net, `net.fabricmc*` and `fabric-loom` on maven.fabricmc.net).

## Build tuple

| Pin | Value |
| --- | --- |
| Minecraft | `1.21.5` |
| Fabric Loader | `0.16.10` |
| Fabric API | `0.128.2+1.21.5` |
| Fabric Loom | `1.10.5` |
| Locked coordinates | 213 |
| Resolved coordinates (platform-normalized) | 235 |
| Logical components | 164 |

Loom folding accounts for the difference: byte-remapped Fabric API copies fold into their upstream modules and the two split Minecraft JARs fold into one Minecraft component.

## Components by scope

| Scope | Components | Meaning |
| --- | --- | --- |
| `build-plugin` | 20 | Gradle plugin classpath (Loom and its build-time libraries) |
| `build-tooling` | 7 | mappings, decompilers, and annotation processors |
| `dev-runtime` | 2 | Loom development launch environment only |
| `game-runtime` | 71 | loaded by the user's own Minecraft/Fabric runtime |
| `mod-api` | 52 | compiled against and remapped by Loom (Fabric Loader and Fabric API) |
| `test` | 12 | JUnit and GameTest classpaths only |
| `bundled` | 0 | embedded in the distributed mod JAR (must be empty) |

## Components by license

| License | Components |
| --- | --- |
| `Apache-2.0` | 109 |
| `BSD-3-Clause` | 17 |
| `EPL-2.0` | 9 |
| `LGPL-2.0-or-later` | 1 |
| `LGPL-3.0-or-later` | 1 |
| `LicenseRef-Minecraft-EULA` | 1 |
| `LicenseRef-Mojang-Mappings` | 1 |
| `LicenseRef-Mojang-Proprietary` | 6 |
| `MIT` | 18 |
| `Unicode-3.0` | 1 |

## Component inventory

### Apache-2.0

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `ca.weblite:java-objc-bridge` | `1.1` | `game-runtime` | [https://github.com/shannah/Java-Objective-C-Bridge/blob/master/LICENSE](https://github.com/shannah/Java-Objective-C-Bridge/blob/master/LICENSE) | concluded |
| `com.fasterxml.jackson.core:jackson-annotations` | `2.13.4` | `game-runtime` | [http://github.com/FasterXML/jackson](http://github.com/FasterXML/jackson) | declared |
| `com.fasterxml.jackson.core:jackson-core` | `2.13.4` | `game-runtime` | [https://github.com/FasterXML/jackson-core](https://github.com/FasterXML/jackson-core) | declared |
| `com.fasterxml.jackson.core:jackson-databind` | `2.13.4.2` | `game-runtime` | [http://github.com/FasterXML/jackson](http://github.com/FasterXML/jackson) | declared |
| `com.github.stephenc.jcip:jcip-annotations` | `1.0-1` | `game-runtime` | [http://stephenc.github.com/jcip-annotations](http://stephenc.github.com/jcip-annotations) | declared |
| `com.google.code.gson:gson` | `2.10.1` | `build-plugin` | — | declared |
| `com.google.code.gson:gson` | `2.11.0` | `game-runtime` | — | declared |
| `com.google.errorprone:error_prone_annotations` | `2.50.0` | `build-plugin` | — | declared |
| `com.google.guava:failureaccess` | `1.0.3` | `game-runtime` | — | declared |
| `com.google.guava:guava` | `33.7.2-jre` | `game-runtime` | [https://github.com/google/guava/blob/v33.7.2/LICENSE](https://github.com/google/guava/blob/v33.7.2/LICENSE) | concluded |
| `com.google.guava:listenablefuture` | `9999.0-empty-to-avoid-conflict-with-guava` | `build-plugin` | — | declared |
| `com.google.j2objc:j2objc-annotations` | `3.1` | `build-plugin` | [https://github.com/google/j2objc/](https://github.com/google/j2objc/) | declared |
| `com.nimbusds:content-type` | `2.3` | `game-runtime` | [https://bitbucket.org/connect2id/nimbus-content-type](https://bitbucket.org/connect2id/nimbus-content-type) | declared |
| `com.nimbusds:lang-tag` | `1.7` | `game-runtime` | [https://bitbucket.org/connect2id/nimbus-language-tags](https://bitbucket.org/connect2id/nimbus-language-tags) | declared |
| `com.nimbusds:nimbus-jose-jwt` | `9.40` | `game-runtime` | [https://bitbucket.org/connect2id/nimbus-jose-jwt](https://bitbucket.org/connect2id/nimbus-jose-jwt) | declared |
| `com.nimbusds:oauth2-oidc-sdk` | `11.18` | `game-runtime` | [https://bitbucket.org/connect2id/oauth-2.0-sdk-with-openid-connect-extensions](https://bitbucket.org/connect2id/oauth-2.0-sdk-with-openid-connect-extensions) | declared |
| `commons-codec:commons-codec` | `1.17.1` | `game-runtime` | [https://commons.apache.org/proper/commons-codec/](https://commons.apache.org/proper/commons-codec/) | declared |
| `commons-io:commons-io` | `2.15.1` | `build-plugin` | [https://commons.apache.org/proper/commons-io/](https://commons.apache.org/proper/commons-io/) | declared |
| `commons-io:commons-io` | `2.17.0` | `game-runtime` | [https://commons.apache.org/proper/commons-io/](https://commons.apache.org/proper/commons-io/) | declared |
| `commons-logging:commons-logging` | `1.3.4` | `game-runtime` | [https://commons.apache.org/proper/commons-logging/](https://commons.apache.org/proper/commons-logging/) | declared |
| `io.netty:netty-buffer` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-codec` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-common` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-handler` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-resolver` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-transport-classes-epoll` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-transport-native-epoll` | `4.1.118.Final` | `game-runtime` | [https://github.com/netty/netty/blob/4.1/LICENSE.txt](https://github.com/netty/netty/blob/4.1/LICENSE.txt) | concluded |
| `io.netty:netty-transport-native-unix-common` | `4.1.118.Final` | `game-runtime` | — | declared |
| `io.netty:netty-transport` | `4.1.118.Final` | `game-runtime` | — | declared |
| `it.unimi.dsi:fastutil` | `8.5.15` | `game-runtime` | [http://fastutil.di.unimi.it/](http://fastutil.di.unimi.it/) | declared |
| `net.fabricmc.fabric-api:fabric-api-base` | `0.4.62+73a52b4b49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-api-base](https://github.com/FabricMC/fabric/tree/HEAD/fabric-api-base) | declared |
| `net.fabricmc.fabric-api:fabric-api-deprecated` | `0.128.2+1.21.5` | `mod-api` | [https://github.com/FabricMC/fabric-api/blob/26.2/LICENSE](https://github.com/FabricMC/fabric-api/blob/26.2/LICENSE) | concluded |
| `net.fabricmc.fabric-api:fabric-api-lookup-api-v1` | `1.6.96+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-api-lookup-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-api-lookup-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-api` | `0.128.2+1.21.5` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/](https://github.com/FabricMC/fabric/tree/HEAD/) | declared |
| `net.fabricmc.fabric-api:fabric-biome-api-v1` | `16.0.7+2dd063df49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-biome-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-biome-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-block-api-v1` | `1.1.0+ed91556f49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-block-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-block-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-block-view-api-v2` | `1.0.26+aa6d566c49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-block-view-api-v2](https://github.com/FabricMC/fabric/tree/HEAD/fabric-block-view-api-v2) | declared |
| `net.fabricmc.fabric-api:fabric-blockrenderlayer-v1` | `2.0.16+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-blockrenderlayer-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-blockrenderlayer-v1) | declared |
| `net.fabricmc.fabric-api:fabric-client-gametest-api-v1` | `4.2.0+dac2d6e349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-client-gametest-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-client-gametest-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-client-tags-api-v1` | `1.1.37+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-client-tags-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-client-tags-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-command-api-v1` | `1.2.70+f71b366f49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-command-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-command-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-command-api-v2` | `2.2.49+73a52b4b49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-command-api-v2](https://github.com/FabricMC/fabric/tree/HEAD/fabric-command-api-v2) | declared |
| `net.fabricmc.fabric-api:fabric-commands-v0` | `0.2.87+df3654b349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-commands-v0](https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-commands-v0) | declared |
| `net.fabricmc.fabric-api:fabric-content-registries-v0` | `10.0.14+3e6c1f7d49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-content-registries-v0](https://github.com/FabricMC/fabric/tree/HEAD/fabric-content-registries-v0) | declared |
| `net.fabricmc.fabric-api:fabric-convention-tags-v1` | `2.1.33+7f945d5b49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-convention-tags-v1](https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-convention-tags-v1) | declared |
| `net.fabricmc.fabric-api:fabric-convention-tags-v2` | `2.15.2+6d9989f349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-convention-tags-v2](https://github.com/FabricMC/fabric/tree/HEAD/fabric-convention-tags-v2) | declared |
| `net.fabricmc.fabric-api:fabric-crash-report-info-v1` | `0.3.12+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-crash-report-info-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-crash-report-info-v1) | declared |
| `net.fabricmc.fabric-api:fabric-data-attachment-api-v1` | `1.8.5+6d9989f349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-data-attachment-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-data-attachment-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-data-generation-api-v1` | `22.5.2+a0fab30c49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-data-generation-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-data-generation-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-dimensions-v1` | `4.0.17+3e6c1f7d49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-dimensions-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-dimensions-v1) | declared |
| `net.fabricmc.fabric-api:fabric-entity-events-v1` | `2.1.0+3ce7866349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-entity-events-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-entity-events-v1) | declared |
| `net.fabricmc.fabric-api:fabric-events-interaction-v0` | `4.0.15+64e3057949` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-events-interaction-v0](https://github.com/FabricMC/fabric/tree/HEAD/fabric-events-interaction-v0) | declared |
| `net.fabricmc.fabric-api:fabric-game-rule-api-v1` | `1.0.70+c327076a49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-game-rule-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-game-rule-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-gametest-api-v1` | `3.1.3+2a6ec84b49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-gametest-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-gametest-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-item-api-v1` | `11.5.0+0620fe4749` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-item-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-item-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-item-group-api-v1` | `4.2.9+3459fc6149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-item-group-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-item-group-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-key-binding-api-v1` | `1.0.63+ecf51cdc49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-key-binding-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-key-binding-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-keybindings-v0` | `0.2.61+df3654b349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-keybindings-v0](https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-keybindings-v0) | declared |
| `net.fabricmc.fabric-api:fabric-lifecycle-events-v1` | `2.6.0+230071a049` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-lifecycle-events-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-lifecycle-events-v1) | declared |
| `net.fabricmc.fabric-api:fabric-loot-api-v2` | `3.0.48+3f89f5a549` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-loot-api-v2](https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-loot-api-v2) | declared |
| `net.fabricmc.fabric-api:fabric-loot-api-v3` | `1.0.36+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-loot-api-v3](https://github.com/FabricMC/fabric/tree/HEAD/fabric-loot-api-v3) | declared |
| `net.fabricmc.fabric-api:fabric-message-api-v1` | `6.1.0+fe971bba49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-message-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-message-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-model-loading-api-v1` | `5.2.3+5281b90749` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-model-loading-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-model-loading-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-networking-api-v1` | `4.5.0+775be32c49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-networking-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-networking-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-object-builder-api-v1` | `21.1.2+b8d6ba7049` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-object-builder-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-object-builder-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-particles-v1` | `4.1.2+112e550e49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-particles-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-particles-v1) | declared |
| `net.fabricmc.fabric-api:fabric-recipe-api-v1` | `8.1.8+3235ab3249` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-recipe-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-recipe-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-registry-sync-v0` | `6.1.22+cd64004349` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-registry-sync-v0](https://github.com/FabricMC/fabric/tree/HEAD/fabric-registry-sync-v0) | declared |
| `net.fabricmc.fabric-api:fabric-renderer-api-v1` | `6.1.2+bdf4842249` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-renderer-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-renderer-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-renderer-indigo` | `3.1.2+5281b90749` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-renderer-indigo](https://github.com/FabricMC/fabric/tree/HEAD/fabric-renderer-indigo) | declared |
| `net.fabricmc.fabric-api:fabric-rendering-data-attachment-v1` | `0.3.64+73761d2e49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-rendering-data-attachment-v1](https://github.com/FabricMC/fabric/tree/HEAD/deprecated/fabric-rendering-data-attachment-v1) | declared |
| `net.fabricmc.fabric-api:fabric-rendering-fluids-v1` | `3.1.27+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-rendering-fluids-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-rendering-fluids-v1) | declared |
| `net.fabricmc.fabric-api:fabric-rendering-v1` | `11.3.0+a272b33849` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-rendering-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-rendering-v1) | declared |
| `net.fabricmc.fabric-api:fabric-resource-conditions-api-v1` | `5.0.21+73a52b4b49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-resource-conditions-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-resource-conditions-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-resource-loader-v0` | `3.1.7+847e5f5c49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-resource-loader-v0](https://github.com/FabricMC/fabric/tree/HEAD/fabric-resource-loader-v0) | declared |
| `net.fabricmc.fabric-api:fabric-screen-api-v1` | `2.0.46+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-screen-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-screen-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-screen-handler-api-v1` | `1.3.129+c327076a49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-screen-handler-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-screen-handler-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-sound-api-v1` | `1.0.38+86c3a9f149` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-sound-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-sound-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-tag-api-v1` | `1.0.17+ecf51cdc49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-tag-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-tag-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-transfer-api-v1` | `5.4.24+7b20cbb049` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-transfer-api-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-transfer-api-v1) | declared |
| `net.fabricmc.fabric-api:fabric-transitive-access-wideners-v1` | `6.3.17+f17a180c49` | `mod-api` | [https://github.com/FabricMC/fabric/tree/HEAD/fabric-transitive-access-wideners-v1](https://github.com/FabricMC/fabric/tree/HEAD/fabric-transitive-access-wideners-v1) | declared |
| `net.fabricmc:access-widener` | `2.1.0` | `build-plugin` | [https://github.com/FabricMC/access-widener](https://github.com/FabricMC/access-widener) | declared |
| `net.fabricmc:dev-launch-injector` | `0.2.1+build.8` | `game-runtime` | [https://github.com/FabricMC/dev-launch-injector/blob/master/LICENSE](https://github.com/FabricMC/dev-launch-injector/blob/master/LICENSE) | concluded |
| `net.fabricmc:fabric-fernflower` | `2.0.0` | `build-tooling` | [https://github.com/FabricMC/intellij-fernflower/blob/master/README.md](https://github.com/FabricMC/intellij-fernflower/blob/master/README.md) | concluded |
| `net.fabricmc:fabric-loader-junit` | `0.16.10` | `test` | [https://github.com/FabricMC/fabric-loader/blob/master/LICENSE](https://github.com/FabricMC/fabric-loader/blob/master/LICENSE) | concluded |
| `net.fabricmc:fabric-loader` | `0.16.10` | `mod-api` | [https://github.com/FabricMC/fabric-loader/blob/master/LICENSE](https://github.com/FabricMC/fabric-loader/blob/master/LICENSE) | concluded |
| `net.fabricmc:mapping-io` | `0.2.1` | `build-tooling` | [https://github.com/FabricMC/mapping-io/blob/dev/LICENSE](https://github.com/FabricMC/mapping-io/blob/dev/LICENSE) | concluded |
| `net.fabricmc:mapping-io` | `0.7.1` | `build-plugin` | [https://github.com/FabricMC/mapping-io](https://github.com/FabricMC/mapping-io) | declared |
| `net.fabricmc:stitch` | `0.6.2` | `build-plugin` | [https://github.com/FabricMC/stitch/blob/master/LICENSE](https://github.com/FabricMC/stitch/blob/master/LICENSE) | concluded |
| `net.fabricmc:tiny-mappings-parser` | `0.3.0+build.17` | `build-plugin` | [https://github.com/FabricMC/tiny-mappings-parser/blob/master/LICENSE](https://github.com/FabricMC/tiny-mappings-parser/blob/master/LICENSE) | concluded |
| `net.java.dev.jna:jna-platform` | `5.15.0` | `game-runtime` | [https://github.com/java-native-access/jna](https://github.com/java-native-access/jna) | concluded |
| `net.java.dev.jna:jna` | `5.15.0` | `game-runtime` | [https://github.com/java-native-access/jna](https://github.com/java-native-access/jna) | concluded |
| `net.minidev:accessors-smart` | `2.5.1` | `game-runtime` | [https://urielch.github.io/](https://urielch.github.io/) | declared |
| `net.minidev:json-smart` | `2.5.1` | `game-runtime` | [https://urielch.github.io/](https://urielch.github.io/) | declared |
| `org.apache.commons:commons-compress` | `1.27.1` | `game-runtime` | [https://commons.apache.org/proper/commons-compress/](https://commons.apache.org/proper/commons-compress/) | declared |
| `org.apache.commons:commons-lang3` | `3.17.0` | `game-runtime` | [https://commons.apache.org/proper/commons-lang/](https://commons.apache.org/proper/commons-lang/) | declared |
| `org.apache.httpcomponents:httpclient` | `4.5.14` | `game-runtime` | [http://hc.apache.org/httpcomponents-client-ga](http://hc.apache.org/httpcomponents-client-ga) | declared |
| `org.apache.httpcomponents:httpcore` | `4.4.16` | `game-runtime` | [http://hc.apache.org/httpcomponents-core-ga](http://hc.apache.org/httpcomponents-core-ga) | declared |
| `org.apache.logging.log4j:log4j-api` | `2.25.5` | `game-runtime` | [https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-api/2.25.5/log4j-api-2.25.5.jar](https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-api/2.25.5/log4j-api-2.25.5.jar) | concluded |
| `org.apache.logging.log4j:log4j-core` | `2.25.5` | `game-runtime` | [https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-core/2.25.5/log4j-core-2.25.5.jar](https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-core/2.25.5/log4j-core-2.25.5.jar) | concluded |
| `org.apache.logging.log4j:log4j-slf4j2-impl` | `2.25.5` | `game-runtime` | [https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-slf4j2-impl/2.25.5/log4j-slf4j2-impl-2.25.5.jar](https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-slf4j2-impl/2.25.5/log4j-slf4j2-impl-2.25.5.jar) | concluded |
| `org.apiguardian:apiguardian-api` | `1.1.2` | `test` | [https://github.com/apiguardian-team/apiguardian/blob/main/LICENSE](https://github.com/apiguardian-team/apiguardian/blob/main/LICENSE) | concluded |
| `org.jetbrains.kotlin:kotlin-metadata-jvm` | `2.0.21` | `build-plugin` | [https://kotlinlang.org/](https://kotlinlang.org/) | declared |
| `org.jetbrains:annotations` | `20.1.0` | `build-tooling` | [https://github.com/JetBrains/java-annotations](https://github.com/JetBrains/java-annotations) | declared |
| `org.jetbrains:annotations` | `25.0.0` | `test` | [https://github.com/JetBrains/java-annotations/blob/master/LICENSE.txt](https://github.com/JetBrains/java-annotations/blob/master/LICENSE.txt) | concluded |
| `org.jspecify:jspecify` | `1.0.1` | `build-plugin` | [https://github.com/jspecify/jspecify/blob/v1.0.1/LICENSE](https://github.com/jspecify/jspecify/blob/v1.0.1/LICENSE) | concluded |
| `org.lz4:lz4-java` | `1.8.0` | `game-runtime` | [https://github.com/lz4/lz4-java](https://github.com/lz4/lz4-java) | declared |
| `org.opentest4j:opentest4j` | `1.3.0` | `test` | [https://github.com/ota4j-team/opentest4j/blob/main/LICENSE](https://github.com/ota4j-team/opentest4j/blob/main/LICENSE) | concluded |
| `org.vineflower:vineflower` | `1.11.1` | `build-tooling` | [https://vineflower.org](https://vineflower.org) | declared |

### BSD-3-Clause

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `org.cadixdev:bombe` | `0.3.4` | `build-plugin` | [https://www.jamiemansfield.me/projects/bombe](https://www.jamiemansfield.me/projects/bombe) | declared |
| `org.jline:jline-reader` | `3.20.0` | `dev-runtime` | [https://github.com/jline/jline3/blob/master/LICENSE.txt](https://github.com/jline/jline3/blob/master/LICENSE.txt) | concluded |
| `org.jline:jline-terminal` | `3.20.0` | `dev-runtime` | [https://github.com/jline/jline3/blob/master/LICENSE.txt](https://github.com/jline/jline3/blob/master/LICENSE.txt) | concluded |
| `org.lwjgl:lwjgl-freetype` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl-glfw` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl-jemalloc` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl-openal` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl-opengl` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl-stb` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl-tinyfd` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.lwjgl:lwjgl` | `3.3.3` | `game-runtime` | [https://www.lwjgl.org](https://www.lwjgl.org) | declared |
| `org.ow2.asm:asm-analysis` | `9.7.1` | `game-runtime` | [http://asm.ow2.io/](http://asm.ow2.io/) | declared |
| `org.ow2.asm:asm-commons` | `9.7.1` | `game-runtime` | [http://asm.ow2.io/](http://asm.ow2.io/) | declared |
| `org.ow2.asm:asm-tree` | `9.7.1` | `game-runtime` | [http://asm.ow2.io/](http://asm.ow2.io/) | declared |
| `org.ow2.asm:asm-util` | `9.7.1` | `game-runtime` | [http://asm.ow2.io/](http://asm.ow2.io/) | declared |
| `org.ow2.asm:asm` | `9.6` | `game-runtime` | [http://asm.ow2.io/](http://asm.ow2.io/) | declared |
| `org.ow2.asm:asm` | `9.7.1` | `game-runtime` | [http://asm.ow2.io/](http://asm.ow2.io/) | declared |

### EPL-2.0

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `net.fabricmc:mercury` | `0.4.2` | `build-plugin` | [https://github.com/FabricMC/Mercury](https://github.com/FabricMC/Mercury) | declared |
| `org.junit.jupiter:junit-jupiter-api` | `5.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit.jupiter:junit-jupiter-engine` | `5.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit.jupiter:junit-jupiter-params` | `5.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit.jupiter:junit-jupiter` | `5.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit.platform:junit-platform-commons` | `1.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit.platform:junit-platform-engine` | `1.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit.platform:junit-platform-launcher` | `1.11.4` | `test` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) | concluded |
| `org.junit:junit-bom` | `5.11.4` | `test` | [https://junit.org/junit5/](https://junit.org/junit5/) | declared |

### LGPL-2.0-or-later

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `org.jcraft:jorbis` | `0.0.17` | `game-runtime` | [http://www.jcraft.com/jorbis](http://www.jcraft.com/jorbis) | concluded |

### LGPL-3.0-or-later

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `net.fabricmc:tiny-remapper` | `0.11.1` | `build-plugin` | [https://github.com/FabricMC/tiny-remapper/blob/master/LICENSE](https://github.com/FabricMC/tiny-remapper/blob/master/LICENSE) | concluded |

### LicenseRef-Minecraft-EULA — Minecraft End User License Agreement (proprietary, not redistributable)

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `net.minecraft:minecraft` | `1.21.5` | `game-runtime` | [https://www.minecraft.net/en-us/eula](https://www.minecraft.net/en-us/eula) | concluded |

### LicenseRef-Mojang-Mappings — Mojang official obfuscation-mapping terms (not redistributable)

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `net.minecraft:mojang-mappings` | `1.21.5` | `build-tooling` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |

### LicenseRef-Mojang-Proprietary — Mojang-published library with no declared license (Minecraft EULA and Usage Guidelines apply)

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `com.mojang:authlib` | `6.0.58` | `game-runtime` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |
| `com.mojang:blocklist` | `1.0.10` | `game-runtime` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |
| `com.mojang:jtracy` | `1.0.29` | `game-runtime` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |
| `com.mojang:logging` | `1.5.10` | `game-runtime` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |
| `com.mojang:patchy` | `2.2.10` | `game-runtime` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |
| `com.mojang:text2speech` | `1.18.11` | `game-runtime` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) | concluded |

### MIT

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `com.github.oshi:oshi-core` | `6.6.5` | `game-runtime` | — | declared |
| `com.microsoft.azure:msal4j` | `1.17.2` | `game-runtime` | [https://github.com/AzureAD/microsoft-authentication-library-for-java](https://github.com/AzureAD/microsoft-authentication-library-for-java) | declared |
| `com.mojang:brigadier` | `1.3.10` | `game-runtime` | [https://github.com/Mojang/brigadier/blob/master/LICENSE](https://github.com/Mojang/brigadier/blob/master/LICENSE) | concluded |
| `com.mojang:datafixerupper` | `8.0.16` | `game-runtime` | [https://github.com/Mojang/DataFixerUpper/blob/master/LICENSE](https://github.com/Mojang/DataFixerUpper/blob/master/LICENSE) | concluded |
| `fabric-loom:fabric-loom.gradle.plugin` | `1.10.5` | `build-plugin` | [https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE](https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE) | concluded |
| `io.github.llamalad7:mixinextras-fabric` | `0.4.1` | `game-runtime` | [https://github.com/LlamaLad7/MixinExtras](https://github.com/LlamaLad7/MixinExtras) | declared |
| `net.fabricmc:cfr` | `0.2.2` | `build-tooling` | [https://github.com/FabricMC/cfr/blob/master/LICENSE](https://github.com/FabricMC/cfr/blob/master/LICENSE) | concluded |
| `net.fabricmc:fabric-loom-native` | `0.2.0` | `build-plugin` | [https://github.com/FabricMC/fabric-loom-native/blob/main/LICENSE](https://github.com/FabricMC/fabric-loom-native/blob/main/LICENSE) | concluded |
| `net.fabricmc:fabric-loom` | `1.10.5` | `build-plugin` | [https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE](https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE) | concluded |
| `net.fabricmc:fabric-mixin-compile-extensions` | `0.6.0` | `build-tooling` | [https://github.com/FabricMC/fabric-mixin-compile-extensions/blob/master/LICENSE](https://github.com/FabricMC/fabric-mixin-compile-extensions/blob/master/LICENSE) | concluded |
| `net.fabricmc:lorenz-tiny` | `4.0.2` | `build-plugin` | [https://github.com/FabricMC/lorenz-tiny/blob/master/LICENSE](https://github.com/FabricMC/lorenz-tiny/blob/master/LICENSE) | concluded |
| `net.fabricmc:sponge-mixin` | `0.15.4+mixin.0.8.7` | `game-runtime` | [https://github.com/FabricMC/Mixin](https://github.com/FabricMC/Mixin) | declared |
| `net.minecrell:terminalconsoleappender` | `1.3.0` | `game-runtime` | [https://github.com/Minecrell/TerminalConsoleAppender/blob/master/LICENSE](https://github.com/Minecrell/TerminalConsoleAppender/blob/master/LICENSE) | concluded |
| `net.sf.jopt-simple:jopt-simple` | `5.0.4` | `game-runtime` | [http://jopt-simple.github.io/jopt-simple](http://jopt-simple.github.io/jopt-simple) | declared |
| `org.cadixdev:at` | `0.1.0-rc1` | `build-plugin` | [https://github.com/CadixDev/at](https://github.com/CadixDev/at) | declared |
| `org.cadixdev:lorenz` | `0.5.7` | `build-plugin` | [https://github.com/CadixDev/Lorenz/blob/develop/LICENSE.txt](https://github.com/CadixDev/Lorenz/blob/develop/LICENSE.txt) | concluded |
| `org.joml:joml` | `1.10.8` | `game-runtime` | [http://joml.org](http://joml.org) | declared |
| `org.slf4j:slf4j-api` | `2.0.16` | `game-runtime` | [http://www.slf4j.org](http://www.slf4j.org) | declared |

### Unicode-3.0

| Component | Version | Scope | Source | Basis |
| --- | --- | --- | --- | --- |
| `com.ibm.icu:icu4j` | `76.1` | `game-runtime` | [https://icu.unicode.org/](https://icu.unicode.org/) | declared |

## Attribution obligations

Apache-2.0 components carry NOTICE requirements that attach on redistribution. This repository redistributes none of them, so this document is the standing attribution record for the build environment rather than a shipped NOTICE file. Anyone republishing a built artifact that embeds these components must reproduce the corresponding upstream NOTICE files; the source URLs above locate them.

## Copyleft components

| Component | License | Scope | Why it is acceptable |
| --- | --- | --- | --- |
| `net.fabricmc:tiny-remapper:0.11.1` | `LGPL-3.0-or-later` | `build-plugin` | Weak copyleft. The repository LICENSE is the LGPL v3 text and every source file header reads 'either version 3 of the License, or (at your option) any later version'. It runs only inside the Gradle plugin classpath (build-plugin scope) as a separate build-time remapping tool; nothing links it into, or distributes it with, any project artifact. |
| `org.jcraft:jorbis:0.0.17` | `LGPL-2.0-or-later` | `game-runtime` | Weak copyleft. The POM declares the unversioned name 'GNU Lesser General Public License'; JCraft's site says only 'JOrbis is under LGPL'. Every upstream source header reads 'the GNU Library General Public License ... either version 2 of the License, or (at your option) any later version', which is SPDX LGPL-2.0-or-later. Minecraft loads it unmodified at game runtime from the user's own cache; this repository distributes no copy and no derivative. |

## Non-redistributable components

| Component | Terms | Scope |
| --- | --- | --- |
| `com.mojang:authlib:6.0.58` | `LicenseRef-Mojang-Proprietary` | `game-runtime` |
| `com.mojang:blocklist:1.0.10` | `LicenseRef-Mojang-Proprietary` | `game-runtime` |
| `com.mojang:jtracy:1.0.29` | `LicenseRef-Mojang-Proprietary` | `game-runtime` |
| `com.mojang:logging:1.5.10` | `LicenseRef-Mojang-Proprietary` | `game-runtime` |
| `com.mojang:patchy:2.2.10` | `LicenseRef-Mojang-Proprietary` | `game-runtime` |
| `com.mojang:text2speech:1.18.11` | `LicenseRef-Mojang-Proprietary` | `game-runtime` |
| `net.minecraft:minecraft:1.21.5` | `LicenseRef-Minecraft-EULA` | `game-runtime` |
| `net.minecraft:mojang-mappings:1.21.5` | `LicenseRef-Mojang-Mappings` | `build-tooling` |

These are obtained by the user's own Gradle build and Minecraft launcher into ignored caches. They are never committed, published, or embedded in a project artifact.

## Curated license conclusions

Each entry below could not be derived from Maven metadata alone. The full reasoning is tracked in `docs/compliance/jvm-license-overrides.json`.

| Component | Concluded | Evidence |
| --- | --- | --- |
| `ca.weblite:java-objc-bridge:1.1` | `Apache-2.0` | [https://github.com/shannah/Java-Objective-C-Bridge/blob/master/LICENSE](https://github.com/shannah/Java-Objective-C-Bridge/blob/master/LICENSE) |
| `com.google.guava:guava:33.7.2-jre` | `Apache-2.0` | [https://github.com/google/guava/blob/v33.7.2/LICENSE](https://github.com/google/guava/blob/v33.7.2/LICENSE) |
| `com.mojang:authlib:6.0.58` | `LicenseRef-Mojang-Proprietary` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `com.mojang:blocklist:1.0.10` | `LicenseRef-Mojang-Proprietary` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `com.mojang:brigadier:1.3.10` | `MIT` | [https://github.com/Mojang/brigadier/blob/master/LICENSE](https://github.com/Mojang/brigadier/blob/master/LICENSE) |
| `com.mojang:datafixerupper:8.0.16` | `MIT` | [https://github.com/Mojang/DataFixerUpper/blob/master/LICENSE](https://github.com/Mojang/DataFixerUpper/blob/master/LICENSE) |
| `com.mojang:jtracy:1.0.29` | `LicenseRef-Mojang-Proprietary` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `com.mojang:logging:1.5.10` | `LicenseRef-Mojang-Proprietary` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `com.mojang:patchy:2.2.10` | `LicenseRef-Mojang-Proprietary` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `com.mojang:text2speech:1.18.11` | `LicenseRef-Mojang-Proprietary` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `fabric-loom:fabric-loom.gradle.plugin:1.10.5` | `MIT` | [https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE](https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE) |
| `io.netty:netty-transport-native-epoll:4.1.118.Final` | `Apache-2.0` | [https://github.com/netty/netty/blob/4.1/LICENSE.txt](https://github.com/netty/netty/blob/4.1/LICENSE.txt) |
| `net.fabricmc.fabric-api:fabric-api-deprecated:0.128.2+1.21.5` | `Apache-2.0` | [https://github.com/FabricMC/fabric-api/blob/26.2/LICENSE](https://github.com/FabricMC/fabric-api/blob/26.2/LICENSE) |
| `net.fabricmc:cfr:0.2.2` | `MIT` | [https://github.com/FabricMC/cfr/blob/master/LICENSE](https://github.com/FabricMC/cfr/blob/master/LICENSE) |
| `net.fabricmc:dev-launch-injector:0.2.1+build.8` | `Apache-2.0` | [https://github.com/FabricMC/dev-launch-injector/blob/master/LICENSE](https://github.com/FabricMC/dev-launch-injector/blob/master/LICENSE) |
| `net.fabricmc:fabric-fernflower:2.0.0` | `Apache-2.0` | [https://github.com/FabricMC/intellij-fernflower/blob/master/README.md](https://github.com/FabricMC/intellij-fernflower/blob/master/README.md) |
| `net.fabricmc:fabric-loader-junit:0.16.10` | `Apache-2.0` | [https://github.com/FabricMC/fabric-loader/blob/master/LICENSE](https://github.com/FabricMC/fabric-loader/blob/master/LICENSE) |
| `net.fabricmc:fabric-loader:0.16.10` | `Apache-2.0` | [https://github.com/FabricMC/fabric-loader/blob/master/LICENSE](https://github.com/FabricMC/fabric-loader/blob/master/LICENSE) |
| `net.fabricmc:fabric-loom-native:0.2.0` | `MIT` | [https://github.com/FabricMC/fabric-loom-native/blob/main/LICENSE](https://github.com/FabricMC/fabric-loom-native/blob/main/LICENSE) |
| `net.fabricmc:fabric-loom:1.10.5` | `MIT` | [https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE](https://github.com/FabricMC/fabric-loom/blob/dev/1.10/LICENSE) |
| `net.fabricmc:fabric-mixin-compile-extensions:0.6.0` | `MIT` | [https://github.com/FabricMC/fabric-mixin-compile-extensions/blob/master/LICENSE](https://github.com/FabricMC/fabric-mixin-compile-extensions/blob/master/LICENSE) |
| `net.fabricmc:lorenz-tiny:4.0.2` | `MIT` | [https://github.com/FabricMC/lorenz-tiny/blob/master/LICENSE](https://github.com/FabricMC/lorenz-tiny/blob/master/LICENSE) |
| `net.fabricmc:mapping-io:0.2.1` | `Apache-2.0` | [https://github.com/FabricMC/mapping-io/blob/dev/LICENSE](https://github.com/FabricMC/mapping-io/blob/dev/LICENSE) |
| `net.fabricmc:stitch:0.6.2` | `Apache-2.0` | [https://github.com/FabricMC/stitch/blob/master/LICENSE](https://github.com/FabricMC/stitch/blob/master/LICENSE) |
| `net.fabricmc:tiny-mappings-parser:0.3.0+build.17` | `Apache-2.0` | [https://github.com/FabricMC/tiny-mappings-parser/blob/master/LICENSE](https://github.com/FabricMC/tiny-mappings-parser/blob/master/LICENSE) |
| `net.fabricmc:tiny-remapper:0.11.1` | `LGPL-3.0-or-later` | [https://github.com/FabricMC/tiny-remapper/blob/master/LICENSE](https://github.com/FabricMC/tiny-remapper/blob/master/LICENSE) |
| `net.java.dev.jna:jna-platform:5.15.0` | `Apache-2.0` | [https://github.com/java-native-access/jna/blob/master/LICENSE](https://github.com/java-native-access/jna/blob/master/LICENSE) |
| `net.java.dev.jna:jna:5.15.0` | `Apache-2.0` | [https://github.com/java-native-access/jna/blob/master/LICENSE](https://github.com/java-native-access/jna/blob/master/LICENSE) |
| `net.minecraft:minecraft:1.21.5` | `LicenseRef-Minecraft-EULA` | [https://www.minecraft.net/en-us/eula](https://www.minecraft.net/en-us/eula) |
| `net.minecraft:mojang-mappings:1.21.5` | `LicenseRef-Mojang-Mappings` | [https://www.minecraft.net/en-us/usage-guidelines](https://www.minecraft.net/en-us/usage-guidelines) |
| `net.minecrell:terminalconsoleappender:1.3.0` | `MIT` | [https://github.com/Minecrell/TerminalConsoleAppender/blob/master/LICENSE](https://github.com/Minecrell/TerminalConsoleAppender/blob/master/LICENSE) |
| `org.apache.logging.log4j:log4j-api:2.25.5` | `Apache-2.0` | [https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-api/2.25.5/log4j-api-2.25.5.jar](https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-api/2.25.5/log4j-api-2.25.5.jar) |
| `org.apache.logging.log4j:log4j-core:2.25.5` | `Apache-2.0` | [https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-core/2.25.5/log4j-core-2.25.5.jar](https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-core/2.25.5/log4j-core-2.25.5.jar) |
| `org.apache.logging.log4j:log4j-slf4j2-impl:2.25.5` | `Apache-2.0` | [https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-slf4j2-impl/2.25.5/log4j-slf4j2-impl-2.25.5.jar](https://repo.maven.apache.org/maven2/org/apache/logging/log4j/log4j-slf4j2-impl/2.25.5/log4j-slf4j2-impl-2.25.5.jar) |
| `org.apiguardian:apiguardian-api:1.1.2` | `Apache-2.0` | [https://github.com/apiguardian-team/apiguardian/blob/main/LICENSE](https://github.com/apiguardian-team/apiguardian/blob/main/LICENSE) |
| `org.cadixdev:lorenz:0.5.7` | `MIT` | [https://github.com/CadixDev/Lorenz/blob/develop/LICENSE.txt](https://github.com/CadixDev/Lorenz/blob/develop/LICENSE.txt) |
| `org.jcraft:jorbis:0.0.17` | `LGPL-2.0-or-later` | [https://github.com/ymnk/jorbis/blob/master/com/jcraft/jorbis/VorbisFile.java](https://github.com/ymnk/jorbis/blob/master/com/jcraft/jorbis/VorbisFile.java) |
| `org.jetbrains:annotations:25.0.0` | `Apache-2.0` | [https://github.com/JetBrains/java-annotations/blob/master/LICENSE.txt](https://github.com/JetBrains/java-annotations/blob/master/LICENSE.txt) |
| `org.jline:jline-reader:3.20.0` | `BSD-3-Clause` | [https://github.com/jline/jline3/blob/master/LICENSE.txt](https://github.com/jline/jline3/blob/master/LICENSE.txt) |
| `org.jline:jline-terminal:3.20.0` | `BSD-3-Clause` | [https://github.com/jline/jline3/blob/master/LICENSE.txt](https://github.com/jline/jline3/blob/master/LICENSE.txt) |
| `org.jspecify:jspecify:1.0.1` | `Apache-2.0` | [https://github.com/jspecify/jspecify/blob/v1.0.1/LICENSE](https://github.com/jspecify/jspecify/blob/v1.0.1/LICENSE) |
| `org.junit.jupiter:junit-jupiter-api:5.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.junit.jupiter:junit-jupiter-engine:5.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.junit.jupiter:junit-jupiter-params:5.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.junit.jupiter:junit-jupiter:5.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.junit.platform:junit-platform-commons:1.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.junit.platform:junit-platform-engine:1.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.junit.platform:junit-platform-launcher:1.11.4` | `EPL-2.0` | [https://github.com/junit-team/junit-framework/blob/main/LICENSE.md](https://github.com/junit-team/junit-framework/blob/main/LICENSE.md) |
| `org.opentest4j:opentest4j:1.3.0` | `Apache-2.0` | [https://github.com/ota4j-team/opentest4j/blob/main/LICENSE](https://github.com/ota4j-team/opentest4j/blob/main/LICENSE) |

## Minecraft and Mojang material

NeuroCraft Fly is not an official Minecraft product and is not approved by or associated with Mojang or Microsoft. Minecraft, the Mojang-published libraries, and the official obfuscation mappings are proprietary; follow the current [Minecraft Usage Guidelines](https://www.minecraft.net/en-us/usage-guidelines) and the applicable EULA.
