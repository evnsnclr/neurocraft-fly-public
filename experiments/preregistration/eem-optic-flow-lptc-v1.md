# EEM preregistration draft — engineered optic-flow front end for an LPTC circuit, v1

- **Status: `draft_not_frozen`, and additionally gated on a fixture that does
  not exist yet.** No result may be generated under this document; nothing in
  it may be built into a live loop before the freeze conditions at the end are
  met. This draft records the design direction ADR 0008 authorizes for
  Engineered-Encoder Mode (EEM), Mode 3 of 3.
- **Relationship to D-037 / AB-07 (stated up front).** `VISION-FEASIBILITY-01`
  measured that every retinotopic construction exceeds the live verification
  ceiling, and that the one population that fits — 159 nodes / 584 edges at the w≥5 floor,
  built on lobula plate tangential cells (LPTCs) — receives 82.32% of its input
  weight from T4/T5, so any encoder feeding it must compute optic flow itself.
  Under Faithful Organism Mode that is policy in the encoder and remains
  refused. Under EEM it is **the declared experiment**: an engineered,
  disclosed Hassenstein–Reichardt-style motion front end feeding a measured
  circuit whose biological inputs carry exactly that signal. The refusal is
  not weakened; the claim type changes, and the mode identity says so.

## What must exist before a freeze

1. **The LPTC fixture.** A new curated fixture extracted by the existing
   deterministic extractor lane (new pin module, loader untouched), from the
   population `VISION-FEASIBILITY-01` measured as fitting the ceiling.
   Deliverables: fixture JSON, pin, receipt, three-way byte-identical
   extraction — the `FIXTURE-MCNS-02` bar.
2. **Sign coverage measured** for that fixture (a `SIGN-OVERLAY-02`-class
   receipt) before any dynamics claim.
3. **The encoder specification:** input source (Minecraft rays/luminance or
   offline stimulus), the exact motion computation, its parameter count, and
   its identity hash. The encoder is engineered and must be presented as such
   in every artifact; its parameters are declared, never tuned on outcomes.
4. **Task and metric:** the candidate demonstration is optomotor-style gain
   versus stimulus temporal frequency, connectome versus matched nulls
   (the TRM condition set applies unchanged to the LPTC fixture).

## Explicitly out of scope until then

Any Minecraft integration (a retina/luminance observation is a
protocol-schema-scale change), any claim about fly vision, and any FOM
wording. If the LPTC extraction misses the ceiling or the sign coverage is
poor, that measurement is the reportable outcome and this draft is amended,
not forced.

## Freeze checklist

- [ ] LPTC fixture receipt + identity
- [ ] sign-coverage receipt
- [ ] encoder spec + identity hash
- [ ] task, metric, and null set bound
- [ ] adversarial review recorded
- [ ] DECISIONS entry recorded
