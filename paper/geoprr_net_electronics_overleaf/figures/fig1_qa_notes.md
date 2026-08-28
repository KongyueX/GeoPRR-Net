# Figure 1 QA notes

## Figure contract

- Core claim: capture conditions change which visual cue is reliable, so the reader preserves geometry/base, polar, and relational evidence and routes their candidate means before returning one posterior.
- Archetype: operating-regime image plate plus evidence-flow schematic.
- Panel hierarchy: panel (a) establishes the application problem; panel (b) is the hero architecture panel.
- Final PNG: 2142 × 1657 pixels at 300 dpi, approximately 181.4 × 140.3 mm.
- Source preflight: `validate_figure.py --strict` returned 20 PASS, 0 WARN, and 0 FAIL.
- PDF audit: not run because this review revision intentionally generates and links only the PNG asset.

## Panel audit

| Panel | Unique claim | Quantitative summary | Replicate unit | Labels | Collision check | Pass |
|---|---|---|---|---|---|---|
| a | The same dial presents different cue reliability under stable, blurred, and oblique capture | None; examples are explicitly illustrative | Not applicable | Context and failure mechanism are directly labeled | Images, labels, and provenance note remain separate at final size | Yes |
| b | Three distinct evidence paths feed a sample-conditioned router and one moment-consistent posterior | None; no sample-specific weights or measured activations are shown | Not applicable | Every stage and the deployment envelope are directly labeled | Arrows, branch labels, icons, and output box are clear at final size | Yes |

## Image-integrity record

- Raw file: `assets/rf100_test_000000.png`.
- Provenance: public RF100-VL `needle-base-tip-min-max` test split; manuscript citation and `assets/README.md` record the source.
- Crop: the displayed clean image is the prepared gauge ROI. The support-normalized illustration crops only the non-background support of the projectively transformed example.
- Blur: global Gaussian blur with sigma 5.2 pixels, used only as an explanatory view.
- Projective transform: one global four-corner homography with a neutral constant border, used only as an explanatory view.
- Brightness/contrast/gamma: unchanged.
- Pseudo-color: none on photographic examples; the small relational tile is a schematic glyph, not image evidence.
- Stitching: none.
- Reuse: the finished PNG is copied unchanged to the Chinese Markdown asset directory.
- Quantification link: none; the caption explicitly states that the examples are not routing measurements or performance evidence.
