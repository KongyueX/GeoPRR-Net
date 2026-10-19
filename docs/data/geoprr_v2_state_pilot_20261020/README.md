# One controlled pointer-state generation pilot

One fixed-scene group contains four rendered pointer states: **0.15, 0.35, 0.65 and 0.85** of full scale. All **11** render-pre, render-post and subsequent-call checks preserve the recorded non-intervened geometry, appearance and camera factors. The existing source-pair loader accepts four diagnostic state pairs. The fixed U/M fields and camera transform are identical across states; all **16** ground-truth donor–recipient combinations reproduce the recipient state with a maximum error of **0.000073016%FS** and minimum support mass **0.999999940**.

These are source-generation and label-contract results. No learned model participates in the combinations, and the four images do not establish improved training, accuracy, robustness or generalization. All generated manifests carry `diagnostic_only=true`.

## Generation

- External source: [official SyncG repository](https://github.com/SKYWOWDYH/SyncG), revision `337b868f0f5495b31acc383433cac44040247bf2`.
- Runtime: portable Blender 4.1.1; bundled Python 3.11.7 and NumPy 1.24.3; isolated OpenCV 4.10.0.
- Execution: Cycles CPU, four threads, 16 samples, 512×512 pixels; no GPU rendering.
- Intervention: construct the scene once and change only the same selected pointer's Z rotation. Record actual objects, geometry, fonts, materials, environment, camera and rendering settings.
- Labels: upstream geometry annotations projected at the actual image dimensions; virtual Q is constructed on the shared dial plane by the existing dataset code. The visible physical pointer tip remains a separate annotation.
- Data identity: a separate diagnostic directory and manifests; the frozen original training, source-validation and test lists remain unchanged.

`geometry_diagnostic.json` contains per-image geometry audits and the 16 GT compositions. It has no images, learned weights or external Blender assets. Renderer and preparation entry points are described in [the design document](../../GEOPRR_V2_COMPOSITIONAL_DESIGN_CN.md).

## Limitations

This pilot covers one layout, one camera and one environment, with four interior states. It does not test exact range endpoints, difficult viewpoints, unseen layouts, learned component swapping or matched-exposure training. Its low-sample CPU configuration is a pipeline diagnostic; a formal rendering-quality and compute protocol remains to be established. Multi-recipient training and held-out combination edges are not yet implemented.
