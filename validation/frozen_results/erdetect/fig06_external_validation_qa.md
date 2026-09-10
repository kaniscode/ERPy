# Figure 6 publication QA — Round 03

The final standard/JNM publication-scale PNG and independent full-page PDF rasterization (254 dpi) were inspected. At 180 × 162 mm, there is no clipping, missing glyph, text/legend collision or adjacent boundary-tick ambiguity. Both sensitivity axes keep their original ranges, ticks and grids; only their 100 labels are omitted. Participant symbols remain separated. All visible text is at least 8 pt.

Measured performance is exactly unchanged: all 29 rows and every column in the plotted-data CSV match the pre-change CSV with exact equality. Source metrics SHA256: `a3e932b306da2e91c0a6c58a0bd6202c5319191222cef3a21d00c5cf77a3214e`. The figure retains 32,048 paired records and 32,121 all-source evaluable records in 13 independent participants.

The footer/caption/alt text state that 32,121/33,694 (95.33%) is conditional coverage after source and site eligibility. The caption also reports 5,901 otherwise eligible records without source events and expanded coverage 32,121/39,595 (81.12%). Measured performance is unchanged; unavailable records are not assigned negative calls.

Exports: vector PDF/SVG, 300-dpi PNG, and 600-dpi RGB LZW TIFF (4251 × 3826 pixels). Frontiers retains native ≥2-point visible strokes; Neuroinformatics uses lowercase a/b. Source/plotter/export hashes and text/stroke checks are in the JSON companion. Frozen round_02 copies were not modified.
