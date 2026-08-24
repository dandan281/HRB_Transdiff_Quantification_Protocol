"""T04 candidate: centreline tracing for crossing myotubes.

Exists because of a limitation measured on T02, not a preference. Omnipose --
like Cellpose, like any model whose output is a label map -- assigns **one
instance per pixel**. Myotubes cross: 59.5% of fibres in `plate32_dense_v1` are
broken by a crossing, and the corpus records 24,263 overlap pixels in well B02
alone, all of which had to be marked `ignore` because the representation cannot
hold two objects at one location. `false_split_count` (52/375) is the
predeclared T03 primary and is exactly the cost of that.

A tracer does not partition pixels. It predicts a centreline, an orientation
field and a crossing map, then follows one fibre at a time through the graph, so
two fibres may share a pixel and both keep their identity. The operator's
annotation is already centrelines -- freehand polylines in Fiji, one per fibre --
so the target for this model is closer to what was actually drawn than the
rasterised ribbons T02 trains on.

Layout mirrors `omnipose_lab`: data first and provable offline, model after.
"""
