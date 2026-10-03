# Active capture photometry contract

The active capture pipeline uses exactly one condition:

```text
both = (D - B) / (W - B)
```

`D` is the selected FP32 camera-count image, and `B` and `W` are the recorded per-DNG black and white levels. The capture runner writes only the `both` inference and report output. Active photometry APIs reject retired condition inputs.

Historical condition folders, manifests, numeric archives, previews, and comparison evidence are preserved as provenance. Current reports and indexes render condition-keyed records and clipping details from `both` only. HTML refreshes do not change historical manifest or index JSON, numeric arrays, checkpoint outputs, or prior PNGs; any new both-only gallery uses separate filenames.
