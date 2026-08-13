# web/

Static frontend (vanilla TS/JS, no framework; Vite for bundling only). Empty until M3 —
no frontend code before gates G1–G5 pass (spec §4).

`vendor/rerun/` will hold the pinned Rerun web-viewer wasm/JS plus a `VERSION` file;
CI asserts that version equals the `rerun-sdk` pin in `uv.lock`.
