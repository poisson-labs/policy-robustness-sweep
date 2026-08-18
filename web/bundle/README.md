# Embeddable bundle for the Astro/Netlify site (M2-03)

The primary deployment target (DECISION, DEVLOG Session 17): the artifact is consumed by
`poisson-labs-site` (Astro 5 + MDX, vanilla `.astro` components, Netlify) as static
assets + a mount component; the Modal app is reached through a same-origin `/api/*`
proxy. `poisson-sweep.modal.run` remains the standalone mirror.

## What ships (this directory)

| path | goes to (in the site repo) | role |
|---|---|---|
| `public/one-policy-400-worlds/` | `public/one-policy-400-worlds/` | static assets: viewer wasm/js (fetched by `scripts/fetch_viewer.sh`, hash-verified — never CDN at runtime), the G5 test page, later the M3 bundle |
| `component/OnePolicy400Worlds.astro` | `src/components/` | the mount component (M3 fills it; today it mounts the viewer for a given .rrd) |
| `netlify/netlify.toml.stanza` | merged into `netlify.toml` | `/api/*` → Modal proxy (200 rewrite), COOP/COEP scoped to the artifact paths, immutable caching for vendor assets |

## Runtime contract

- Browser talks ONLY to the site origin: `/api/probe`, `/api/rrd/<key>.rrd`,
  `/api/manifest`, `/api/health`, `/api/receipt` (proxied to Modal by Netlify).
- Viewer wasm served from `/one-policy-400-worlds/vendor/rerun/` (same origin).
- COOP/COEP are scoped to `/one-policy-400-worlds/*` and the post route ONLY — the
  site's other pages keep loading Google Fonts / goatcounter unaffected. NOTE: those
  two external loads happen in `BaseLayout` and therefore ALSO on the artifact post;
  under `require-corp` they will be blocked unless the post uses a layout variant
  without them or they are served with CORP headers. Flagged for Taylor (Session 28).

## Reproducing the vendored viewer (no bytes in git)

    scripts/fetch_viewer.sh public/one-policy-400-worlds/vendor/rerun

fetches `@rerun-io/web-viewer@0.36.0` from npm, verifies sha256 against
`web/vendor/rerun/SHA256SUMS`, and extracts `re_viewer.js`, `re_viewer_bg.wasm`,
`index.js` (+ an extensionless `re_viewer` copy for the viewer's bare import).
