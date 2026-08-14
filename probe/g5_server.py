"""G5 test server — serves the vendored viewer, the test page, and .rrd files from the
Volume. Throwaway gate infrastructure (the real app/ arrives in M2), but it exercises
the eventual serving path: static assets baked into the image, .rrd from Volume.

    uv run modal deploy probe/g5_server.py
"""

from __future__ import annotations

import modal

# Standalone slim app: no imports from train/ (the image carries no python packages —
# the first deploy failed exactly there); volume + mount declared inline.
VOLUME_MOUNT = "/vol"
checkpoints = modal.Volume.from_name("opw-checkpoints", create_if_missing=True, version=2)

RERUN_VIEWER_VERSION = "0.36.0"
RERUN_VIEWER_TGZ_SHA256 = "3d517b20d264b2fdc0a073a002fca2560a9d7a0554fad1e0c027612fcd570c56"
RERUN_VIEWER_TGZ_URL = (
    f"https://registry.npmjs.org/@rerun-io/web-viewer/-/web-viewer-{RERUN_VIEWER_VERSION}.tgz"
)


def _fetch_viewer_assets() -> None:
    """Image-build step: fetch the pinned viewer tarball from npm (immutable per
    version), verify sha256, extract the runtime assets. Keeps the 48 MB wasm out of
    git while runtime serving stays fully our-origin (DEVLOG Session 15). Backup copy
    of the tarball: Volume opw-checkpoints /vendor-backup/."""
    import hashlib
    import shutil
    import tarfile
    import urllib.request

    urllib.request.urlretrieve(RERUN_VIEWER_TGZ_URL, "/tmp/wv.tgz")
    with open("/tmp/wv.tgz", "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    assert digest == RERUN_VIEWER_TGZ_SHA256, f"viewer tarball sha256 mismatch: {digest}"
    with tarfile.open("/tmp/wv.tgz") as tf:
        tf.extractall("/tmp/wv", filter="data")
    import os

    os.makedirs("/assets/vendor/rerun", exist_ok=True)
    for name in ("re_viewer.js", "re_viewer_bg.wasm", "index.js"):
        shutil.copy(f"/tmp/wv/package/{name}", f"/assets/vendor/rerun/{name}")


server_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi[standard]==0.141.1")
    .run_function(_fetch_viewer_assets)
    .add_local_dir("web/g5", remote_path="/assets/g5")
)

app = modal.App("opw-g5-test")


@app.function(image=server_image, volumes={VOLUME_MOUNT: checkpoints})
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve():  # type: ignore[no-untyped-def]
    from pathlib import Path

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    api = FastAPI()

    @api.get("/rrd/{name}")
    def rrd(name: str) -> FileResponse:
        if "/" in name or ".." in name or not name.endswith(".rrd"):
            raise HTTPException(status_code=404)
        checkpoints.reload()
        path = Path(VOLUME_MOUNT) / "rrd" / name
        if not path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="application/octet-stream")

    @api.get("/vendor/rerun/re_viewer")
    def re_viewer_alias() -> FileResponse:
        # The vendored index.js does a browser-side `import("./re_viewer")` with no
        # extension (bundler idiom); alias it to the real file.
        return FileResponse("/assets/vendor/rerun/re_viewer.js", media_type="text/javascript")

    api.mount("/vendor/rerun", StaticFiles(directory="/assets/vendor/rerun"), name="vendor")
    api.mount("/", StaticFiles(directory="/assets/g5", html=True), name="page")
    return api
