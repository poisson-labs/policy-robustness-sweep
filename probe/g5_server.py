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

server_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi[standard]==0.141.1")
    .add_local_dir("web/vendor/rerun", remote_path="/assets/vendor/rerun")
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
