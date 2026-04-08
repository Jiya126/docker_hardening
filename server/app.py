# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server — wraps DockerHardeningEnvironment via OpenEnv's create_app."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required.  Install with: uv sync") from e

try:
    from ..models import DockerHardeningAction, DockerHardeningObservation
    from .docker_hardening_environment import DockerHardeningEnvironment
except ImportError:
    from models import DockerHardeningAction, DockerHardeningObservation
    from server.docker_hardening_environment import DockerHardeningEnvironment

app = create_app(
    DockerHardeningEnvironment,
    DockerHardeningAction,
    DockerHardeningObservation,
    env_name="docker_hardening",
    max_concurrent_envs=1,
)


@app.get("/")
async def root():
    return {
        "name": "SCA-Gym",
        "env": "docker_hardening",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state", "/health", "/schema"],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
