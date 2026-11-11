"""Container entry point. SageMaker runs the image as ``<image> serve``."""

import logging
import os
import sys

import uvicorn

from .app import create_app


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if command != "serve":
        sys.exit(f"unknown command {command!r}; SageMaker starts the container with 'serve'")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host="0.0.0.0", port=int(os.environ.get("SAGEMAKER_BIND_TO_PORT", "8080")),
                timeout_keep_alive=75)  # longer than the SageMaker front end's 60 s


if __name__ == "__main__":
    main()
