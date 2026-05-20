# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS server entry point.

Launches two GMS server processes per GPU (one for weights, one for kv_cache),
then supervises them: terminates the rest if any child exits, and propagates
the first non-zero exit code. Runs until SIGTERM (pod termination kills it)
or until a child exits.
"""

from __future__ import annotations

import argparse
import logging
import signal
import subprocess
import sys
import time

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_TAGS = ("weights", "kv_cache")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU Memory Service supervisor (one server per (device, tag))."
    )
    parser.add_argument(
        "--device-kind",
        type=str,
        default=VMMDeviceType.CUDA.value,
        choices=[d.value for d in VMMDeviceType],
        help="VMM device kind forwarded to server (default: cuda).",
    )
    args = parser.parse_args()

    vmm = get_vmm_device(VMMDeviceType.from_str(args.device_kind))
    vmm.ensure_initialized()
    devices = vmm.list_devices()
    processes = []
    for device in devices:
        for tag in _TAGS:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gpu_memory_service",
                    "--device",
                    str(device),
                    "--tag",
                    tag,
                    "--device-kind",
                    args.device_kind,
                ]
            )
            logger.info(
                "Started GMS device=%d tag=%s device_kind=%s pid=%d",
                device,
                tag,
                args.device_kind,
                proc.pid,
            )
            processes.append(proc)

    def shutdown() -> None:
        for process in processes:
            if process.poll() is None:
                process.terminate()

    def terminate(*_args) -> None:
        shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)

    while True:
        running = False
        for process in processes:
            exit_code = process.poll()
            if exit_code is None:
                running = True
                continue
            shutdown()
            raise SystemExit(exit_code)

        if not running:
            return
        time.sleep(1)


if __name__ == "__main__":
    main()
