"""User-defined shell commands and app launches (§5.5)."""

from __future__ import annotations

import logging
import os
import subprocess

log = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000  # don't flash a console for shell commands


class ShellBackend:
    def run_shell(self, cmd: str) -> None:
        log.info("shell: %s", cmd)
        subprocess.Popen(cmd, shell=True, creationflags=CREATE_NO_WINDOW,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def launch(self, target: str) -> None:
        log.info("launch: %s", target)
        os.startfile(target)  # apps, documents, and URIs alike
