"""Activate Quest roofline patches in this benchmark-only Python path."""

import os
import sys

from quest_roofline_patch import ENV_NAME, install, mode_description


mode = os.environ.get(ENV_NAME)
if mode:
    install(mode)
    sys.stderr.write(f"[quest-roofline] mode={mode}: {mode_description(mode)}\n")
    sys.stderr.flush()
