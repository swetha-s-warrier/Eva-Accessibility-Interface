import time
from .actions import (
    launch_browser,
    launch_media,
    launch_files,
    launch_notes,
    launch_camera
)

VALID_TARGETS = {
    "BROWSER": launch_browser,
    "MEDIA": launch_media,
    "FILES": launch_files,
    "NOTES": launch_notes,
    "CAMERA": launch_camera
}

class ActionDispatcher:
    def __init__(self, cooldown_seconds: float = 3.0):
        self.cooldown_seconds = cooldown_seconds
        self.last_execution_time = 0.0
        self.last_target = None

    def execute_action(self, target_id: str, force: bool = False):
        """
        Executes action for an internal feature ID.
        Includes a cooldown throttle to avoid rapid multiple triggers from gaze jitter.
        """
        now = time.time()
        
        target = str(target_id).strip().upper()
        if target not in VALID_TARGETS:
            return {
                "status": "error", 
                "message": f"Invalid Target ID: '{target_id}'. Expected one of {list(VALID_TARGETS.keys())}"
            }

        # Prevent double-firing within cooldown window unless explicitly forced
        if not force and (now - self.last_execution_time) < self.cooldown_seconds:
            return {
                "status": "throttled",
                "message": f"Ignored: Action '{target}' received within {self.cooldown_seconds}s cooldown."
            }

        self.last_execution_time = now
        self.last_target = target
        action_func = VALID_TARGETS[target]
        result = action_func()
        return result