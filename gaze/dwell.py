"""
dwell.py
---------
Stage 5+6: stable target (hysteresis) -> dwell -> selection.

update(raw_target, valid) is called once per frame.
  * A NEW target must be seen for `acquire_time` before it becomes the
    stable target (a single noisy frame can never switch it).
  * Once a target is stable, brief misses (< `release_time`) do NOT reset
    the dwell; only a sustained departure does.
  * Dwell only counts time on the STABLE target.
  * Invalid frames neither advance nor cancel (until release_time passes).
"""

import time


class DwellDetector:
    def __init__(self, dwell_time=2.0, acquire_time=0.25, release_time=0.35, cooldown=1.0):
        self.dwell_time = dwell_time
        self.acquire_time = acquire_time
        self.release_time = release_time
        self.cooldown = cooldown
        self.reset()

    def reset(self):
        self.stable_target = None
        self._dwell_start = None
        self._candidate = None
        self._candidate_since = None
        self._last_seen_stable = None
        self._cooldown_until = 0.0

    def update(self, raw_target, valid=True):
        """Returns (stable_target, progress 0-1, selected)."""
        now = time.time()

        if now < self._cooldown_until:
            return self.stable_target, 0.0, False

        if not valid:
            # hold state; drop it only if invalid for too long
            if self.stable_target and now - self._last_seen_stable > self.release_time:
                self._drop()
            return self.stable_target, self._progress(now), False

        if self.stable_target is not None:
            if raw_target == self.stable_target:
                self._last_seen_stable = now
                self._candidate = None
            else:
                # outside current target: wait out release_time before leaving
                if now - self._last_seen_stable > self.release_time:
                    self._drop()
                    self._track_candidate(raw_target, now)
                return self.stable_target, self._progress(now), False
        else:
            self._track_candidate(raw_target, now)
            if self.stable_target is None:
                return None, 0.0, False

        progress = self._progress(now)
        if progress >= 1.0:
            sel = self.stable_target
            self.reset()
            self._cooldown_until = now + self.cooldown
            return sel, 1.0, True
        return self.stable_target, progress, False

    def _track_candidate(self, raw_target, now):
        if raw_target is None:
            self._candidate = None
            return
        if raw_target != self._candidate:
            self._candidate = raw_target
            self._candidate_since = now
            return
        if now - self._candidate_since >= self.acquire_time:
            self.stable_target = raw_target
            self._dwell_start = now
            self._last_seen_stable = now
            self._candidate = None

    def _progress(self, now):
        if self.stable_target is None or self._dwell_start is None:
            return 0.0
        return min((now - self._dwell_start) / self.dwell_time, 1.0)

    def _drop(self):
        self.stable_target = None
        self._dwell_start = None
        self._candidate = None
