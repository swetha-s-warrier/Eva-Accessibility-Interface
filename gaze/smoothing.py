"""
smoothing.py
-------------
Stage 3+4 of the pipeline:  calibrated gaze -> validity/outlier check -> EMA.

update() returns (x, y, valid).  A single wild frame is REJECTED (the
smoothed point is held, not moved).  If the new position persists for
`accept_after` consecutive frames it is treated as a genuine saccade and
the filter re-anchors, so it stays responsive.
"""

import math


class GazeSmoother:
    def __init__(self, alpha=0.30, max_jump=0.25, accept_after=4, margin=0.15):
        self.alpha = alpha              # EMA weight of newest sample
        self.max_jump = max_jump        # normalized jump treated as outlier
        self.accept_after = accept_after  # consecutive far frames => real move
        self.margin = margin            # allowed overshoot outside [0,1] before invalid
        self._smoothed = None
        self._outliers = 0
        self._last_far = None

    def _is_valid(self, x, y):
        if x is None or y is None:
            return False
        if not (math.isfinite(x) and math.isfinite(y)):
            return False
        lo, hi = -self.margin, 1.0 + self.margin
        return lo <= x <= hi and lo <= y <= hi

    def update(self, x, y):
        """Returns (sx, sy, valid). Held value is returned when frame rejected."""
        if not self._is_valid(x, y):
            self._outliers = 0
            if self._smoothed is None:
                return None, None, False
            return self._smoothed[0], self._smoothed[1], False

        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)

        if self._smoothed is None:
            self._smoothed = (x, y)
            return x, y, True

        px, py = self._smoothed
        dist = math.hypot(x - px, y - py)

        if dist > self.max_jump:
            self._outliers += 1
            # consistent with the previous far frame? then it's a real move
            consistent = (self._last_far is None or
                          math.hypot(x - self._last_far[0], y - self._last_far[1]) < self.max_jump)
            self._last_far = (x, y)
            if self._outliers < self.accept_after or not consistent:
                return px, py, False          # reject this frame, hold position
            # sustained: re-anchor halfway so target change is quick but not a snap
            self._outliers = 0
            self._last_far = None
            self._smoothed = (px + 0.5 * (x - px), py + 0.5 * (y - py))
            return self._smoothed[0], self._smoothed[1], True

        self._outliers = 0
        self._last_far = None
        self._smoothed = (px + self.alpha * (x - px), py + self.alpha * (y - py))
        return self._smoothed[0], self._smoothed[1], True

    def reset(self):
        self._smoothed = None
        self._outliers = 0
        self._last_far = None
