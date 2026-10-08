"""
piece_counter.py - Machine-centric sewing piece counter (v4)

Counting logic:
    IDLE -> HAND_IN_SRC -> IN_TRANSIT -> HAND_IN_DST -> CYCLE -> COOLDOWN

A cycle belongs to a fixed machine, not to a volatile person/track ID.

The optional payload gate is deliberately NOT a simple "dark pixels near wrist"
heuristic. It combines:
    1) Source-area change after the wrist leaves the pickup area.
    2) Local temporal motion around the active wrist during transit.

Source-area change is the primary evidence because the camera scene has a
mostly static/light table and dark fabric. Wrist motion is supporting evidence.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import os

import cv2
import numpy as np


class MachineState(Enum):
    IDLE = "IDLE"
    HAND_IN_SRC = "HAND_IN_SRC"
    IN_TRANSIT = "IN_TRANSIT"
    HAND_IN_DST = "HAND_IN_DST"
    COOLDOWN = "COOLDOWN"


Phase = MachineState


@dataclass
class CounterParams:
    src_dwell: float = 0.30
    dst_dwell: float = 0.30
    max_transit: float = 6.0
    cooldown: float = 1.5
    kp_gap: float = 0.30
    zone_margin: int = 8
    ema_alpha: float = 0.5
    hand_crop_radius: int = 50
    verbose: bool = True

    # Existing/legacy compatibility options
    verify_src: bool = False
    src_change_min: float = 6.0
    verify_hold: bool = False
    hold_min: float = 0.40
    hold_px_frac: float = 0.25
    hold_radius: int = 50
    fabric_tol: float = 40.0

    # Payload verification
    payload_check: bool = True
    payload_radius: int = 60
    payload_inner_radius: int = 18
    payload_score_threshold: float = 0.50
    payload_min_samples: int = 1
    payload_dark_v: int = 115
    payload_motion_thresh: float = 2.5
    payload_direction_cos: float = 0.55

    # New source-change evidence
    payload_source_change_threshold: float = 0.18
    payload_source_settle_sec: float = 0.16
    payload_motion_accept_threshold: float = 0.78
    payload_source_local_radius: int = 90


class _Debounce:
    __slots__ = ("gap", "since", "last_in")

    def __init__(self, gap: float):
        self.gap = float(gap)
        self.since: Optional[float] = None
        self.last_in: Optional[float] = None

    def reset(self):
        self.since = None
        self.last_in = None

    def update(self, inside: bool, now: float):
        if inside:
            if self.since is None or (
                self.last_in is not None and now - self.last_in > self.gap
            ):
                self.since = now
            self.last_in = now

    def dwell(self, now: float) -> float:
        if self.since is None or self.last_in is None:
            return 0.0
        if now - self.last_in > self.gap:
            return 0.0
        return self.last_in - self.since

    def away(self, now: float) -> float:
        return float("inf") if self.last_in is None else now - self.last_in


@dataclass
class MachineCycleTracker:
    mzid: int
    name: str
    state: MachineState = MachineState.IDLE
    cycle_count: int = 0

    src_deb: _Debounce = field(default_factory=lambda: _Debounce(0.30))
    dst_deb: _Debounce = field(default_factory=lambda: _Debounce(0.30))

    t_pick: float = 0.0
    t_place: float = 0.0
    t_src_exit: float = 0.0
    cooldown_until: float = 0.0

    active_worker_id: Optional[int] = None
    active_wrist_side: Optional[str] = None
    last_wrist_pos: Optional[Tuple[float, float]] = None
    wrist_confs: List[float] = field(default_factory=list)
    hand_crop: Optional[np.ndarray] = None

    # Payload state
    payload_scores: deque = field(default_factory=lambda: deque(maxlen=30))
    payload_confirmed: bool = False
    payload_positive_samples: int = 0
    payload_peak_score: float = 0.0
    payload_last_score: float = 0.0
    payload_prev_gray: Optional[np.ndarray] = None
    payload_prev_wrist_pos: Optional[Tuple[float, float]] = None

    # Source-area evidence
    src_baseline_gray: Optional[np.ndarray] = None
    src_baseline_zone: Optional[Tuple[int, int, int, int]] = None
    src_change_peak: float = 0.0
    src_change_last: float = 0.0
    pickup_src_point: Optional[Tuple[float, float]] = None
    src_change_checked: bool = False
    src_change_ready_time: float = 0.0

    def reset_cycle_evidence(self):
        self.wrist_confs.clear()
        self.hand_crop = None
        self.payload_scores.clear()
        self.payload_confirmed = False
        self.payload_positive_samples = 0
        self.payload_peak_score = 0.0
        self.payload_last_score = 0.0
        self.payload_prev_gray = None
        self.payload_prev_wrist_pos = None
        self.src_change_peak = 0.0
        self.src_change_last = 0.0
        self.pickup_src_point = None
        self.src_change_checked = False
        self.src_change_ready_time = 0.0

    def reset_to_idle(self):
        self.state = MachineState.IDLE
        self.src_deb.reset()
        self.dst_deb.reset()
        self.t_pick = 0.0
        self.t_src_exit = 0.0
        self.active_worker_id = None
        self.active_wrist_side = None
        self.last_wrist_pos = None
        self.reset_cycle_evidence()



def _inside(z, x: float, y: float, margin: int) -> bool:
    if z is None:
        return False
    m = min(
        max(0, int(margin)),
        max(0, (int(z.x2) - int(z.x1)) // 4),
        max(0, (int(z.y2) - int(z.y1)) // 4),
    )
    return (z.x1 + m) <= x <= (z.x2 - m) and (z.y1 + m) <= y <= (z.y2 - m)


def _crop_wrist(frame: np.ndarray, x: float, y: float, r: int) -> Optional[np.ndarray]:
    if frame is None or frame.size == 0:
        return None
    h, w = frame.shape[:2]
    ix, iy = int(round(x)), int(round(y))
    x1, y1 = max(0, ix - r), max(0, iy - r)
    x2, y2 = min(w, ix + r), min(h, iy + r)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    return frame[y1:y2, x1:x2].copy()


def _crop_gray(gray: Optional[np.ndarray], cx: float, cy: float, r: int) -> Optional[np.ndarray]:
    if gray is None or gray.size == 0:
        return None
    h, w = gray.shape[:2]
    ix, iy = int(round(cx)), int(round(cy))
    x1, y1 = max(0, ix - r), max(0, iy - r)
    x2, y2 = min(w, ix + r), min(h, iy + r)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return gray[y1:y2, x1:x2]


def _crop_zone_gray(gray: Optional[np.ndarray], zone) -> Optional[np.ndarray]:
    if gray is None or gray.size == 0 or zone is None:
        return None
    h, w = gray.shape[:2]
    x1 = max(0, min(w, int(zone.x1)))
    y1 = max(0, min(h, int(zone.y1)))
    x2 = max(0, min(w, int(zone.x2)))
    y2 = max(0, min(h, int(zone.y2)))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    return gray[y1:y2, x1:x2]


def _select_active_wrist(
    candidates: List[Tuple[float, float, float, int, str]],
    worker_id: Optional[int],
    last_pos: Optional[Tuple[float, float]],
    side: Optional[str] = None,
) -> Optional[Tuple[float, float, float, int, str]]:
    """Select a single wrist candidate; never return a scalar/list mismatch."""
    if not candidates:
        return None

    valid = []
    for w in candidates:
        if not isinstance(w, (tuple, list)) or len(w) < 5:
            continue
        try:
            valid.append((float(w[0]), float(w[1]), float(w[2]), int(w[3]), str(w[4])))
        except (TypeError, ValueError):
            continue
    if not valid:
        return None

    pool = valid
    if worker_id is not None:
        same_worker = [w for w in pool if w[3] == worker_id]
        if same_worker:
            pool = same_worker

    if side:
        same_side = [w for w in pool if w[4] == side]
        if same_side:
            pool = same_side

    if last_pos is not None:
        return min(
            pool,
            key=lambda w: (
                float(np.hypot(w[0] - last_pos[0], w[1] - last_pos[1])),
                -w[2],
            ),
        )
    return max(pool, key=lambda w: w[2])


def _payload_motion_score(
    prev_gray: Optional[np.ndarray],
    curr_gray: Optional[np.ndarray],
    prev_pos: Optional[Tuple[float, float]],
    curr_pos: Optional[Tuple[float, float]],
    params: CounterParams,
) -> float:
    """Supporting evidence: local image motion that follows wrist travel."""
    if prev_gray is None or curr_gray is None or prev_pos is None or curr_pos is None:
        return 0.0

    dx = float(curr_pos[0] - prev_pos[0])
    dy = float(curr_pos[1] - prev_pos[1])
    travel = float(np.hypot(dx, dy))
    if travel < 2.5:
        return 0.0

    radius = max(28, int(params.payload_radius))
    inner = max(6, min(int(params.payload_inner_radius), radius - 8))
    cx = 0.5 * (prev_pos[0] + curr_pos[0])
    cy = 0.5 * (prev_pos[1] + curr_pos[1])

    p = _crop_gray(prev_gray, cx, cy, radius)
    c = _crop_gray(curr_gray, cx, cy, radius)
    if p is None or c is None or p.shape != c.shape:
        return 0.0

    p = cv2.GaussianBlur(p, (5, 5), 0)
    c = cv2.GaussianBlur(c, (5, 5), 0)

    try:
        flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    except cv2.error:
        return 0.0

    h, w = p.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    pcx, pcy = w / 2.0, h / 2.0
    rr = np.sqrt((xx - pcx) ** 2 + (yy - pcy) ** 2)
    mask = (rr >= inner) & (rr <= max(inner + 5, radius - 5))
    if not np.any(mask):
        return 0.0

    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)
    ux, uy = dx / travel, dy / travel
    projection = fx * ux + fy * uy
    cos_sim = projection / (mag + 1e-6)

    moving = mask & (mag >= float(params.payload_motion_thresh))
    aligned = moving & (cos_sim >= float(params.payload_direction_cos))
    aligned_ratio = float(np.count_nonzero(aligned)) / max(1, int(np.count_nonzero(mask)))

    motion_score = np.clip(aligned_ratio / 0.14, 0.0, 1.0)
    return float(motion_score)


def _source_change_score(
    baseline: Optional[np.ndarray],
    current: Optional[np.ndarray],
    local_point_xy: Optional[Tuple[float, float]],
    zone_xyxy: Optional[Tuple[int, int, int, int]],
    local_radius: int,
) -> float:
    """
    Compare pickup source before/after the hand leaves.

    Scores are based on image structure/appearance change, not a fixed "black
    pixels = fabric" rule. A local comparison near the pickup wrist is combined
    with a whole-source comparison.
    """
    if baseline is None or current is None:
        return 0.0
    if baseline.size == 0 or current.size == 0:
        return 0.0

    def prep(a: np.ndarray) -> Optional[np.ndarray]:
        if a.shape != current.shape:
            try:
                a = cv2.resize(a, (current.shape[1], current.shape[0]), interpolation=cv2.INTER_AREA)
            except cv2.error:
                return None
        return cv2.GaussianBlur(a, (5, 5), 0)

    b = prep(baseline)
    c = prep(current)
    if b is None or c is None or b.shape != c.shape:
        return 0.0

    def score_pair(xb: np.ndarray, xc: np.ndarray) -> float:
        diff = cv2.absdiff(xb, xc).astype(np.float32)
        mean_norm = float(np.mean(diff)) / 255.0
        changed = float(np.mean(diff >= 18.0))

        # Helpful in this scene: a removed dark fabric patch reduces dark area.
        db = float(np.mean(xb <= 115.0))
        dc = float(np.mean(xc <= 115.0))
        dark_reduction = max(0.0, db - dc)

        s1 = float(np.clip(mean_norm / 0.075, 0.0, 1.0))
        s2 = float(np.clip(changed / 0.10, 0.0, 1.0))
        s3 = float(np.clip(dark_reduction / 0.08, 0.0, 1.0))
        return float(np.clip(0.48 * s1 + 0.37 * s2 + 0.15 * s3, 0.0, 1.0))

    whole = score_pair(b, c)

    local = 0.0
    if local_point_xy is not None and zone_xyxy is not None:
        zx1, zy1, _, _ = zone_xyxy
        lx = int(round(local_point_xy[0] - zx1))
        ly = int(round(local_point_xy[1] - zy1))
        r = max(30, int(local_radius))
        x1, y1 = max(0, lx - r), max(0, ly - r)
        x2, y2 = min(b.shape[1], lx + r), min(b.shape[0], ly + r)
        if x2 - x1 >= 20 and y2 - y1 >= 20:
            local = score_pair(b[y1:y2, x1:x2], c[y1:y2, x1:x2])

    return float(np.clip(max(whole, local), 0.0, 1.0))


class PieceCounter:
    """Fixed-station machine-centric counter with an optional pickup evidence gate."""

    def __init__(self, params: Optional[CounterParams] = None):
        self.p = params or CounterParams()
        self.machines: Dict[int, MachineCycleTracker] = {}
        self._ema: Dict[Tuple[int, str], Tuple[float, float]] = {}

    def get_tracker(self, mzid: int, name: str = "") -> MachineCycleTracker:
        if mzid not in self.machines:
            self.machines[mzid] = MachineCycleTracker(
                mzid=mzid,
                name=name or f"Machine-{mzid}",
                src_deb=_Debounce(self.p.kp_gap),
                dst_deb=_Debounce(self.p.kp_gap),
            )
        return self.machines[mzid]

    def smooth_wrists(self, tid: int, lw, rw) -> Dict[str, Tuple[float, float]]:
        out: Dict[str, Tuple[float, float]] = {}
        for side, w in (("L", lw), ("R", rw)):
            key = (int(tid), side)
            if w is None:
                self._ema.pop(key, None)
                continue
            prev = self._ema.get(key)
            a = float(self.p.ema_alpha)
            cur = (
                (float(w[0]), float(w[1]))
                if prev is None
                else (
                    a * float(w[0]) + (1.0 - a) * prev[0],
                    a * float(w[1]) + (1.0 - a) * prev[1],
                )
            )
            self._ema[key] = cur
            out[side] = cur
        return out

    def _update_source_baseline(self, m: MachineCycleTracker, gray_frame: Optional[np.ndarray], src_zone) -> None:
        if gray_frame is None or src_zone is None:
            return
        patch = _crop_zone_gray(gray_frame, src_zone)
        if patch is None:
            return
        if m.src_baseline_gray is None or m.src_baseline_gray.shape != patch.shape:
            m.src_baseline_gray = patch.copy()
        else:
            # Slow EMA adapts to illumination while remaining stable to short events.
            m.src_baseline_gray = cv2.addWeighted(m.src_baseline_gray, 0.90, patch, 0.10, 0)
        m.src_baseline_zone = (int(src_zone.x1), int(src_zone.y1), int(src_zone.x2), int(src_zone.y2))

    def _update_source_change(self, m: MachineCycleTracker, gray_frame: Optional[np.ndarray], src_zone, now: float) -> None:
        if not self.p.payload_check or gray_frame is None or src_zone is None:
            return
        if m.src_baseline_gray is None:
            return
        if now < m.src_change_ready_time:
            return
        current = _crop_zone_gray(gray_frame, src_zone)
        if current is None:
            return
        zone_xyxy = m.src_baseline_zone or (src_zone.x1, src_zone.y1, src_zone.x2, src_zone.y2)
        s = _source_change_score(
            m.src_baseline_gray,
            current,
            m.pickup_src_point,
            zone_xyxy,
            self.p.payload_source_local_radius,
        )
        m.src_change_last = s
        m.src_change_peak = max(m.src_change_peak, s)
        m.src_change_checked = True

    def _update_payload_motion(
        self,
        m: MachineCycleTracker,
        gray_frame: Optional[np.ndarray],
        candidates,
        now: float,
    ) -> None:
        if not self.p.payload_check or gray_frame is None or not candidates:
            return
        aw = _select_active_wrist(
            list(candidates),
            m.active_worker_id,
            m.last_wrist_pos,
            m.active_wrist_side,
        )
        if aw is None:
            return

        current_pos = (aw[0], aw[1])
        score = _payload_motion_score(
            m.payload_prev_gray,
            gray_frame,
            m.payload_prev_wrist_pos,
            current_pos,
            self.p,
        )
        if m.payload_prev_gray is not None and m.payload_prev_wrist_pos is not None:
            m.payload_scores.append(score)
            m.payload_last_score = score
            m.payload_peak_score = max(m.payload_peak_score, score)
            if score >= self.p.payload_score_threshold:
                m.payload_positive_samples += 1

        m.last_wrist_pos = current_pos
        m.active_worker_id = aw[3] if aw[3] is not None else m.active_worker_id
        m.active_wrist_side = aw[4]
        m.wrist_confs.append(aw[2])
        m.payload_prev_gray = gray_frame.copy()
        m.payload_prev_wrist_pos = current_pos

    def _payload_ok(self, m: MachineCycleTracker) -> Tuple[bool, float, str]:
        if not self.p.payload_check:
            return True, 1.0, "disabled"

        # Primary: source changed after the pickup hand left.
        source_ok = m.src_change_peak >= self.p.payload_source_change_threshold

        # Secondary: strong local temporal motion, but require at least a little
        # source change so ordinary empty-hand motion is not enough by itself.
        motion_ok = (
            m.payload_peak_score >= self.p.payload_motion_accept_threshold
            and m.payload_positive_samples >= max(1, self.p.payload_min_samples)
            and m.src_change_peak >= 0.06
        )

        if source_ok and motion_ok:
            return True, max(m.src_change_peak, m.payload_peak_score), "source+motion"
        if source_ok:
            return True, m.src_change_peak, "source-change"
        if motion_ok:
            return True, m.payload_peak_score, "wrist-motion+source"

        combined = max(
            0.65 * m.src_change_peak + 0.35 * m.payload_peak_score,
            m.src_change_peak,
            m.payload_peak_score * 0.75,
        )
        return False, float(np.clip(combined, 0.0, 1.0)), "insufficient"

    def update_machine(
        self,
        mzid: int,
        name: str,
        now: float,
        wrists_info: List[Tuple[float, float, float, int, str]],
        src_zone,
        dst_zone,
        frame: Optional[np.ndarray] = None,
        gray_frame: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        p = self.p
        m = self.get_tracker(mzid, name)

        if gray_frame is None and frame is not None:
            try:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                gray_frame = None

        if m.state == MachineState.COOLDOWN:
            if now >= m.cooldown_until:
                m.reset_to_idle()
            else:
                return None

        src_wrists = [w for w in wrists_info if _inside(src_zone, w[0], w[1], p.zone_margin)]
        dst_wrists = [w for w in wrists_info if _inside(dst_zone, w[0], w[1], p.zone_margin)]
        in_src = bool(src_wrists)
        in_dst = bool(dst_wrists)

        m.src_deb.update(in_src, now)
        m.dst_deb.update(in_dst, now)

        # Keep a background reference while nobody is actively interacting with SRC.
        if m.state == MachineState.IDLE and not in_src:
            self._update_source_baseline(m, gray_frame, src_zone)

        active = _select_active_wrist(
            wrists_info,
            m.active_worker_id,
            m.last_wrist_pos,
            m.active_wrist_side,
        ) if m.state != MachineState.IDLE else None

        # IDLE -> HAND_IN_SRC
        if m.state == MachineState.IDLE:
            if m.src_deb.dwell(now) >= p.src_dwell:
                best = max(src_wrists, key=lambda w: w[2]) if src_wrists else None
                if best is None:
                    return None

                m.state = MachineState.HAND_IN_SRC
                m.t_pick = now
                m.active_worker_id = int(best[3]) if best[3] is not None else None
                m.active_wrist_side = best[4]
                m.last_wrist_pos = (best[0], best[1])
                m.pickup_src_point = m.last_wrist_pos
                m.wrist_confs = [float(best[2])]
                m.reset_cycle_evidence()
                m.active_worker_id = int(best[3]) if best[3] is not None else None
                m.active_wrist_side = best[4]
                m.last_wrist_pos = (best[0], best[1])
                m.pickup_src_point = m.last_wrist_pos
                m.src_change_ready_time = now + max(0.05, p.payload_source_settle_sec)
                m.payload_prev_gray = gray_frame.copy() if gray_frame is not None else None
                m.payload_prev_wrist_pos = m.last_wrist_pos
                self._log(f"[PICK] {m.name}: Hand in SRC (Worker ID: {m.active_worker_id})")
            return None

        # Timeout
        if m.state in (MachineState.HAND_IN_SRC, MachineState.IN_TRANSIT, MachineState.HAND_IN_DST):
            if (now - m.t_pick) > p.max_transit:
                transit_dur = round(now - m.t_pick, 2)
                self._log(f"[TIMEOUT] {m.name}: transit timeout ({transit_dur}s > {p.max_transit}s)")
                event = {
                    "event": "timeout",
                    "machine": m.name,
                    "machine_id": mzid,
                    "transit_sec": transit_dur,
                    "payload_score": round(max(m.payload_peak_score, m.src_change_peak), 3),
                    "payload_confirmed": False,
                    "payload_samples": int(m.payload_positive_samples),
                    "payload_peak": round(m.payload_peak_score, 3),
                    "source_change": round(m.src_change_peak, 3),
                }
                m.reset_to_idle()
                return event

        # HAND_IN_SRC
        if m.state == MachineState.HAND_IN_SRC:
            if active is not None:
                m.last_wrist_pos = (active[0], active[1])
                m.active_worker_id = active[3]
                m.active_wrist_side = active[4]
                m.wrist_confs.append(active[2])

            # If zone geometry overlaps, allow direct arrival at DST.
            if in_dst:
                self._update_payload_motion(m, gray_frame, (active,) if active else dst_wrists, now)
                self._update_source_change(m, gray_frame, src_zone, now)
                m.state = MachineState.HAND_IN_DST
                m.dst_deb.reset()
                m.dst_deb.update(True, now)
            elif m.src_deb.away(now) > p.kp_gap:
                m.state = MachineState.IN_TRANSIT
                m.t_src_exit = now
                m.src_change_ready_time = now + max(0.05, p.payload_source_settle_sec)
                m.payload_prev_gray = gray_frame.copy() if gray_frame is not None else m.payload_prev_gray
                m.payload_prev_wrist_pos = m.last_wrist_pos
                self._update_source_change(m, gray_frame, src_zone, now)
            else:
                # Capture motion while hand is still in the pickup area; this is only support evidence.
                if active is not None:
                    self._update_payload_motion(m, gray_frame, (active,), now)
            return None

        # IN_TRANSIT
        if m.state == MachineState.IN_TRANSIT:
            self._update_source_change(m, gray_frame, src_zone, now)
            if wrists_info:
                self._update_payload_motion(m, gray_frame, wrists_info, now)

            # Returned to SRC: treat as a failed/aborted attempt and re-arm only after dwell.
            if m.src_deb.dwell(now) >= p.src_dwell:
                m.state = MachineState.HAND_IN_SRC
                m.t_pick = now
                m.pickup_src_point = _select_active_wrist(
                    src_wrists, m.active_worker_id, m.last_wrist_pos, m.active_wrist_side
                )
                if m.pickup_src_point is not None:
                    m.pickup_src_point = (m.pickup_src_point[0], m.pickup_src_point[1])
                m.payload_scores.clear()
                m.payload_confirmed = False
                m.payload_positive_samples = 0
                m.payload_peak_score = 0.0
                m.payload_last_score = 0.0
                m.src_change_peak = 0.0
                m.src_change_last = 0.0
                m.payload_prev_gray = gray_frame.copy() if gray_frame is not None else None
                m.payload_prev_wrist_pos = m.last_wrist_pos
                m.src_change_ready_time = now + max(0.05, p.payload_source_settle_sec)
                return None

            if in_dst:
                best = _select_active_wrist(dst_wrists, m.active_worker_id, m.last_wrist_pos, m.active_wrist_side)
                if best is not None:
                    m.last_wrist_pos = (best[0], best[1])
                    m.active_worker_id = best[3]
                    m.active_wrist_side = best[4]
                    m.wrist_confs.append(best[2])
                    self._update_payload_motion(m, gray_frame, (best,), now)
                m.state = MachineState.HAND_IN_DST
                m.dst_deb.reset()
                m.dst_deb.update(True, now)
            return None

        # HAND_IN_DST
        if m.state == MachineState.HAND_IN_DST:
            best = _select_active_wrist(dst_wrists, m.active_worker_id, m.last_wrist_pos, m.active_wrist_side)
            if best is not None:
                m.last_wrist_pos = (best[0], best[1])
                m.active_worker_id = best[3]
                m.active_wrist_side = best[4]
                m.wrist_confs.append(best[2])
                self._update_payload_motion(m, gray_frame, (best,), now)

            self._update_source_change(m, gray_frame, src_zone, now)

            if m.dst_deb.dwell(now) >= p.dst_dwell:
                transit_sec = now - m.t_pick
                payload_ok, payload_score, payload_method = self._payload_ok(m)

                hand_crop = None
                if frame is not None and m.last_wrist_pos is not None:
                    hand_crop = _crop_wrist(frame, m.last_wrist_pos[0], m.last_wrist_pos[1], p.hand_crop_radius)
                m.hand_crop = hand_crop

                if not payload_ok:
                    self._log(
                        f"[REJECT] {m.name}: no fabric pickup evidence "
                        f"(payload={payload_score:.2f} source={m.src_change_peak:.2f} "
                        f"motion={m.payload_peak_score:.2f} samples={m.payload_positive_samples})"
                    )
                    event = {
                        "event": "reject_payload",
                        "machine_id": mzid,
                        "machine": m.name,
                        "cycle_num": m.cycle_count,
                        "worker_id": m.active_worker_id or 0,
                        "transit_sec": round(transit_sec, 2),
                        "confidence": round(payload_score, 2),
                        "payload_score": round(payload_score, 3),
                        "payload_confirmed": False,
                        "payload_samples": int(m.payload_positive_samples),
                        "payload_peak": round(m.payload_peak_score, 3),
                        "source_change": round(m.src_change_peak, 3),
                        "payload_method": payload_method,
                        "state_str": "SRC -> DST (REJECTED: NO FABRIC)",
                        "wrist_pos": m.last_wrist_pos,
                        "hand_crop": hand_crop,
                        "timestamp_sec": round(now, 2),
                    }
                    m.state = MachineState.COOLDOWN
                    m.cooldown_until = now + p.cooldown
                    m.src_deb.reset()
                    m.dst_deb.reset()
                    return event

                m.cycle_count += 1
                m.t_place = now
                avg_wrist_conf = float(np.mean(m.wrist_confs)) if m.wrist_confs else 0.80
                dwell_score = min(1.0, m.dst_deb.dwell(now) / max(0.1, p.dst_dwell))
                transit_score = 1.0 if 0.5 <= transit_sec <= 4.5 else max(0.6, 1.0 - abs(transit_sec - 2.5) / 5.0)
                conf = 0.40 * avg_wrist_conf + 0.15 * dwell_score + 0.15 * transit_score + 0.30 * payload_score
                conf = float(np.clip(conf, 0.65, 0.99))

                self._log(
                    f"[CYCLE] {m.name}: +1 (total {m.cycle_count}) | "
                    f"Transit: {transit_sec:.2f}s | Conf: {conf:.2f} | "
                    f"Payload: {payload_score:.2f} ({payload_method}) | Op: ID {m.active_worker_id}"
                )

                event = {
                    "event": "count",
                    "machine_id": mzid,
                    "machine": m.name,
                    "cycle_num": m.cycle_count,
                    "worker_id": m.active_worker_id or 0,
                    "transit_sec": round(transit_sec, 2),
                    "confidence": round(conf, 2),
                    "payload_score": round(payload_score, 3),
                    "payload_confirmed": True,
                    "payload_samples": int(m.payload_positive_samples),
                    "payload_peak": round(m.payload_peak_score, 3),
                    "source_change": round(m.src_change_peak, 3),
                    "payload_method": payload_method,
                    "state_str": "SRC -> DST (CYCLE)",
                    "wrist_pos": m.last_wrist_pos,
                    "hand_crop": hand_crop,
                    "timestamp_sec": round(now, 2),
                }

                m.state = MachineState.COOLDOWN
                m.cooldown_until = now + p.cooldown
                m.src_deb.reset()
                m.dst_deb.reset()
                return event

            if m.dst_deb.away(now) > p.kp_gap:
                m.state = MachineState.IN_TRANSIT
            return None

        return None

    def update(
        self,
        tid: int,
        mzid: int,
        now: float,
        wrists: List[Tuple[float, float]],
        src_zone,
        dst_zone,
        frame: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        wrists_info = [
            (float(w[0]), float(w[1]), 0.85, int(tid), f"W{i}")
            for i, w in enumerate(wrists)
        ]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        return self.update_machine(
            mzid, f"Machine-{mzid}", now, wrists_info, src_zone, dst_zone, frame, gray
        )

    def observe_src(self, mzid: int, frame: np.ndarray, src_zone, occupied: bool):
        """Legacy compatibility hook."""
        m = self.get_tracker(mzid, f"Machine-{mzid}")
        if not occupied:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                gray = None
            self._update_source_baseline(m, gray, src_zone)

    def _log(self, msg: str):
        if self.p.verbose:
            print("  " + msg)


def render_machine_audit_snapshot(
    frame: np.ndarray,
    output_path: str,
    machine_name: str,
    machine_zone: Any,
    src_zone: Any,
    dst_zone: Any,
    cycle_num: int,
    state_str: str,
    confidence: float,
    transit_sec: float,
    timestamp_str: str,
    worker_id: Optional[int] = None,
    wrists: Optional[List[Tuple[float, float]]] = None,
    hand_crop: Optional[np.ndarray] = None,
    payload_score: Optional[float] = None,
    payload_confirmed: Optional[bool] = None,
    source_change: Optional[float] = None,
    payload_method: Optional[str] = None,
):
    """Industrial-style audit frame compatible with person_tracker.py."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    if machine_zone is not None:
        cv2.rectangle(canvas, (machine_zone.x1, machine_zone.y1), (machine_zone.x2, machine_zone.y2), (255, 200, 0), 2)
        cv2.putText(canvas, machine_name, (machine_zone.x1 + 6, machine_zone.y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

    if src_zone is not None:
        cv2.rectangle(overlay, (src_zone.x1, src_zone.y1), (src_zone.x2, src_zone.y2), (0, 165, 255), -1)
        cv2.rectangle(canvas, (src_zone.x1, src_zone.y1), (src_zone.x2, src_zone.y2), (0, 200, 255), 3)
        cv2.putText(canvas, "SRC [PICKUP]", (src_zone.x1 + 4, max(20, src_zone.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

    if dst_zone is not None:
        cv2.rectangle(overlay, (dst_zone.x1, dst_zone.y1), (dst_zone.x2, dst_zone.y2), (0, 200, 50), -1)
        cv2.rectangle(canvas, (dst_zone.x1, dst_zone.y1), (dst_zone.x2, dst_zone.y2), (50, 255, 50), 3)
        cv2.putText(canvas, "DST [PLACEMENT]", (dst_zone.x1 + 4, max(20, dst_zone.y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

    if src_zone is not None and dst_zone is not None:
        src_c = ((src_zone.x1 + src_zone.x2) // 2, (src_zone.y1 + src_zone.y2) // 2)
        dst_c = ((dst_zone.x1 + dst_zone.x2) // 2, (dst_zone.y1 + dst_zone.y2) // 2)
        cv2.arrowedLine(canvas, src_c, dst_c, (0, 255, 255), 2, tipLength=0.15)

    if wrists:
        for wx, wy in wrists:
            pt = (int(round(wx)), int(round(wy)))
            cv2.circle(canvas, pt, 10, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, pt, 4, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Wrist", (pt[0] + 12, pt[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    card_w, card_h = 455, 255
    card_x, card_y = 20, 20
    card_ov = canvas.copy()
    cv2.rectangle(card_ov, (card_x, card_y), (card_x + card_w, card_y + card_h), (20, 20, 20), -1)
    cv2.addWeighted(card_ov, 0.82, canvas, 0.18, 0, canvas)
    cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), (0, 220, 255), 2)

    payload_txt = "Payload: N/A"
    payload_color = (180, 180, 180)
    if payload_confirmed is not None or payload_score is not None:
        payload_txt = f"Payload: {'YES' if payload_confirmed else 'NO'}  ({payload_score or 0.0:.2f})"
        payload_color = (50, 255, 50) if payload_confirmed else (80, 180, 255)

    lines = [
        (f"{machine_name.upper()} AUDIT", (0, 255, 255), 0.7, 2),
        (f"Cycle Count: #{cycle_num}", (255, 255, 255), 0.55, 1),
        (f"State: {state_str}", (100, 255, 100), 0.55, 1),
        (f"Operator: ID {worker_id if worker_id is not None else 'N/A'}", (220, 220, 220), 0.55, 1),
        (f"Transit Time: {transit_sec:.2f}s", (220, 220, 220), 0.55, 1),
        (f"Confidence: {confidence * 100:.1f}%", (50, 255, 50), 0.55, 2),
        (payload_txt, payload_color, 0.55, 2),
        (f"SRC change: {(source_change if source_change is not None else 0.0):.2f}", (220, 220, 220), 0.50, 1),
        (f"Method: {payload_method or 'N/A'}", (180, 180, 180), 0.48, 1),
        (f"Timestamp: {timestamp_str}", (180, 180, 180), 0.50, 1),
    ]
    ty = card_y + 29
    for text, color, scale, thick in lines:
        cv2.putText(canvas, text, (card_x + 16, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        ty += 23

    if hand_crop is not None and hand_crop.size > 0:
        crop_sz = 150
        disp_crop = cv2.resize(hand_crop, (crop_sz, crop_sz), interpolation=cv2.INTER_LINEAR)
        ix = w - crop_sz - 30
        iy = 42
        cv2.rectangle(canvas, (ix - 6, iy - 24), (ix + crop_sz + 6, iy + crop_sz + 6), (20, 20, 20), -1)
        cv2.rectangle(canvas, (ix - 6, iy - 24), (ix + crop_sz + 6, iy + crop_sz + 6), (50, 255, 50), 2)
        cv2.putText(canvas, "HAND / FABRIC CROP", (ix - 2, iy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 255, 50), 1, cv2.LINE_AA)
        if iy + crop_sz <= h and ix + crop_sz <= w:
            canvas[iy:iy + crop_sz, ix:ix + crop_sz] = disp_crop

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, canvas)
