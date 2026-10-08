"""
Advanced Person Detection & Tracking  (v2 – Hand / Pose Edition)
=================================================================
Model    : YOLOv8m-pose  (person detection + 17 COCO keypoints)
Tracker  : ByteTrack (built into Ultralytics)

Features:
  - Interactive ROI zone drawing per machine (drag to draw on first frame)
  - Source / Destination box drawing per machine (for piece-work counting)
  - Hand (wrist-keypoint) tracking for Working / Idle status
  - Piece-work counting: Source → Destination = 1 unit; must return to Source before next count
  - 1-person-per-zone crowding alert
  - Re-ID ghost buffer: lost ID remapped to same canonical ID when
    re-detected within GHOST_BUFFER_SEC (default 3 s)
  - Full HUD: per-zone counters, piece counts, alert log, FPS, track stats
  - End-of-run piece-work summary table
  - Temporal fabric-payload gate: repeated local wrist-crop motion evidence is required before a cycle counts

Controls – Machine-zone drawing phase:
  Drag (LMB)  : Draw a zone rectangle
  D           : Delete last drawn zone
  C           : Clear all zones
  S / Enter   : Confirm and proceed
  Q / Esc     : Skip zones

Controls – Source / Dest box drawing phase (per machine):
  Drag (LMB)  : Draw the box
  D           : Delete / redo current box
  S / Enter   : Confirm current box, proceed to next
  Q / Esc     : Skip ALL remaining source/dest boxes

Controls – Tracking phase:
  Q / Esc     : Quit

Usage:
    python person_tracker.py [--video PATH] [--conf 0.25]
                             [--ghost-sec 3] [--idle-sec 10] [--no-show]
"""

import argparse
import os
import sys
import time
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import json
import cv2
import numpy as np
from ultralytics import YOLO

from gemma_analyzer import generate_gemma_efficiency_report, check_ollama_available
from piece_counter import (
    PieceCounter,
    CounterParams,
    MachineState,
    render_machine_audit_snapshot,
)


# ═══════════════════════════════════════════════════════════
#  TUNEABLE CONSTANTS
# ═══════════════════════════════════════════════════════════
GHOST_BUFFER_SEC    = 3.0    # seconds to keep lost track IDs for re-ID
IDLE_ALERT_SEC      = 10.0   # person idle > this → alert
WORK_BUFFER_SEC     = 3.0    # buffer (sec) to hold 'Working' status after motion ceases
WRIST_MOTION_THRESH = 8.0    # wrist displacement (px) over sampling window → Working
WRIST_SMOOTH        = 8      # number of sampled wrist snapshots to smooth over
WRIST_CONF_MIN      = 0.3    # min keypoint confidence to consider a wrist visible
FLOW_SKIP_MS        = 80     # gap (ms) between wrist snapshots for motion calc
TRAIL_LEN           = 30     # motion trail length (frames)
REID_IOU_THRESH     = 0.15   # min IoU for re-ID candidate
REID_DIST_THRESH    = 130    # max centre-distance (px) for re-ID candidate
PIECE_ACTION_BUFFER_SEC = 1.5  # minimum seconds between piece-work actions

# COCO keypoint indices (YOLOv8-pose outputs 17 keypoints)
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10


# ═══════════════════════════════════════════════════════════
#  DATA CLASSES
# ═══════════════════════════════════════════════════════════
@dataclass
class Zone:
    zone_id: int
    name: str
    x1: int
    y1: int
    x2: int
    y2: int

    def contains_center(self, cx: int, cy: int) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def contains_point(self, px: float, py: float) -> bool:
        return self.x1 <= px <= self.x2 and self.y1 <= py <= self.y2


class PickState(Enum):
    WAIT_SOURCE       = "wait_source"
    WAIT_DESTINATION  = "wait_destination"
    WAIT_SOURCE_EXIT  = "wait_source_exit"


@dataclass
class PickPlaceConfig:
    """Source and destination boxes for one machine zone."""
    machine_zone: Zone
    source: Optional[Zone] = None
    destination: Optional[Zone] = None


@dataclass
class PieceWorkState:
    state: PickState = PickState.WAIT_SOURCE
    count: int = 0
    last_event_time: float = 0.0


@dataclass
class GhostTrack:
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    zone_ids: List[int]
    lost_time: float
    saved_state: Optional["TrackState"] = None


@dataclass
class TrackState:
    track_id: int
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    last_conf: float = 0.0
    zone_ids:  List[int] = field(default_factory=list)
    # Wrist positions – sampled every flow_skip frames for motion detection
    wrist_history: deque = field(default_factory=lambda: deque(maxlen=WRIST_SMOOTH))
    working: bool = False
    last_work_time: Optional[float] = None
    idle_since: Optional[float] = None
    # Current-frame wrist positions (updated every frame for piece-work)
    left_wrist:  Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    left_wrist_conf:  float = 0.0
    right_wrist_conf: float = 0.0
    # Piece-work state per machine-zone ID
    piece_work: Dict[int, PieceWorkState] = field(default_factory=dict)
    # Trail
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))
    last_seen: float = field(default_factory=time.time)

    @property
    def total_pieces(self) -> int:
        return sum(pw.count for pw in self.piece_work.values())


# ═══════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════
def id_color(tid: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(int(tid) * 6364136223846793005 + 1)
    r, g, b = rng.integers(80, 230, size=3).tolist()
    return (int(b), int(g), int(r))   # BGR

_SRC_COLOR  = (0, 220, 0)     # Green  – source / pickup
_DST_COLOR  = (0, 160, 255)   # Orange – destination / place


# ═══════════════════════════════════════════════════════════
#  GHOST / RE-ID TRACKER
# ═══════════════════════════════════════════════════════════
class GhostTracker:
    """
    Maintains a buffer of recently-lost track IDs (ghosts).
    When a brand-new tracker ID appears, we check whether it matches
    any ghost (same spatial position, within GHOST_BUFFER_SEC).
    If yes → remap new raw ID to the old canonical ID.
    """

    def __init__(self, buffer_sec: float = GHOST_BUFFER_SEC,
                 iou_thresh: float = REID_IOU_THRESH,
                 dist_thresh: float = REID_DIST_THRESH):
        self.buffer_sec  = buffer_sec
        self.iou_thresh  = iou_thresh
        self.dist_thresh = dist_thresh
        self.ghosts: Dict[int, GhostTrack] = {}
        # raw_id -> canonical_id  (persisted so the same raw_id keeps resolving)
        self.remap: Dict[int, int] = {}

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0

    def add_ghost(self, state: TrackState, current_time: Optional[float] = None):
        """Called when a track ID disappears from the current frame."""
        cx = (state.last_bbox[0] + state.last_bbox[2]) // 2
        cy = (state.last_bbox[1] + state.last_bbox[3]) // 2
        now = current_time if current_time is not None else time.time()
        self.ghosts[state.track_id] = GhostTrack(
            track_id=state.track_id,
            bbox=state.last_bbox,
            center=(cx, cy),
            zone_ids=state.zone_ids[:],
            lost_time=now,
            saved_state=state,
        )

    def pop_ghost_state(self, cid: int) -> Optional[TrackState]:
        """Retrieve and remove the saved TrackState for a re-identified ghost."""
        ghost = self.ghosts.pop(cid, None)
        return ghost.saved_state if ghost else None

    def resolve(self, raw_id: int, bbox, current_time: Optional[float] = None) -> int:
        """
        Return the canonical ID for raw_id.
        Checks existing remap first, then tries ghost matching.
        """
        # Already known remap?
        if raw_id in self.remap:
            return self.remap[raw_id]

        now = current_time if current_time is not None else time.time()
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2

        best_ghost: Optional[GhostTrack] = None
        best_score = -1.0

        for gid in list(self.ghosts):
            g = self.ghosts[gid]
            if now - g.lost_time > self.buffer_sec:
                del self.ghosts[gid]
                continue
            iou  = self._iou(bbox, g.bbox)
            dist = float(np.hypot(cx - g.center[0], cy - g.center[1]))
            if iou >= self.iou_thresh or dist <= self.dist_thresh:
                score = iou + max(0.0, 1.0 - dist / self.dist_thresh)
                if score > best_score:
                    best_score = score
                    best_ghost = g

        if best_ghost is not None:
            print(f"  [Re-ID] raw={raw_id} -> canonical={best_ghost.track_id}  "
                  f"(lost {now - best_ghost.lost_time:.1f}s ago)")
            self.remap[raw_id] = best_ghost.track_id
            return best_ghost.track_id

        return raw_id

    def cleanup_expired(self, current_time: Optional[float] = None):
        now = current_time if current_time is not None else time.time()
        for gid in [g for g in self.ghosts if now - self.ghosts[g].lost_time > self.buffer_sec]:
            self.ghosts.pop(gid, None)


# ═══════════════════════════════════════════════════════════
#  WRIST-BASED MOTION DETECTION
# ═══════════════════════════════════════════════════════════
def compute_wrist_motion(wrist_history: deque) -> float:
    """
    Average wrist displacement (px) across consecutive sampled snapshots.
    Each snapshot is a tuple:  (left_wrist_xy | None, right_wrist_xy | None)
    Uses the *maximum* of left/right displacement per step so that
    a single moving hand is sufficient to register 'Working'.
    """
    if len(wrist_history) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(1, len(wrist_history)):
        prev_lw, prev_rw = wrist_history[i - 1]
        curr_lw, curr_rw = wrist_history[i]
        disps: List[float] = []
        if prev_lw is not None and curr_lw is not None:
            disps.append(float(np.hypot(curr_lw[0] - prev_lw[0],
                                        curr_lw[1] - prev_lw[1])))
        if prev_rw is not None and curr_rw is not None:
            disps.append(float(np.hypot(curr_rw[0] - prev_rw[0],
                                        curr_rw[1] - prev_rw[1])))
        if disps:
            total += max(disps)
            count += 1
    return total / count if count > 0 else 0.0


# ═══════════════════════════════════════════════════════════
#  ROI ZONE DRAWING – Phase 1: Machine Zones
# ═══════════════════════════════════════════════════════════
def draw_zones_interactive(first_frame: np.ndarray) -> List[Zone]:
    """
    Display the first frame and let the user drag to define machine zones.
    Returns the confirmed list of Zone objects.
    """
    zones: List[Zone] = []
    drawing = False
    ix = iy = 0
    tmp_pt: Optional[Tuple] = None
    BASE = first_frame.copy()
    WIN  = "DRAW ZONES  |  Drag=draw  D=del last  C=clear  S/Enter=start  Q=skip"

    def _redraw(canvas_override=None):
        canvas = (canvas_override if canvas_override is not None else BASE).copy()
        # Instructions bar
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (canvas.shape[1], 52), (10, 10, 10), -1)
        cv2.addWeighted(ov, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(canvas,
                    "Drag to draw zone box  |  D=delete last  C=clear all  S/Enter=start  Q=skip zones",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 1, cv2.LINE_AA)
        # Confirmed zones
        for z in zones:
            col = id_color(z.zone_id)
            cv2.rectangle(canvas, (z.x1, z.y1), (z.x2, z.y2), col, 2)
            lbl = z.name
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            cv2.rectangle(canvas, (z.x1, z.y1 - th - 10), (z.x1 + tw + 8, z.y1), col, -1)
            cv2.putText(canvas, lbl, (z.x1 + 4, z.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        # In-progress rect
        if tmp_pt:
            cv2.rectangle(canvas, (ix, iy), tmp_pt, (0, 255, 255), 1)
        cv2.imshow(WIN, canvas)

    def _mouse(event, x, y, flags, param):
        nonlocal drawing, ix, iy, tmp_pt
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing, ix, iy = True, x, y
            tmp_pt = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            tmp_pt = (x, y)
            _redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            tmp_pt = None
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            if x2 - x1 > 20 and y2 - y1 > 20:
                zid = len(zones) + 1
                zones.append(Zone(zone_id=zid, name=f"Machine-{zid}",
                                   x1=x1, y1=y1, x2=x2, y2=y2))
            _redraw()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse)
    _redraw()

    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('s'), 13):      # S or Enter → confirm
            break
        elif key in (ord('q'), 27):    # Q / Esc → skip zones
            zones.clear()
            break
        elif key == ord('d') and zones:
            zones.pop()
            _redraw()
        elif key == ord('c'):
            zones.clear()
            _redraw()

    cv2.destroyWindow(WIN)
    return zones


# ═══════════════════════════════════════════════════════════
#  ROI ZONE DRAWING – Phase 2: Source / Destination Boxes
# ═══════════════════════════════════════════════════════════
def _draw_one_box(first_frame: np.ndarray,
                  machine_zones: List[Zone],
                  drawn_boxes: List[dict],
                  prompt: str,
                  box_color: Tuple[int, int, int]) -> Optional[Zone]:
    """
    Let the user draw exactly ONE box on the frame.
    Returns a Zone or None if the user presses Q / Esc.
    """
    result: List[Optional[Zone]] = [None]
    drawing  = [False]
    start_xy = [0, 0]
    tmp_pt:   List[Optional[Tuple]] = [None]
    BASE = first_frame.copy()
    WIN  = "DRAW BOX"

    def _redraw():
        canvas = BASE.copy()
        # Machine zones (translucent)
        for z in machine_zones:
            col = id_color(z.zone_id)
            ov = canvas.copy()
            cv2.rectangle(ov, (z.x1, z.y1), (z.x2, z.y2), col, -1)
            cv2.addWeighted(ov, 0.10, canvas, 0.90, 0, canvas)
            cv2.rectangle(canvas, (z.x1, z.y1), (z.x2, z.y2), col, 2)
            cv2.putText(canvas, z.name, (z.x1 + 4, z.y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        # Previously drawn source / dest boxes
        for eb in drawn_boxes:
            bz = eb["zone"]
            cv2.rectangle(canvas, (bz.x1, bz.y1), (bz.x2, bz.y2), eb["color"], 2)
            (tw, th), _ = cv2.getTextSize(eb["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            cv2.rectangle(canvas, (bz.x1, bz.y1 - th - 6), (bz.x1 + tw + 4, bz.y1), eb["color"], -1)
            cv2.putText(canvas, eb["label"], (bz.x1 + 2, bz.y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        # Current confirmed box
        if result[0] is not None:
            r = result[0]
            cv2.rectangle(canvas, (r.x1, r.y1), (r.x2, r.y2), box_color, 2)
        # In-progress drag rect
        if tmp_pt[0] is not None:
            cv2.rectangle(canvas, (start_xy[0], start_xy[1]), tmp_pt[0], box_color, 1)
        # Instruction bar
        ov2 = canvas.copy()
        cv2.rectangle(ov2, (0, 0), (canvas.shape[1], 52), (10, 10, 10), -1)
        cv2.addWeighted(ov2, 0.65, canvas, 0.35, 0, canvas)
        cv2.putText(canvas,
                    f"{prompt}  |  S/Enter=confirm  D=redo  Q=skip all",
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 220, 255), 1, cv2.LINE_AA)
        cv2.imshow(WIN, canvas)

    def _mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            start_xy[0], start_xy[1] = x, y
            tmp_pt[0] = None
        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            tmp_pt[0] = (x, y)
            _redraw()
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            tmp_pt[0] = None
            x1, y1 = min(start_xy[0], x), min(start_xy[1], y)
            x2, y2 = max(start_xy[0], x), max(start_xy[1], y)
            if x2 - x1 > 15 and y2 - y1 > 15:
                result[0] = Zone(zone_id=0, name="", x1=x1, y1=y1, x2=x2, y2=y2)
            _redraw()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, _mouse)
    _redraw()

    skip_all = False
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('s'), 13):   # Confirm
            break
        elif key in (ord('q'), 27): # Skip ALL remaining
            result[0] = None
            skip_all = True
            break
        elif key == ord('d'):       # Redo
            result[0] = None
            _redraw()

    cv2.destroyWindow(WIN)
    return result[0] if not skip_all else "__SKIP__"  # sentinel


def draw_source_dest_interactive(first_frame: np.ndarray,
                                 machine_zones: List[Zone]
                                 ) -> Dict[int, PickPlaceConfig]:
    """
    Phase 2 – for each machine zone, let the user draw a Source (pickup)
    box and a Destination (place) box.
    Returns {machine_zone_id: PickPlaceConfig}.
    """
    configs: Dict[int, PickPlaceConfig] = {}
    if not machine_zones:
        return configs

    drawn_boxes: List[dict] = []   # running list for visual context

    for mz in machine_zones:
        # --- Source box ---
        src = _draw_one_box(
            first_frame, machine_zones, drawn_boxes,
            prompt=f"Draw SOURCE (pickup) box for {mz.name}",
            box_color=_SRC_COLOR,
        )
        if src == "__SKIP__":
            # User pressed Q – skip all remaining
            configs[mz.zone_id] = PickPlaceConfig(machine_zone=mz)
            for rmz in machine_zones:
                configs.setdefault(rmz.zone_id, PickPlaceConfig(machine_zone=rmz))
            return configs
        if src is not None:
            src.name = f"{mz.name} Source"
            src.zone_id = mz.zone_id * 100 + 1
            drawn_boxes.append({"zone": src, "color": _SRC_COLOR,
                                "label": f"SRC: {mz.name}"})

        # --- Destination box ---
        dst = _draw_one_box(
            first_frame, machine_zones, drawn_boxes,
            prompt=f"Draw DESTINATION (place) box for {mz.name}",
            box_color=_DST_COLOR,
        )
        if dst == "__SKIP__":
            configs[mz.zone_id] = PickPlaceConfig(machine_zone=mz, source=src)
            for rmz in machine_zones:
                configs.setdefault(rmz.zone_id, PickPlaceConfig(machine_zone=rmz))
            return configs
        if dst is not None:
            dst.name = f"{mz.name} Dest"
            dst.zone_id = mz.zone_id * 100 + 2
            drawn_boxes.append({"zone": dst, "color": _DST_COLOR,
                                "label": f"DST: {mz.name}"})

        configs[mz.zone_id] = PickPlaceConfig(machine_zone=mz,
                                               source=src, destination=dst)

    return configs


def save_zones_config(filepath: str, zones: List[Zone], pick_place_configs: Dict[int, PickPlaceConfig]):
    """Persists machine zones and SRC/DST bounding boxes to a JSON file."""
    try:
        data = []
        for z in zones:
            ppc = pick_place_configs.get(z.zone_id)
            item = {
                "zone_id": z.zone_id,
                "name": z.name,
                "bbox": [z.x1, z.y1, z.x2, z.y2],
                "source": [ppc.source.x1, ppc.source.y1, ppc.source.x2, ppc.source.y2] if (ppc and ppc.source) else None,
                "destination": [ppc.destination.x1, ppc.destination.y1, ppc.destination.x2, ppc.destination.y2] if (ppc and ppc.destination) else None,
            }
            data.append(item)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[INFO] Saved zones configuration to {filepath}")
    except Exception as e:
        print(f"[WARN] Failed to save zones configuration: {e}")


def load_zones_config(filepath: str) -> Tuple[List[Zone], Dict[int, PickPlaceConfig]]:
    """Loads machine zones and SRC/DST bounding boxes from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    zones = []
    configs = {}
    for item in data:
        zid = int(item["zone_id"])
        zname = str(item["name"])
        bx = item["bbox"]
        mz = Zone(zone_id=zid, name=zname, x1=int(bx[0]), y1=int(bx[1]), x2=int(bx[2]), y2=int(bx[3]))
        zones.append(mz)
        src = None
        if item.get("source"):
            sb = item["source"]
            src = Zone(zone_id=zid * 100 + 1, name=f"{zname} Source",
                       x1=int(sb[0]), y1=int(sb[1]), x2=int(sb[2]), y2=int(sb[3]))
        dst = None
        if item.get("destination"):
            db = item["destination"]
            dst = Zone(zone_id=zid * 100 + 2, name=f"{zname} Dest",
                       x1=int(db[0]), y1=int(db[1]), x2=int(db[2]), y2=int(db[3]))
        configs[zid] = PickPlaceConfig(machine_zone=mz, source=src, destination=dst)
    return zones, configs


# ═══════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ═══════════════════════════════════════════════════════════
_STATUS_CLR = {"Working": (0, 200, 80), "Idle": (0, 130, 255)}
_ALERT_CLR  = (0, 0, 220)


def _draw_person(frame, bbox, cid: int, conf: float, status: str,
                 pieces: int = 0):
    x1, y1, x2, y2 = map(int, bbox)
    col   = id_color(cid)
    s_col = _STATUS_CLR.get(status, (180, 180, 180))
    # Main box
    cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
    # Status bar at bottom
    cv2.rectangle(frame, (x1, y2 - 22), (x2, y2), s_col, -1)
    status_txt = status
    if pieces > 0:
        status_txt += f" | {pieces} pcs"
    cv2.putText(frame, status_txt, (x1 + 4, y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    # Top label
    lbl = f"ID:{cid}  {conf:.0%}"
    if pieces > 0:
        lbl += f"  [{pieces}pcs]"
    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), col, -1)
    cv2.putText(frame, lbl, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def _draw_wrists(frame, left_wrist, right_wrist,
                 lw_conf: float, rw_conf: float,
                 source_zones: List[Zone], dest_zones: List[Zone]):
    """Draw wrist keypoints with colour showing zone containment."""
    for wrist, conf, label in [
        (left_wrist, lw_conf, "L"),
        (right_wrist, rw_conf, "R"),
    ]:
        if wrist is None or conf < WRIST_CONF_MIN:
            continue
        wx, wy = int(wrist[0]), int(wrist[1])
        # Default: white
        color = (255, 255, 255)
        for sz in source_zones:
            if sz.contains_point(wrist[0], wrist[1]):
                color = _SRC_COLOR
                break
        for dz in dest_zones:
            if dz.contains_point(wrist[0], wrist[1]):
                color = _DST_COLOR
                break
        cv2.circle(frame, (wx, wy), 7, color, -1)
        cv2.circle(frame, (wx, wy), 7, (0, 0, 0), 1)
        cv2.putText(frame, label, (wx + 9, wy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _draw_trail(frame, trail, cid: int):
    pts  = list(trail)
    col  = id_color(cid)
    for j in range(1, len(pts)):
        alpha  = j / len(pts)
        c = tuple(int(v * alpha) for v in col)
        cv2.line(frame, pts[j-1], pts[j], c, 2)


def _draw_zones(frame, zones: List[Zone], zone_counts: Dict[int, int],
                pick_place_configs: Dict[int, PickPlaceConfig],
                zone_pieces: Dict[int, int],
                machine_states: Optional[Dict[int, str]] = None):
    for z in zones:
        cnt     = zone_counts.get(z.zone_id, 0)
        crowded = cnt > 1
        col     = _ALERT_CLR if crowded else id_color(z.zone_id)
        # Translucent fill
        ov = frame.copy()
        cv2.rectangle(ov, (z.x1, z.y1), (z.x2, z.y2), col, -1)
        cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)
        # Border
        cv2.rectangle(frame, (z.x1, z.y1), (z.x2, z.y2), col, 2)
        # Label
        alert_tag = "  !! CROWDED !!" if crowded else ""
        pcs = zone_pieces.get(z.zone_id, 0)
        pcs_tag = f"  | {pcs} pcs" if pcs > 0 else ""
        st_tag = f"  [{machine_states.get(z.zone_id)}]" if (machine_states and z.zone_id in machine_states) else ""
        zlbl = f"{z.name}  [{cnt} person{'s' if cnt != 1 else ''}]{alert_tag}{pcs_tag}{st_tag}"
        (tw, th), _ = cv2.getTextSize(zlbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(frame, (z.x1, z.y1 - th - 10), (z.x1 + tw + 6, z.y1), col, -1)
        cv2.putText(frame, zlbl, (z.x1 + 3, z.y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw source / dest boxes
    for mz_id, ppc in pick_place_configs.items():
        for bx, clr, tag in [
            (ppc.source, _SRC_COLOR, "SRC"),
            (ppc.destination, _DST_COLOR, "DST"),
        ]:
            if bx is None:
                continue
            ov = frame.copy()
            cv2.rectangle(ov, (bx.x1, bx.y1), (bx.x2, bx.y2), clr, -1)
            cv2.addWeighted(ov, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (bx.x1, bx.y1), (bx.x2, bx.y2), clr, 2)
            lbl = f"{tag}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (bx.x1, bx.y1 - th - 6),
                          (bx.x1 + tw + 4, bx.y1), clr, -1)
            cv2.putText(frame, lbl, (bx.x1 + 2, bx.y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_alert_log(frame, alerts: deque):
    if not alerts:
        return
    panel_h = min(len(alerts) * 22 + 14, 200)
    fh = frame.shape[0]
    ov = frame.copy()
    cv2.rectangle(ov, (0, fh - panel_h), (520, fh), (10, 10, 40), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    for i, msg in enumerate(list(reversed(list(alerts)))):
        y = fh - panel_h + 18 + i * 22
        if y > fh - 4:
            break
        cv2.putText(frame, msg, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (80, 210, 255), 1, cv2.LINE_AA)


def detect_video_fps(cap: cv2.VideoCapture, default_fps: float = 25.0) -> Tuple[float, int]:
    """
    Robustly determine the true frame rate and frame count of the video.
    CCTV / surveillance camera AVI files frequently store bogus FPS metadata
    (e.g., 600.0 fps or 90000) in the container header. We inspect the millisecond
    deltas of the initial frames to find the real playback FPS.
    Returns: (detected_fps, estimated_total_frames)
    """
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If the reported FPS is in a normal real-world range (10..65 fps):
    if raw_fps and 10.0 <= raw_fps <= 65.0:
        return float(raw_fps), raw_total

    # Probe timestamp deltas from first 30 frames
    cur_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    deltas = []
    last_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
    for _ in range(30):
        ret, _ = cap.read()
        if not ret:
            break
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if last_msec is not None and msec > last_msec:
            diff = msec - last_msec
            if diff >= 5.0:  # Ignore sub-frame tick artifacts
                deltas.append(diff)
        last_msec = msec

    cap.set(cv2.CAP_PROP_POS_FRAMES, cur_pos)  # rewind

    detected_fps = default_fps
    if deltas:
        deltas.sort()
        median_delta = deltas[len(deltas) // 2]
        if median_delta > 0:
            calc_fps = 1000.0 / median_delta
            # Match standard frame rates (24, 25, 29.97, 30, 50, 60)
            for std_fps in [24.0, 25.0, 29.97, 30.0, 50.0, 60.0]:
                if abs(calc_fps - std_fps) < 1.0:
                    detected_fps = std_fps
                    break
            else:
                if 5.0 <= calc_fps <= 120.0:
                    detected_fps = round(calc_fps, 2)

    # If raw_fps was bogus (e.g. 600 fps) and raw_total was scaled up accordingly:
    if raw_fps and raw_fps > 65.0 and raw_total > 0:
        actual_total = max(1, int(round(raw_total * (detected_fps / raw_fps))))
    else:
        actual_total = raw_total

    return detected_fps, actual_total


def _draw_hud(frame, frame_idx: int, total_fr: int, fps: float, total_ids: int,
              active: int, total_pieces: int):
    hud_h = 140 if total_pieces > 0 else 115
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (310, hud_h), (10, 10, 10), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
    fr_str = f"{frame_idx}/{total_fr}" if total_fr > 0 else f"{frame_idx}"
    lines = [
        f"Frame     : {fr_str}",
        f"Proc FPS  : {fps:5.1f}",
        f"Active    : {active}",
        f"Total IDs : {total_ids}",
    ]
    if total_pieces > 0:
        lines.append(f"Pieces    : {total_pieces}")
    for i, ln in enumerate(lines):
        cv2.putText(frame, ln, (8, 22 + i * 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 230, 255), 1, cv2.LINE_AA)


# ═══════════════════════════════════════════════════════════
#  MAIN TRACKING LOOP
# ═══════════════════════════════════════════════════════════
def run(video_path: str, conf_thres: float, output_path: str, show: bool,
        ghost_sec: float = GHOST_BUFFER_SEC, idle_sec: float = IDLE_ALERT_SEC,
        work_buffer_sec: float = WORK_BUFFER_SEC,
        out_fps: float = 0.0, src_fps_override: float = 0.0,
        enable_gemma: bool = True, gemma_model: str = "gemma3:12b",
        ollama_url: str = "http://localhost:11434",
        max_frames: int = 0,
        imgsz: int = 1280,
        zones_config: str = "",
        redraw_zones: bool = False,
        draw_only: bool = False,
        src_dwell: float = 0.30, dst_dwell: float = 0.30,
        max_transit: float = 6.0, cooldown: float = 1.5,
        kp_gap: float = 0.30, zone_margin: int = 8,
        verify_src: bool = False, src_change_min: float = 6.0,
        audit_dir: str = "",
        verify_hold: bool = False, hold_min: float = 0.40,
        hold_px_frac: float = 0.25, hold_radius: int = 50,
        fabric_tol: float = 40.0,
        payload_check: bool = True, payload_radius: int = 60,
        payload_threshold: float = 0.55, payload_min_samples: int = 2,
        payload_dark_v: int = 115, payload_motion: float = 2.5,
        payload_direction: float = 0.55, payload_source_change: float = 0.18,
        payload_source_settle: float = 0.16, payload_motion_accept: float = 0.78):

    # ── Load model (skipped in draw-only mode for instant startup) ──
    model = None
    if not draw_only:
        print(f"[INFO] Loading YOLOv8m-pose (inference imgsz={imgsz}) ...")
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "yolov8m-pose.pt")
        if not os.path.isfile(model_path):
            model_path = "yolov8m-pose.pt"   # fallback: ultralytics auto-download
        model = YOLO(model_path)
        model.fuse()

    # ── Open video ──────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    detected_fps, est_total_fr = detect_video_fps(cap)
    src_fps  = src_fps_override if src_fps_override > 0 else detected_fps
    src_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = est_total_fr

    # FPS-adaptive wrist-sampling skip (~80ms gap)
    flow_skip = max(1, int(round(src_fps * FLOW_SKIP_MS / 1000.0)))
    print(f"[INFO] Video: {src_w}x{src_h} @ {src_fps:.1f} fps (detected true FPS) | ~{total_fr} frames")
    print(f"[INFO] Wrist-motion sampling skip: {flow_skip} frames (~{FLOW_SKIP_MS}ms gap)")
    print(f"[INFO] Working status buffer: {work_buffer_sec:.1f}s hold time")

    # ── First frame → interactive zone drawing or load config ─
    ret, first_frame = cap.read()
    if not ret:
        sys.exit("[ERROR] Could not read first frame.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind

    zones: List[Zone] = []
    pick_place_configs: Dict[int, PickPlaceConfig] = {}

    zones_file = zones_config or "zones_config.json"
    if not redraw_zones and not draw_only and os.path.isfile(zones_file):
        try:
            zones, pick_place_configs = load_zones_config(zones_file)
            print(f"[INFO] Loaded {len(zones)} zone(s) from '{zones_file}'. (Use --redraw-zones to redraw)")
        except Exception as e:
            print(f"[WARN] Could not load zones config '{zones_file}': {e}. Falling back to interactive setup.")
            zones = []
            pick_place_configs = {}

    if not zones and show:
        # Phase 1 – machine zones
        print("\n[INFO] >> Phase 1: Draw machine zones, then press S / Enter.\n"
              "          Press Q / Esc to skip zones.")
        zones = draw_zones_interactive(first_frame)
        if zones:
            print(f"[INFO] {len(zones)} zone(s) defined: {[z.name for z in zones]}")
        else:
            print("[INFO] No zones defined — running full-frame tracking.")

        # Phase 2 – source / destination boxes
        if zones:
            print("\n[INFO] >> Phase 2: Draw Source (pickup) and Destination (place) boxes\n"
                  "          for each machine zone.  Press Q / Esc to skip.")
            pick_place_configs = draw_source_dest_interactive(first_frame, zones)
            n_src = sum(1 for c in pick_place_configs.values() if c.source)
            n_dst = sum(1 for c in pick_place_configs.values() if c.destination)
            print(f"[INFO] Source boxes: {n_src}  |  Destination boxes: {n_dst}")
            save_zones_config(zones_file, zones, pick_place_configs)
            if draw_only:
                print(f"\n[SUCCESS] Zones saved to '{zones_file}'! Exiting as requested by --draw-only.")
                cap.release()
                return
            pairs = sum(1 for c in pick_place_configs.values()
                        if c.source and c.destination)
            if pairs:
                print(f"[INFO] Piece-work counting enabled for {pairs} machine(s).")
            else:
                print("[INFO] No complete source+dest pairs — piece counting disabled.")

    # Pre-build lookup tables for piece-work detection
    source_to_mz: Dict[int, Zone] = {}   # source zone_id -> source Zone
    dest_to_mz:   Dict[int, Zone] = {}
    source_mz_id: Dict[int, int] = {}    # source zone_id -> machine zone_id
    dest_mz_id:   Dict[int, int] = {}
    all_source_zones: List[Zone] = []
    all_dest_zones:   List[Zone] = []
    for mz_id, ppc in pick_place_configs.items():
        if ppc.source is not None:
            source_to_mz[ppc.source.zone_id] = ppc.source
            source_mz_id[ppc.source.zone_id] = mz_id
            all_source_zones.append(ppc.source)
        if ppc.destination is not None:
            dest_to_mz[ppc.destination.zone_id] = ppc.destination
            dest_mz_id[ppc.destination.zone_id] = mz_id
            all_dest_zones.append(ppc.destination)

    has_pairs = any(c.source and c.destination for c in pick_place_configs.values())
    piece_counter = None
    if has_pairs:
        piece_counter = PieceCounter(CounterParams(
            src_dwell=src_dwell, dst_dwell=dst_dwell, max_transit=max_transit,
            cooldown=cooldown, kp_gap=kp_gap, zone_margin=zone_margin,
            verify_src=verify_src, src_change_min=src_change_min,
            verify_hold=verify_hold, hold_min=hold_min,
            hold_px_frac=hold_px_frac, hold_radius=hold_radius,
            fabric_tol=fabric_tol,
            payload_check=(payload_check or verify_hold),
            payload_radius=payload_radius,
            payload_score_threshold=payload_threshold,
            payload_min_samples=payload_min_samples,
            payload_dark_v=payload_dark_v,
            payload_motion_thresh=payload_motion,
            payload_direction_cos=payload_direction,
            payload_source_change_threshold=payload_source_change,
            payload_source_settle_sec=payload_source_settle,
            payload_motion_accept_threshold=payload_motion_accept))
        if audit_dir:
            os.makedirs(audit_dir, exist_ok=True)
        print(f"[INFO] Counter: src_dwell={src_dwell}s dst_dwell={dst_dwell}s "
              f"max_transit={max_transit}s cooldown={cooldown}s "
              f"verify_src={verify_src} verify_hold={verify_hold} "
              f"payload_check={payload_check or verify_hold} "
              f"payload_threshold={payload_threshold} min_samples={payload_min_samples}")

    # ── Output writer ────────────────────────────────────────
    # Default out_fps to src_fps for natural 1:1 real-time playback
    target_out_fps = out_fps if out_fps > 0 else src_fps
    # Write every frame so video plays at normal speed (no skipping)
    write_every = max(1, int(round(src_fps / target_out_fps))) if src_fps > target_out_fps * 1.5 else 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_path, fourcc, target_out_fps, (src_w, src_h))
    print(f"[INFO] Output -> {output_path}")
    print(f"[INFO] Output FPS: {target_out_fps:.1f}  (writing every {write_every} frame(s), real-time 1x playback)\n")

    # ── State ────────────────────────────────────────────────
    ghost_tracker  = GhostTracker(buffer_sec=ghost_sec)
    track_states:  Dict[int, TrackState] = {}
    all_ids:       set  = set()
    alert_log:     deque = deque(maxlen=10)
    alert_cooldown: Dict[str, float] = {}

    # ── Industrial Analytics Telemetry ───────────────────────
    worker_present_sec:    Dict[int, float] = defaultdict(float)
    worker_working_sec:    Dict[int, float] = defaultdict(float)
    worker_idle_sec:       Dict[int, float] = defaultdict(float)
    worker_machine_pieces: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    worker_last_piece_time: Dict[int, float] = {}
    worker_cycle_times:    Dict[int, List[float]] = defaultdict(list)

    machine_occupied_sec:  Dict[str, float] = defaultdict(float)
    machine_working_sec:   Dict[str, float] = defaultdict(float)
    machine_idle_sec:      Dict[str, float] = defaultdict(float)
    machine_pieces:        Dict[str, int]   = defaultdict(int)
    machine_workers:       Dict[str, set]   = defaultdict(set)
    machine_cycle_times:   Dict[str, List[float]] = defaultdict(list)
    machine_last_piece_time: Dict[str, float] = {}

    piece_events_log:      List[Dict[str, Any]] = []
    alert_events_log:      List[Dict[str, Any]] = []
    keyframe_samples:      List[np.ndarray] = []

    ref_counter = 0       # counts up to flow_skip for wrist-motion sampling
    frame_idx   = 0
    t_prev      = time.perf_counter()
    fps_draw    = 0.0
    cur_time    = 0.0

    def _alert(msg: str, cooldown: float = 5.0):
        if cur_time - alert_cooldown.get(msg, -999.0) > cooldown:
            alert_cooldown[msg] = cur_time
            mins = int(cur_time // 60)
            secs = int(cur_time % 60)
            alert_log.append(f"[{mins:02d}:{secs:02d}] {msg}")
            alert_events_log.append({
                "timestamp_sec": round(cur_time, 2),
                "timestamp_str": f"{mins:02d}:{secs:02d}",
                "message": msg
            })
            print(f"  [ALERT @ {mins:02d}:{secs:02d}] {msg}")

    import signal
    interrupted = [False]
    def _sigint_handler(sig, frame):
        interrupted[0] = True
        print("\n[INFO] Interrupt received (Ctrl+C). Finalizing tracking and generating Gemma report...")

    old_sigint = signal.signal(signal.SIGINT, _sigint_handler)

    print("[INFO] Tracking started. Press Q / Esc in the window to quit, or Ctrl+C in terminal.\n")

    while not interrupted[0]:
        if max_frames > 0 and frame_idx >= max_frames:
            print(f"\n[INFO] Reached max-frames limit ({max_frames}). Stopping tracking.")
            break
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        cur_time = frame_idx / max(src_fps, 1e-3)

        # One shared grayscale frame for lightweight temporal fabric-payload checks.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Advance wrist-motion sampling counter
        ref_counter += 1
        update_wrist_ref = (ref_counter >= flow_skip)
        if update_wrist_ref:
            ref_counter = 0

        # ── YOLOv8-pose + ByteTrack ──────────────────────────
        results = model.track(
            frame,
            imgsz=imgsz,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=conf_thres,
            iou=0.45,
            verbose=False,
        )

        # Collect raw detections {raw_id: (xyxy, conf, kp_array)}
        raw_dets: Dict[int, Tuple] = {}
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            kps   = results[0].keypoints  # may be None if no detections
            if boxes.id is not None:
                ids_np  = boxes.id.cpu().numpy().astype(int)
                xyxy_np = boxes.xyxy.cpu().numpy()
                conf_np = boxes.conf.cpu().numpy()
                kp_data = kps.data.cpu().numpy() if kps is not None else None
                for i, (rid, xyxy, cf) in enumerate(zip(ids_np, xyxy_np, conf_np)):
                    kp_i = kp_data[i] if kp_data is not None else None  # (17, 3)
                    raw_dets[int(rid)] = (xyxy, float(cf), kp_i)

        # ── Re-ID resolution ─────────────────────────────────
        resolved: Dict[int, Tuple] = {}
        for rid, (xyxy, cf, kp_i) in raw_dets.items():
            cid = ghost_tracker.resolve(rid, xyxy, cur_time)
            resolved[cid] = (xyxy, cf, kp_i)
        all_ids.update(resolved.keys())

        # ── Vanished tracks → ghost buffer ───────────────────
        current_cids = set(resolved.keys())
        for cid in list(track_states.keys()):
            if cid not in current_cids:
                ghost_tracker.add_ghost(track_states[cid], cur_time)
                del track_states[cid]
        ghost_tracker.cleanup_expired(cur_time)

        # ── Update TrackState per canonical ID ───────────────
        zone_counts: Dict[int, int] = defaultdict(int)
        now = cur_time
        frame_wrists: List[Tuple[float, float, float, int, str]] = []

        for cid, (xyxy, cf, kp_i) in resolved.items():
            if cid not in track_states:
                restored = ghost_tracker.pop_ghost_state(cid)
                track_states[cid] = restored if restored is not None else TrackState(track_id=cid)
            st = track_states[cid]
            st.last_bbox = tuple(int(v) for v in xyxy)
            st.last_conf = cf
            st.last_seen = now

            # Zone membership
            cx_p = (st.last_bbox[0] + st.last_bbox[2]) // 2
            cy_p = (st.last_bbox[1] + st.last_bbox[3]) // 2
            st.zone_ids = [z.zone_id for z in zones if z.contains_center(cx_p, cy_p)]
            for zid in st.zone_ids:
                zone_counts[zid] += 1

            # ── Extract wrist keypoints ──────────────────────
            lw_pos = rw_pos = None
            lw_conf = rw_conf = 0.0
            if kp_i is not None:
                lw_x, lw_y, lw_c = kp_i[KP_LEFT_WRIST]
                rw_x, rw_y, rw_c = kp_i[KP_RIGHT_WRIST]
                if lw_c >= WRIST_CONF_MIN:
                    lw_pos = (float(lw_x), float(lw_y))
                if rw_c >= WRIST_CONF_MIN:
                    rw_pos = (float(rw_x), float(rw_y))
                lw_conf, rw_conf = float(lw_c), float(rw_c)

            st.left_wrist       = lw_pos
            st.right_wrist      = rw_pos
            st.left_wrist_conf  = lw_conf
            st.right_wrist_conf = rw_conf

            # ── Wrist smoothing & collection for Machine Cycle Tracking ──
            if piece_counter is not None:
                smooth = piece_counter.smooth_wrists(cid, lw_pos, rw_pos)
                if "L" in smooth and lw_conf >= WRIST_CONF_MIN:
                    frame_wrists.append((smooth["L"][0], smooth["L"][1], lw_conf, cid, "L"))
                if "R" in smooth and rw_conf >= WRIST_CONF_MIN:
                    frame_wrists.append((smooth["R"][0], smooth["R"][1], rw_conf, cid, "R"))

            # ── Wrist-based Working / Idle ───────────────────
            if update_wrist_ref:
                st.wrist_history.append((lw_pos, rw_pos))
                avg_motion = compute_wrist_motion(st.wrist_history)
                if avg_motion > WRIST_MOTION_THRESH:
                    st.last_work_time = now

            # Working status with buffer hold time (e.g. 3.0s)
            if st.last_work_time is not None and (now - st.last_work_time) <= work_buffer_sec:
                st.working = True
                st.idle_since = None
            else:
                st.working = False
                if st.idle_since is None:
                    # Idle starts after work buffer expired
                    st.idle_since = (st.last_work_time + work_buffer_sec) if st.last_work_time is not None else now
                if now - st.idle_since >= idle_sec:
                    znames = ([z.name for z in zones if z.zone_id in st.zone_ids]
                              or ["(no zone)"])
                    _alert(f"ID:{cid} IDLE >{idle_sec:.0f}s @ {znames[0]}")

            # Record worker operational time telemetry
            dt = 1.0 / max(src_fps, 1e-3)
            worker_present_sec[cid] += dt
            if st.working:
                worker_working_sec[cid] += dt
            else:
                worker_idle_sec[cid] += dt

            # Trail
            st.trail.append((cx_p, cy_p))

        # ── Machine-Centric Piece Counting ────────────────────
        machine_states: Dict[int, str] = {}
        if piece_counter is not None:
            for mzid, ppc in pick_place_configs.items():
                if ppc.source is None or ppc.destination is None:
                    continue
                mname = ppc.machine_zone.name
                ev = piece_counter.update_machine(
                    mzid=mzid,
                    name=mname,
                    now=now,
                    wrists_info=frame_wrists,
                    src_zone=ppc.source,
                    dst_zone=ppc.destination,
                    frame=frame,
                    gray_frame=gray_frame,
                )
                m_tracker = piece_counter.get_tracker(mzid, mname)
                machine_states[mzid] = m_tracker.state.value

                if ev is not None and ev.get("event") in ("count", "reject_payload"):
                    event_type = ev.get("event")
                    cid = int(ev.get("worker_id", 0) or 0)
                    cycle_num = int(ev.get("cycle_num", 0) or 0)
                    transit_sec = float(ev.get("transit_sec", 0.0) or 0.0)
                    confidence = float(ev.get("confidence", 0.0) or 0.0)
                    payload_score = float(ev.get("payload_score", 0.0) or 0.0)
                    payload_confirmed = bool(ev.get("payload_confirmed", False))
                    payload_samples = int(ev.get("payload_samples", 0) or 0)
                    payload_peak = float(ev.get("payload_peak", 0.0) or 0.0)

                    if event_type == "count":
                        m_prev = machine_last_piece_time.get(mname)
                        m_cycle = round(now - m_prev, 2) if m_prev is not None else None
                        machine_last_piece_time[mname] = now
                        if m_cycle is not None and m_cycle < 300:
                            machine_cycle_times[mname].append(m_cycle)

                        if cid > 0:
                            worker_machine_pieces[cid][mname] += 1
                            w_prev = worker_last_piece_time.get(cid)
                            w_cycle = round(now - w_prev, 2) if w_prev is not None else None
                            worker_last_piece_time[cid] = now
                            if w_cycle is not None and w_cycle < 300:
                                worker_cycle_times[cid].append(w_cycle)

                        machine_pieces[mname] = cycle_num

                        piece_events_log.append({
                            "timestamp_sec": round(now, 2),
                            "timestamp_str": f"{int(now // 60):02d}:{int(now % 60):02d}",
                            "frame": frame_idx,
                            "machine": mname,
                            "worker_id": cid,
                            "machine_piece_num": cycle_num,
                            "cycle_time_sec": m_cycle,
                            "transit_sec": transit_sec,
                            "confidence": confidence,
                            "payload_score": payload_score,
                            "payload_confirmed": payload_confirmed,
                            "payload_samples": payload_samples,
                            "payload_peak": payload_peak,
                        })

                        print(f"  [PIECE] {mname}: Cycle #{cycle_num} (+1) | "
                              f"Operator: ID {cid} | Transit: {transit_sec}s | "
                              f"Conf: {confidence:.2f} | Payload: {payload_score:.2f}")

                        if audit_dir:
                            snap_path = os.path.join(
                                audit_dir,
                                f"{mname}_cycle{cycle_num:04d}_ID{cid}_f{frame_idx}.jpg"
                            )
                            active_wrists = [
                                (w[0], w[1]) for w in frame_wrists
                                if ppc.destination.contains_point(w[0], w[1])
                                or ppc.source.contains_point(w[0], w[1])
                            ]
                            render_machine_audit_snapshot(
                                frame=frame,
                                output_path=snap_path,
                                machine_name=mname,
                                machine_zone=ppc.machine_zone,
                                src_zone=ppc.source,
                                dst_zone=ppc.destination,
                                cycle_num=cycle_num,
                                state_str=ev.get("state_str", "SRC -> DST (CYCLE)"),
                                confidence=confidence,
                                transit_sec=transit_sec,
                                timestamp_str=f"{int(now // 60):02d}:{int(now % 60):02d}.{int((now % 1) * 100):02d}",
                                worker_id=cid if cid > 0 else None,
                                wrists=active_wrists,
                                hand_crop=ev.get("hand_crop"),
                                payload_score=payload_score,
                                payload_confirmed=payload_confirmed,
                            )

                    else:
                        print(f"  [REJECT] {mname}: no fabric payload | "
                              f"score={payload_score:.2f} samples={payload_samples} "
                              f"peak={payload_peak:.2f}")

                        if audit_dir:
                            snap_path = os.path.join(
                                audit_dir,
                                f"{mname}_reject_no_fabric_ID{cid}_f{frame_idx}.jpg"
                            )
                            active_wrists = [
                                (w[0], w[1]) for w in frame_wrists
                                if ppc.destination.contains_point(w[0], w[1])
                                or ppc.source.contains_point(w[0], w[1])
                            ]
                            render_machine_audit_snapshot(
                                frame=frame,
                                output_path=snap_path,
                                machine_name=mname,
                                machine_zone=ppc.machine_zone,
                                src_zone=ppc.source,
                                dst_zone=ppc.destination,
                                cycle_num=cycle_num,
                                state_str=ev.get("state_str", "SRC -> DST (REJECTED: NO FABRIC)"),
                                confidence=confidence,
                                transit_sec=transit_sec,
                                timestamp_str=f"{int(now // 60):02d}:{int(now % 60):02d}.{int((now % 1) * 100):02d}",
                                worker_id=cid if cid > 0 else None,
                                wrists=active_wrists,
                                hand_crop=ev.get("hand_crop"),
                                payload_score=payload_score,
                                payload_confirmed=False,
                            )

        # ── Machine zones occupancy & work telemetry ──────────
        dt = 1.0 / max(src_fps, 1e-3)
        for z in zones:
            workers_in_zone = [cid for cid, st in track_states.items() if z.zone_id in st.zone_ids]
            if workers_in_zone:
                machine_occupied_sec[z.name] += dt
                for cid in workers_in_zone:
                    machine_workers[z.name].add(cid)
                if any(track_states[cid].working for cid in workers_in_zone):
                    machine_working_sec[z.name] += dt
                else:
                    machine_idle_sec[z.name] += dt

        # ── Crowding alerts ───────────────────────────────────
        for z in zones:
            cnt = zone_counts.get(z.zone_id, 0)
            if cnt > 1:
                _alert(f"CROWDED: {z.name} has {cnt} persons!", cooldown=4.0)

        # ── Aggregate piece counts per machine zone ───────────
        zone_pieces: Dict[int, int] = defaultdict(int)
        for mzid, ppc in pick_place_configs.items():
            zone_pieces[mzid] = machine_pieces.get(ppc.machine_zone.name, 0)
        total_pieces = sum(zone_pieces.values())

        # ── Render frame ─────────────────────────────────────
        annotated = frame.copy()
        _draw_zones(annotated, zones, zone_counts, pick_place_configs, zone_pieces, machine_states)

        for cid, (xyxy, cf, kp_i) in resolved.items():
            st = track_states[cid]
            status = "Working" if st.working else "Idle"
            worker_pcs = sum(worker_machine_pieces[cid].values())
            _draw_person(annotated, xyxy, cid, cf, status, pieces=worker_pcs)
            _draw_wrists(annotated, st.left_wrist, st.right_wrist,
                         st.left_wrist_conf, st.right_wrist_conf,
                         all_source_zones, all_dest_zones)
            _draw_trail(annotated, st.trail, cid)

        fps_draw = 1.0 / max(time.perf_counter() - t_prev, 1e-6)
        t_prev   = time.perf_counter()
        _draw_alert_log(annotated, alert_log)
        _draw_hud(annotated, frame_idx, total_fr, fps_draw, len(all_ids),
                  len(resolved), total_pieces)

        if frame_idx % write_every == 0:
            writer.write(annotated)

        # Keyframe sampling for multimodal visual inspection
        if (len(keyframe_samples) == 0 and frame_idx >= 5) or \
           (len(keyframe_samples) == 1 and frame_idx >= int(total_fr * 0.55)):
            keyframe_samples.append(annotated.copy())

        if show:
            win_name = "Person Tracker  |  Q=quit"
            cv2.imshow(win_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                print("[INFO] Quit by user.")
                break
            try:
                if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user.")
                    break
            except Exception:
                pass

        if frame_idx % 200 == 0:
            pct = frame_idx / max(total_fr, 1) * 100
            print(f"  Frame {frame_idx}/{total_fr} ({pct:.1f}%) | "
                  f"active={len(resolved)}  total_ids={len(all_ids)}  "
                  f"pieces={total_pieces}  fps={fps_draw:.1f}")

    signal.signal(signal.SIGINT, old_sigint)

    # ── Cleanup & Summary ────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    if piece_events_log:
        import csv
        csv_path = str(Path(output_path).with_name(Path(output_path).stem + "_pieces.csv"))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(piece_events_log[0].keys()))
            w.writeheader()
            w.writerows(piece_events_log)
        print(f"[INFO] Piece audit CSV -> {csv_path}")

    # Free YOLO model and clear CUDA VRAM so Gemma has maximum GPU memory
    try:
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if not keyframe_samples and 'annotated' in locals():
        keyframe_samples.append(annotated.copy())

    total_duration_sec = frame_idx / max(src_fps, 1e-3)
    duration_str = f"{int(total_duration_sec // 60):02d}:{int(total_duration_sec % 60):02d}"

    print(f"\n[DONE] Processed {frame_idx} frames.")
    print(f"       Video duration            : {duration_str} ({total_duration_sec:.1f}s)")
    print(f"       Unique person IDs tracked : {len(all_ids)}")
    print(f"       Output saved to           : {output_path}")

    # Piece-work summary table
    if pick_place_configs:
        print("\n" + "═" * 60)
        print("  PIECE-WORK SUMMARY (MACHINE-CENTRIC)")
        print("═" * 60)
        print(f"  {'Machine':<18} {'Operator(s)':<22} {'Pieces':>8}")
        print("  " + "─" * 54)
        grand_total = 0
        for mzid, ppc in pick_place_configs.items():
            has_pair = ppc.source and ppc.destination
            if not has_pair:
                continue
            zone_name = ppc.machine_zone.name
            person_strs = []
            for cid in all_ids:
                pieces = worker_machine_pieces[cid].get(zone_name, 0)
                if pieces > 0:
                    person_strs.append(f"ID:{cid}({pieces})")
            persons_txt = ", ".join(person_strs) if person_strs else "-"
            m_total = machine_pieces.get(zone_name, 0)
            grand_total += m_total
            print(f"  {zone_name:<18} {persons_txt:<22} {m_total:>8}")
        print("  " + "─" * 54)
        print(f"  {'TOTAL':<18} {'':<22} {grand_total:>8}")
        print("═" * 60)

    # ── Compile Industrial Analytics Telemetry ────────────────
    machines_summary = []
    for z in zones:
        zname = z.name
        pieces = machine_pieces.get(zname, 0)
        occ = round(machine_occupied_sec.get(zname, 0.0), 1)
        work = round(machine_working_sec.get(zname, 0.0), 1)
        idle = round(machine_idle_sec.get(zname, 0.0), 1)
        eff_pct = round((work / max(occ, 1e-3)) * 100.0, 1) if occ > 0 else 0.0
        util_pct = round((work / max(total_duration_sec, 1e-3)) * 100.0, 1)
        uph = round((pieces / max(total_duration_sec, 1e-3)) * 3600.0, 1)
        c_times = machine_cycle_times.get(zname, [])
        avg_cycle = round(sum(c_times) / len(c_times), 1) if c_times else None

        machines_summary.append({
            "machine_name": zname,
            "total_pieces_produced": pieces,
            "occupied_duration_sec": occ,
            "active_work_duration_sec": work,
            "idle_duration_sec": idle,
            "work_efficiency_pct": eff_pct,
            "overall_utilization_pct": util_pct,
            "units_per_hour_rate": uph,
            "avg_cycle_time_sec": avg_cycle,
            "cycle_times_sample": c_times[:10],
            "operators_involved": list(machine_workers.get(zname, set())),
        })

    workers_summary = []
    transient_workers_count = 0
    transient_workers_time = 0.0

    for cid in sorted(all_ids, key=lambda i: (sum(worker_machine_pieces[i].values()), worker_present_sec.get(i, 0.0)), reverse=True):
        tot = round(worker_present_sec.get(cid, 0.0), 1)
        w_pieces = sum(worker_machine_pieces[cid].values())
        w_sec = round(worker_working_sec.get(cid, 0.0), 1)
        i_sec = round(worker_idle_sec.get(cid, 0.0), 1)
        active_pct = round((w_sec / max(tot, 1e-3)) * 100.0, 1) if tot > 0 else 0.0
        c_times = worker_cycle_times.get(cid, [])
        avg_cycle = round(sum(c_times) / len(c_times), 1) if c_times else None

        if w_pieces > 0 or tot >= 6.0 or len(workers_summary) < 10:
            workers_summary.append({
                "worker_id": cid,
                "total_time_present_sec": tot,
                "active_work_sec": w_sec,
                "idle_sec": i_sec,
                "active_ratio_pct": active_pct,
                "idle_ratio_pct": round(100.0 - active_pct, 1),
                "total_pieces_completed": w_pieces,
                "pieces_per_machine": dict(worker_machine_pieces[cid]),
                "avg_cycle_time_sec": avg_cycle,
            })
        else:
            transient_workers_count += 1
            transient_workers_time += tot

    telemetry = {
        "video_metadata": {
            "source_video": os.path.basename(video_path),
            "total_frames": frame_idx,
            "duration_seconds": round(total_duration_sec, 1),
            "duration_formatted": duration_str,
            "processed_fps": round(src_fps, 2),
            "unique_workers_tracked": len(all_ids),
            "transient_passersby_count": transient_workers_count,
        },
        "machines": machines_summary,
        "workers": workers_summary,
        "recent_piece_events": piece_events_log[-15:],
        "recent_alerts": alert_events_log[-10:],
    }

    # ── Gemma 3 AI Analysis ──────────────────────────────────
    if enable_gemma:
        md_out = str(Path(output_path).parent / f"{Path(output_path).stem}_gemma_efficiency_report.md")
        generate_gemma_efficiency_report(
            telemetry=telemetry,
            frames=keyframe_samples,
            model=gemma_model,
            ollama_url=ollama_url,
            output_md_path=md_out,
        )


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    DEFAULT_VIDEO = r"person video/Cam_192.168.170.64_2026-09-25_10-53-57.avi"

    ap = argparse.ArgumentParser(
        description="YOLOv8m-pose + ByteTrack | Hand Tracking | "
                    "Piece-Work Counting | ROI Zones | Gemma 3 Efficiency AI"
    )
    ap.add_argument("--video",     "-v", default=DEFAULT_VIDEO,
                    help="Input video path")
    ap.add_argument("--conf",      "-c", type=float, default=0.25,
                    help="Detection confidence threshold (default: 0.25)")
    ap.add_argument("--output",    "-o", default=None,
                    help="Output video path (default: <stem>_tracked.avi next to input)")
    ap.add_argument("--no-show",         action="store_true",
                    help="Disable live display (save only)")
    ap.add_argument("--ghost-sec",       type=float, default=3.0,
                    help="Re-ID ghost buffer duration in seconds (default: 3)")
    ap.add_argument("--idle-sec",        type=float, default=10.0,
                    help="Seconds of idle before alert (default: 10)")
    ap.add_argument("--work-buffer", "--work-sec", type=float, default=3.0,
                    help="Working status hold buffer in seconds (default: 3.0)")
    ap.add_argument("--src-fps",         type=float, default=0.0,
                    help="Override source video FPS (default: 0 = auto-detect true FPS from timestamps)")
    ap.add_argument("--out-fps",         type=float, default=0.0,
                    help="Output video FPS (default: 0 = match detected source FPS for 1:1 real-time playback)")
    ap.add_argument("--no-gemma",        action="store_true",
                    help="Disable Gemma AI efficiency report generation")
    ap.add_argument("--gemma-model",     default="gemma3:12b",
                    help="Ollama Gemma model tag (default: gemma3:12b)")
    ap.add_argument("--ollama-url",      default="http://localhost:11434",
                    help="Ollama API URL (default: http://localhost:11434)")
    ap.add_argument("--max-frames",      type=int, default=0,
                    help="Stop after processing N frames (default: 0 = all frames)")
    ap.add_argument("--imgsz",           type=int, default=1280,
                    help="YOLO pose inference resolution (default: 1280 for sharp wrist/pose keypoints)")
    ap.add_argument("--zones-config",    default="zones_config.json",
                    help="Path to JSON file containing machine and SRC/DST zones (default: zones_config.json)")
    ap.add_argument("--redraw-zones",    action="store_true",
                    help="Force interactive zone drawing even if zones-config JSON exists")
    ap.add_argument("--draw-only",       action="store_true",
                    help="Open GUI to draw machine zones and SRC/DST boxes, save to JSON, and exit without running tracking")
    ap.add_argument("--src-dwell",       type=float, default=0.30)
    ap.add_argument("--dst-dwell",       type=float, default=0.30)
    ap.add_argument("--max-transit",     type=float, default=6.0)
    ap.add_argument("--cooldown",        type=float, default=1.5)
    ap.add_argument("--kp-gap",          type=float, default=0.30)
    ap.add_argument("--zone-margin",     type=int,   default=8)
    ap.add_argument("--verify-src",      action="store_true",
                    help="Reject counts when the SRC area did not visibly change")
    ap.add_argument("--src-change-min",  type=float, default=6.0)
    ap.add_argument("--audit-dir",       default="",
                    help="Save a snapshot for every counted piece")
    ap.add_argument("--verify-hold",     action="store_true",
                    help="Reject counts when the hand did not appear to hold fabric")
    ap.add_argument("--hold-min",        type=float, default=0.40)
    ap.add_argument("--hold-px-frac",    type=float, default=0.25)
    ap.add_argument("--hold-radius",     type=int,   default=50)
    ap.add_argument("--fabric-tol",      type=float, default=40.0)

    # Temporal wrist-crop fabric payload gate (enabled by default).
    ap.add_argument("--no-payload-check", action="store_true",
                    help="Disable fabric-payload verification; count SRC->DST from wrist geometry only")
    ap.add_argument("--payload-radius", type=int, default=60,
                    help="Half-size of temporal wrist/fabric ROI in pixels (default: 60)")
    ap.add_argument("--payload-threshold", type=float, default=0.50,
                    help="Payload evidence score required for a verified pickup (default: 0.55)")
    ap.add_argument("--payload-min-samples", type=int, default=1,
                    help="Number of positive transit samples required (default: 2)")
    ap.add_argument("--payload-dark-v", type=int, default=115,
                    help="Maximum grayscale value treated as dark fabric support (default: 115)")
    ap.add_argument("--payload-motion", type=float, default=2.5,
                    help="Minimum local optical-flow magnitude in pixels (default: 1.5)")
    ap.add_argument("--payload-direction", type=float, default=0.55,
                    help="Minimum flow alignment with wrist travel, cosine 0..1 (default: 0.35)")
    ap.add_argument("--payload-source-change", type=float, default=0.18,
                    help="Minimum SRC before/after change score for pickup evidence (default: 0.18)")
    ap.add_argument("--payload-source-settle", type=float, default=0.16,
                    help="Seconds to wait after leaving SRC before measuring source change (default: 0.16)")
    ap.add_argument("--payload-motion-accept", type=float, default=0.78,
                    help="Strong wrist-motion payload score (still requires some SRC change) (default: 0.78)")
    args = ap.parse_args()

    vpath = args.video
    if not os.path.isabs(vpath):
        vpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), vpath)

    out = args.output or str(Path(vpath).parent / f"{Path(vpath).stem}_tracked.avi")

    run(
        video_path       = vpath,
        conf_thres       = args.conf,
        output_path      = out,
        show             = not args.no_show,
        ghost_sec        = args.ghost_sec,
        idle_sec         = args.idle_sec,
        work_buffer_sec  = args.work_buffer,
        out_fps          = args.out_fps,
        src_fps_override = args.src_fps,
        enable_gemma     = not args.no_gemma,
        gemma_model      = args.gemma_model,
        ollama_url       = args.ollama_url,
        max_frames       = args.max_frames,
        imgsz            = args.imgsz,
        zones_config     = args.zones_config,
        redraw_zones     = args.redraw_zones,
        draw_only        = args.draw_only,
        src_dwell        = args.src_dwell,
        dst_dwell        = args.dst_dwell,
        max_transit      = args.max_transit,
        cooldown         = args.cooldown,
        kp_gap           = args.kp_gap,
        zone_margin      = args.zone_margin,
        verify_src       = args.verify_src,
        src_change_min   = args.src_change_min,
        audit_dir        = args.audit_dir,
        verify_hold      = args.verify_hold,
        hold_min         = args.hold_min,
        hold_px_frac     = args.hold_px_frac,
        hold_radius      = args.hold_radius,
        fabric_tol       = args.fabric_tol,
           payload_check    = not args.no_payload_check,
        payload_radius   = args.payload_radius,
        payload_threshold= args.payload_threshold,
        payload_min_samples = args.payload_min_samples,
        payload_dark_v   = args.payload_dark_v,
        payload_motion   = args.payload_motion,
        payload_direction= args.payload_direction,
        payload_source_change=args.payload_source_change,
        payload_source_settle=args.payload_source_settle,
        payload_motion_accept=args.payload_motion_accept,
        )
