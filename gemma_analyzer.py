"""
Gemma 3 Local AI Video & Workstation Efficiency Analyzer
Connects to local Ollama instance running gemma3:12b to perform
detailed industrial engineering, machine efficiency, and operator
productivity analysis on computer-vision tracked telemetry and frames.
"""

import sys
import json
import base64
import urllib.request
import urllib.error
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional


def check_ollama_available(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if the local Ollama server is running."""
    try:
        req = urllib.request.Request(f"{ollama_url.rstrip('/')}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False


def encode_frame_to_b64(frame: np.ndarray, max_width: int = 768) -> str:
    """Resize and JPEG-encode an OpenCV frame to base64 for Gemma multimodal input."""
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret:
        return ""
    return base64.b64encode(buf).decode("utf-8")


def generate_gemma_efficiency_report(
    telemetry: Dict[str, Any],
    frames: Optional[List[np.ndarray]] = None,
    model: str = "gemma3:12b",
    ollama_url: str = "http://localhost:11434",
    output_md_path: Optional[str] = None,
) -> Optional[str]:
    """
    Sends processed tracking telemetry and annotated keyframe(s) to Gemma 3 (via Ollama)
    and writes out an Industrial Engineering & Machine Work Efficiency report.
    """
    if not check_ollama_available(ollama_url):
        print(f"\n[WARNING] Ollama server at {ollama_url} is not reachable. Skipping Gemma analysis.")
        return None

    print(f"\n{'═' * 65}")
    print(f"  SENDING VIDEO TELEMETRY TO GEMMA 3 ({model})")
    print(f"  Generating Machine Efficiency & Operator Productivity Report...")
    print(f"{'═' * 65}\n")

    # Encode 1 representative keyframe to keep payload token footprint optimal
    b64_images: List[str] = []
    if frames:
        for idx, frm in enumerate(frames[:1]):
            encoded = encode_frame_to_b64(frm)
            if encoded:
                b64_images.append(encoded)

    # Build prompt
    prompt = f"""You are a Senior Industrial Engineer, Lean Manufacturing Specialist, and Time-and-Motion Study Expert.
You have been provided with automated computer vision tracking telemetry and visual workstation snapshot(s) from a factory garment/textile manufacturing workstation.

TELEMETRY & PRODUCTION DATA:
{json.dumps(telemetry, indent=2)}

TASK:
Analyze this video tracking data and visual layout to produce a comprehensive, professional "Workstation & Machine Efficiency Study".

Your report MUST include the following structured sections with detailed quantitative analysis and insights:

1. # Executive Summary & Production Overview
   - High-level production throughput summary (Total units produced, video observation duration, total operators).
   - High-level efficiency verdict.

2. # Machine 1 & Workstation Efficiency Analysis
   - For Machine 1 (and each configured machine zone):
     * Active Working Time vs. Idle Time.
     * Work Efficiency Percentage (Active Work Time / Occupied Time).
     * Station Utilization Rate (Active Time / Total Video Time).
     * Output Rate: Units Per Hour (UPH) & Average Cycle Time per piece.
     * Consistency of production pacing (cycle regularity).

3. # Operator Performance & Motion Study
   - Individual breakdown for each Person ID:
     * Active ratio (%) vs. Idle ratio (%).
     * Units completed per person.
     * Wrist motion and physical handling pace.
     * Comments on ergonomics, posture, and manual dexterity observed in the visual snapshot.

4. # Bottlenecks, Downtime & Safety Anomalies
   - Analysis of prolonged idle episodes or gaps between piece completions.
   - Evaluation of crowding incidents (>1 person per zone) and safety considerations.
   - Material supply flow: Pick-up (Source Box) to Placement (Destination Box) fluidity.

5. # Kaizen & Engineering Recommendations
   - Concrete, high-impact recommendations to improve Machine 1 work efficiency.
   - Physical layout adjustments (box placement, reach distances, operator stance).
   - Line balancing and idle-reduction strategies.

Provide crisp formatting with Markdown tables, bullet points, and percentage metrics.
"""

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": 8192,
        }
    }
    if b64_images:
        payload["images"] = b64_images

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )

    try:
        print("═" * 65)
        print("  GEMMA 3 MACHINE EFFICIENCY & PRODUCTIVITY REPORT")
        print("═" * 65)
        report_chunks = []
        with urllib.request.urlopen(req, timeout=300) as response:
            for line in response:
                if line:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    sys.stdout.write(token)
                    sys.stdout.flush()
                    report_chunks.append(token)
        print("\n" + "═" * 65 + "\n")
        report_text = "".join(report_chunks)

        if output_md_path and report_text:
            Path(output_md_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            print(f"[SUCCESS] Gemma 3 Efficiency Report saved to:\n  -> {output_md_path}\n")

        return report_text

    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="replace")
        print(f"\n[ERROR] HTTP Error {e.code} querying Gemma via Ollama: {err_msg}")
        return None
    except Exception as e:
        print(f"\n[ERROR] Failed to query Gemma via Ollama: {e}")
        import traceback
        traceback.print_exc()
        return None
