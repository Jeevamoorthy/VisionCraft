@echo off
echo.
echo  ============================================================
echo   VisionCraft: Industrial Sewing & Piece Tracker
echo   - Draw ROI zones per machine (drag on first frame)
echo   - Crowding alert (>1 person per zone)
echo   - Re-ID buffer (3s ghost window)
echo   - Working / Idle status (wrist motion + 3s buffer)
echo   - Output saved in real-time 1x playback
echo  ============================================================
echo.
cd /d "%~dp0"
python person_tracker.py ^
    --video "person video\Workers_operating_sewing_machines_1080p_20261008140921.mp4" ^
    --conf 0.25 ^
    --ghost-sec 3 ^
    --idle-sec 10 ^
    --work-buffer 3 ^
    --out-fps 0
echo.
echo  [Done] Press any key to close...
pause > nul
