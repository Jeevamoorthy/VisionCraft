@echo off
echo.
echo  ============================================================
echo   YOLOv8m + ByteTrack  Advanced Person Tracker
echo   - Draw ROI zones per machine (drag on first frame)
echo   - Crowding alert (>1 person per zone)
echo   - Re-ID buffer (3s ghost window)
echo   - Working / Idle status (wrist motion + 3s buffer)
echo   - Output saved in real-time 1x playback
echo  ============================================================
echo.
cd /d "%~dp0"
python person_tracker.py ^
    --video "person video\Cam_192.168.170.64_2026-09-25_10-53-57.avi" ^
    --conf 0.25 ^
    --ghost-sec 3 ^
    --idle-sec 10 ^
    --work-buffer 3 ^
    --out-fps 0
echo.
echo  [Done] Press any key to close...
pause > nul
