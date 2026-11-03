@echo off
rem Jarvis's work thread, on this PC.
rem   brain-worker.cmd --user ^<supabase user id^>
rem   brain-worker.cmd --local --repo agent "what to change"
cd /d "%~dp0.."
uv run python -m src.brain.worker %*
