@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set CUDA_VISIBLE_DEVICES=0
cd /d "%~dp0\.."
python -u scripts/_test_torch.py
echo EXIT_CODE=%ERRORLEVEL%
