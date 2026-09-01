@echo off
chcp 65001 >nul
echo ===== 一键上传 GitHub =====
echo 目标: mbti / magicbook / expedition （ %* 则全部上传）
cd /d "d:\Trae CN\Garth"
node "d:\Trae CN\Garth\upload-to-github.js" mbti
echo.
pause
