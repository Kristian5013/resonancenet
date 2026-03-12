@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\Common7\Tools\VsDevCmd.bat" -arch=amd64
cd /d C:\ResonanceNet
ninja -C build rnetd rnet-cli 2>&1
