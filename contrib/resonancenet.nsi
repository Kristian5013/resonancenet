; ResonanceNet NSIS Installer Script
; Requires: NSIS 3.x (https://nsis.sourceforge.io)

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "EnvVarUpdate.nsh"

; ---------------------------------------------------------------------------
; Product metadata
; ---------------------------------------------------------------------------
!define PRODUCT_NAME        "ResonanceNet"
!define PRODUCT_VERSION     "0.1.0"
!define PRODUCT_PUBLISHER   "ResonanceNet Developers"
!define PRODUCT_WEB_SITE    "https://github.com/Kristian5013/resonancenet"
!define PRODUCT_UNINST_KEY  "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"
!define PRODUCT_UNINST_ROOT "HKLM"

; ---------------------------------------------------------------------------
; General settings
; ---------------------------------------------------------------------------
Name            "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile         "resonancenet-${PRODUCT_VERSION}-win64-setup.exe"
InstallDir      "$PROGRAMFILES\ResonanceNet"
InstallDirRegKey HKLM "Software\${PRODUCT_NAME}" "InstallDir"
RequestExecutionLevel admin
SetCompressor   /SOLID lzma

; ---------------------------------------------------------------------------
; MUI settings
; ---------------------------------------------------------------------------
!define MUI_ABORTWARNING
!define MUI_ICON   "..\share\pixmaps\resonancenet.ico"
!define MUI_UNICON "..\share\pixmaps\resonancenet.ico"

; ---------------------------------------------------------------------------
; Pages
; ---------------------------------------------------------------------------
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\COPYING"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

; ---------------------------------------------------------------------------
; Install section
; ---------------------------------------------------------------------------
Section "ResonanceNet Core (required)" SecCore
    SectionIn RO

    SetOutPath "$INSTDIR"

    ; Binaries
    File "..\build\rnetd.exe"
    File "..\build\rnet-cli.exe"
    File "..\build\rnet-tx.exe"
    File "..\build\rnet-wallet-tool.exe"
    File "..\build\rnet-util.exe"
    File "..\build\rnet-miner.exe"

    ; Documentation
    File "..\COPYING"
    File "..\README.md"

    ; Default configuration
    SetOutPath "$APPDATA\ResonanceNet"
    IfFileExists "$APPDATA\ResonanceNet\resonancenet.conf" skip_conf
        FileOpen $0 "$APPDATA\ResonanceNet\resonancenet.conf" w
        FileWrite $0 "# ResonanceNet configuration$\r$\n"
        FileWrite $0 "# See: rnetd -help for all options$\r$\n"
        FileWrite $0 "$\r$\n"
        FileWrite $0 "rpcport=9556$\r$\n"
        FileWrite $0 "rpcbind=127.0.0.1$\r$\n"
        FileWrite $0 "port=9555$\r$\n"
        FileWrite $0 "listen=1$\r$\n"
        FileClose $0
    skip_conf:

    ; Start Menu shortcuts
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortCut  "$SMPROGRAMS\${PRODUCT_NAME}\ResonanceNet Daemon.lnk"  "$INSTDIR\rnetd.exe"
    CreateShortCut  "$SMPROGRAMS\${PRODUCT_NAME}\ResonanceNet CLI.lnk"     "$INSTDIR\rnet-cli.exe"
    CreateShortCut  "$SMPROGRAMS\${PRODUCT_NAME}\ResonanceNet Miner.lnk"   "$INSTDIR\rnet-miner.exe"
    CreateShortCut  "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk"            "$INSTDIR\uninstall.exe"

    ; Add to PATH
    ${EnvVarUpdate} $0 "PATH" "A" "HKLM" "$INSTDIR"

    ; Registry — uninstall information
    WriteRegStr   ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "DisplayName"     "${PRODUCT_NAME}"
    WriteRegStr   ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "DisplayVersion"  "${PRODUCT_VERSION}"
    WriteRegStr   ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "Publisher"        "${PRODUCT_PUBLISHER}"
    WriteRegStr   ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "URLInfoAbout"    "${PRODUCT_WEB_SITE}"
    WriteRegStr   ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr   ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\uninstall.exe"
    WriteRegDWORD ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "NoModify" 1
    WriteRegDWORD ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "NoRepair" 1

    ; Compute installed size
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}" "EstimatedSize" "$0"

    ; Write uninstaller
    WriteUninstaller "$INSTDIR\uninstall.exe"

    ; Store install dir
    WriteRegStr HKLM "Software\${PRODUCT_NAME}" "InstallDir" "$INSTDIR"
SectionEnd

; ---------------------------------------------------------------------------
; Uninstall section
; ---------------------------------------------------------------------------
Section "Uninstall"
    ; Remove binaries
    Delete "$INSTDIR\rnetd.exe"
    Delete "$INSTDIR\rnet-cli.exe"
    Delete "$INSTDIR\rnet-tx.exe"
    Delete "$INSTDIR\rnet-wallet-tool.exe"
    Delete "$INSTDIR\rnet-util.exe"
    Delete "$INSTDIR\rnet-miner.exe"
    Delete "$INSTDIR\COPYING"
    Delete "$INSTDIR\README.md"
    Delete "$INSTDIR\uninstall.exe"
    RMDir  "$INSTDIR"

    ; Remove Start Menu
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\*.lnk"
    RMDir  "$SMPROGRAMS\${PRODUCT_NAME}"

    ; Remove from PATH
    ${un.EnvVarUpdate} $0 "PATH" "R" "HKLM" "$INSTDIR"

    ; Remove registry
    DeleteRegKey ${PRODUCT_UNINST_ROOT} "${PRODUCT_UNINST_KEY}"
    DeleteRegKey HKLM "Software\${PRODUCT_NAME}"

    ; Note: we intentionally do NOT delete $APPDATA\ResonanceNet (user data)
SectionEnd
