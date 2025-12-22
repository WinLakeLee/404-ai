#!/usr/bin/env pwsh
# requirements_check.ps1
# Python이 PATH에 있는지 확인하고, 주요 패키지들을 import하여 버전/상태를 출력합니다.
Write-Host "Starting Python requirements check..." -ForegroundColor Cyan

# Python 실행 확인
$pythonVersion = & python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Python not found. Ensure 'python' is on PATH." -ForegroundColor Red
    exit 1
} else {
    Write-Host $pythonVersion -ForegroundColor Green
}

function Run-PyCheck {
    param(
        [string]$code,
        [string]$label
    )
    Write-Host "`n==> Checking $label ..." -ForegroundColor Cyan
    $out = & python -c $code 2>&1
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Write-Host $out -ForegroundColor Green
    } else {
        Write-Host $out -ForegroundColor Yellow
    }
    return $rc
}

$checks = @(
@{ label = "PyTorch (CUDA)"; code = @'
import sys
try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
except ImportError:
    print("PyTorch is not installed. Please install it to check CUDA availability.")
    sys.exit(1)
'@ },

@{ label = "Ultralytics"; code = @'
import sys
try:
    import ultralytics
    print("Ultralytics version:", ultralytics.__version__)
except ImportError:
    print("Ultralytics is not installed. Please install it to use FastSAM.")
    sys.exit(1)
'@ },

@{ label = "Paho-MQTT"; code = @'
import sys
try:
    import paho.mqtt.client as mqtt
    print("Paho-MQTT", mqtt.MQTT_CLIENT)
except ImportError:
    print("Paho-MQTT is not installed. Please install it to use MQTT features.")
    sys.exit(1)
'@ },

@{ label = "PIL (Pillow)"; code = @'
import sys
try:
    import PIL
    print("PIL version:", PIL.__version__)
except ImportError:
    print("PIL (Pillow) is not installed. Please install it to handle image processing.")
    sys.exit(1)
'@ },

@{ label = "OpenCV (cv2)"; code = @'
import sys
try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except ImportError:
    print("OpenCV is not installed. Please install it to handle image and video processing.")
    sys.exit(1)
'@ },

@{ label = "NumPy"; code = @'
import sys
try:
    import numpy as np
    print("NumPy version:", np.__version__)
except ImportError:
    print("NumPy is not installed. Please install it for numerical operations.")
    sys.exit(1)
'@ },

@{ label = "Matplotlib"; code = @'
import sys
try:
    import matplotlib
    print("Matplotlib version:", matplotlib.__version__)
except ImportError:
    print("Matplotlib is not installed. Please install it for plotting and visualization.")
    sys.exit(1)
'@ },

@{ label = "python-dotenv"; code = @'
import sys
try:
    import dotenv
    ver = getattr(dotenv, '__version__', getattr(dotenv, 'version', None))
    print("python-dotenv", ver)
except ImportError:
    print("python-dotenv is not installed. Please install it to manage environment variables.")
    sys.exit(1)
'@ }
)

# 패키지 -> 설치 커맨드 매핑 (PyTorch는 별도 로직으로 처리)
$installMap = @{
    'Ultralytics' = @{ args = 'install ultralytics==8.1.0' }
    'Paho-MQTT' = @{ args = 'install paho-mqtt==1.6.1' }
    'PIL (Pillow)' = @{ args = 'install pillow==10.1.0' }
    'OpenCV (cv2)' = @{ args = 'install opencv-python>=4.8.0' }
    'NumPy' = @{ args = 'install numpy>=1.24.0' }
    'Matplotlib' = @{ args = 'install matplotlib' }
    'python-dotenv' = @{ args = 'install python-dotenv==1.0.0' }
}

function Install-Package {
    param(
        [string]$label,
        [string]$installArgs
    )
    Write-Host "Installing $label via pip: python -m pip $installArgs" -ForegroundColor Cyan
    $argArray = $installArgs -split ' '
    $out = & python -m pip $argArray 2>&1
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Write-Host $out -ForegroundColor Green
    } else {
        Write-Host $out -ForegroundColor Red
    }
    return $rc
}

function Detect-GpuCudaVersion {
    # nvidia-smi가 설치되어 있고 출력에 CUDA Version이 적혀있는 경우 파싱
    $out = & nvidia-smi 2>&1
    if ($LASTEXITCODE -ne 0) { return $null }
    foreach ($line in $out) {
        if ($line -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') {
            return $matches[1]
        }
    }
    $joined = $out -join "`n"
    if ($joined -match 'CUDA Version:\s*([0-9]+\.[0-9]+)') { return $matches[1] }
    return $null
}

foreach ($c in $checks) {
    if ($c.label -eq 'PyTorch (CUDA)') {
        Write-Host "`n==> Checking PyTorch (CUDA) with detailed logic..." -ForegroundColor Cyan
        $pyOut = & python -c $c.code 2>&1
        $rc = $LASTEXITCODE
        $col = if ($rc -eq 0) { 'Green' } else { 'Yellow' }
        Write-Host $pyOut -ForegroundColor $col

        if ($rc -ne 0) {
            # torch가 없다면 시스템 CUDA 여부를 확인한 뒤 CUDA wheel만 설치합니다
            Write-Host "PyTorch not installed. Detecting system CUDA to decide installation..." -ForegroundColor Yellow
            $detected = Detect-GpuCudaVersion
            if ($null -ne $detected) {
                Write-Host "Detected system CUDA version: $detected -- attempting CUDA-enabled torch install" -ForegroundColor Cyan
                $tag = ($detected -replace '\.','')
                if ($tag.Length -gt 3) { $tag = $tag.Substring(0,3) }
                $torchArgs = "install torch torchvision --index-url https://download.pytorch.org/whl/cu$tag"
                $ir = Install-Package -label "PyTorch (CUDA)" -installArgs $torchArgs
                if ($ir -eq 0) {
                    Write-Host "Re-checking PyTorch after install..." -ForegroundColor Cyan
                    $out2 = & python -c $c.code 2>&1
                    $rc2 = $LASTEXITCODE
                    $col2 = if ($rc2 -eq 0) { 'Green' } else { 'Yellow' }
                    Write-Host $out2 -ForegroundColor $col2
                } else {
                    Write-Host "Failed to install PyTorch with CUDA tag cu$tag." -ForegroundColor Red
                }
            } else {
                Write-Host "No CUDA detected (or nvidia-smi not available). Skipping automatic PyTorch installation." -ForegroundColor Yellow
            }
        } else {
            # torch는 설치되어 있으나 CUDA 불가인 경우 시스템 CUDA 확인 후 CUDA wheel 설치 시도
            if ($pyOut -match 'CUDA available:\s*False') {
                Write-Host "Torch installed but CUDA not available. Detecting system CUDA..." -ForegroundColor Yellow
                $detected = Detect-GpuCudaVersion
                if ($null -ne $detected) {
                    Write-Host "Detected system CUDA version: $detected" -ForegroundColor Cyan
                    $tag = ($detected -replace '\.','')
                    if ($tag.Length -gt 3) { $tag = $tag.Substring(0,3) }
                    $torchArgs = "install torch torchvision --index-url https://download.pytorch.org/whl/cu$tag"
                    Write-Host "Attempting to install PyTorch with CUDA tag cu$tag..." -ForegroundColor Cyan
                    $ir = Install-Package -label "PyTorch (CUDA)" -installArgs $torchArgs
                    if ($ir -eq 0) {
                        Write-Host "Re-checking PyTorch CUDA availability..." -ForegroundColor Cyan
                        $out3 = & python -c $c.code 2>&1
                        $rc3 = $LASTEXITCODE
                        $col3 = if ($rc3 -eq 0) { 'Green' } else { 'Yellow' }
                        Write-Host $out3 -ForegroundColor $col3
                        if ($out3 -match 'CUDA available:\s*True') { Write-Host "CUDA is now available for PyTorch." -ForegroundColor Green }
                    } else {
                        Write-Host "Failed to install PyTorch with CUDA tag cu$tag." -ForegroundColor Red
                    }
                } else {
                    Write-Host "Could not detect system CUDA via nvidia-smi. Skipping automatic CUDA install." -ForegroundColor Yellow
                }
            }
        }

    } else {
        # 일반 패키지 처리 (없으면 설치)
        $rc = Run-PyCheck -code $c.code -label $c.label
        if ($rc -ne 0) {
            if ($installMap.ContainsKey($c.label)) {
                $installArgs = $installMap[$c.label].args
                $ir = Install-Package -label $c.label -installArgs $installArgs
                if ($ir -eq 0) {
                    Write-Host "Re-checking $($c.label) after install..." -ForegroundColor Cyan
                    $rc2 = Run-PyCheck -code $c.code -label $c.label
                    if ($rc2 -ne 0) { Write-Host "$($c.label) still failing after install." -ForegroundColor Yellow }
                } else {
                    Write-Host "Failed to install $($c.label). See pip output above." -ForegroundColor Red
                }
            } else {
                Write-Host "No install mapping for $($c.label). Please install manually." -ForegroundColor Yellow
            }
        }
    }
}

Write-Host "`nAll checks completed." -ForegroundColor Cyan


