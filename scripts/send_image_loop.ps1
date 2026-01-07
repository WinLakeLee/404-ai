param(
# 이미지 경로
    [string]$ImagePath = "img/1.jpg",
# 전송 간격 (초)
    [int]$IntervalSec = 10,
# 전송 방식
    [switch]$Http = $true,
    [switch]$Mqtt = $true
)

Write-Host "Starting loop: python scripts/send_image.py $ImagePath --http:$Http --mqtt:$Mqtt every $IntervalSec seconds"

while ($true) {
    try {
        python "$PSScriptRoot\send_image.py" $ImagePath @(
            if ($Http) { "--http" }
            if ($Mqtt) { "--mqtt" }
        )
    }
    catch {
        Write-Warning "send_image failed: $_"
    }
    Start-Sleep -Seconds $IntervalSec
}
