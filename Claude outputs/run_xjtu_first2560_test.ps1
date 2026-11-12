# Diagnostic: FFT baseline on XJTU-SY, whole minute (32,768 points, D254) against the first 2,560 points
# of each minute (D247 item 1), for the health-stage and RUL tasks. CPU only; can run beside the GPU queue.
# From the repository root:
#     powershell -ExecutionPolicy Bypass -File "Claude outputs\run_xjtu_first2560_test.ps1"
# Stage first (small); the whole-minute RUL run is the heavy one (5 minutes x 2 channels x 16,385 bins per row).
$log = "runs\xjtu_first2560_test.log"
foreach ($name in "fft_xjtu_sy_stage_first2560", "fft_xjtu_sy_stage", "fft_xjtu_sy_first2560", "fft_xjtu_sy") {
    "=== $name  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
    uv run --no-sync python scripts/pretrain.py "configs/$name.yaml" --seeds 0 2>&1 | Tee-Object -FilePath $log -Append |
        Tee-Object -Variable out | Out-Host
    $match = $out | Select-String "^Done: (.+)$"
    if (-not $match) { "FAILED pretraining $name" | Tee-Object -FilePath $log -Append; continue }
    $folder = $match[0].Matches[0].Groups[1].Value
    uv run --no-sync python scripts/evaluate.py $folder 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
}
"=== done  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
