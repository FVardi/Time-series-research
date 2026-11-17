# Runs the benchmark baselines still missing (D267: |FFT|, envelope, MiniRocket), seed 0 (D239), and
# evaluates the Herning MSSP run made under D255 (pretrained 2026-11-11, never evaluated). CPU only.
# From the repository root, in one terminal:
#     powershell -ExecutionPolicy Bypass -File scripts\queue_d267.ps1
# Each config: scripts/pretrain.py, then scripts/evaluate.py on its run folder.
# Skips a config that has already finished, so the queue can be restarted after an interruption.
# Everything printed goes to runs\queue_d267.log as well.

$configs = @(
    "fft_xjtu_sy", "envelope_xjtu_sy", "minirocket_xjtu_sy",  # XJTU-SY RUL task (D247, D266 windows)
    "fft_cwru", "envelope_cwru", "minirocket_cwru"           # CWRU standard runs (D218, D239)
)
$evaluate_only = @("runs\20261111_131844_mssp_herning")       # Herning MSSP under D255

$log = "runs\queue_d267.log"
foreach ($folder in $evaluate_only) {
    if (Test-Path (Join-Path $folder "results.csv")) { "skip $folder (evaluated)" | Tee-Object -FilePath $log -Append; continue }
    "=== evaluate $folder  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
    uv run --no-sync python scripts/evaluate.py $folder 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { "FAILED evaluating $folder" | Tee-Object -FilePath $log -Append }
}
foreach ($name in $configs) {
    # finished = seed 0 stored and the run evaluated
    $done = Get-ChildItem runs -Directory -Filter "*_$name" -ErrorAction SilentlyContinue |
            Where-Object { (Test-Path (Join-Path $_.FullName "seed_0\Z_test.npy")) -and
                           (Test-Path (Join-Path $_.FullName "results.csv")) }
    if ($done) { "skip $name (finished: $($done[0].Name))" | Tee-Object -FilePath $log -Append; continue }

    "=== $name  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
    # --no-sync: use the environment as installed
    uv run --no-sync python scripts/pretrain.py "configs/$name.yaml" --seeds 0 2>&1 | Tee-Object -FilePath $log -Append |
        Tee-Object -Variable out | Out-Host  # shown, logged and kept in $out
    $match = $out | Select-String "^Done: (.+)$"
    if (-not $match) { "FAILED pretraining $name" | Tee-Object -FilePath $log -Append; continue }
    $folder = $match[0].Matches[0].Groups[1].Value  # this config's run folder
    uv run --no-sync python scripts/evaluate.py $folder 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { "FAILED evaluating $name ($folder)" | Tee-Object -FilePath $log -Append }
    else { "evaluated $folder" | Tee-Object -FilePath $log -Append }
}
"=== queue done  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
