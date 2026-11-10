# Runs the D256 configs (and the missing Paderborn TF-C arm) one after the other, seed 0 (D239).
# Two queues, to run at the same time in two terminals from the repository root:
#     powershell -ExecutionPolicy Bypass -File scripts\queue_d256.ps1 -Queue gpu
#     powershell -ExecutionPolicy Bypass -File scripts\queue_d256.ps1 -Queue cpu
# gpu: the self-supervised methods, cheapest first, so a first HUST/Paderborn table exists after one night.
# cpu: the baselines (FFT, envelope, catch22, MiniRocket), which run on the CPU only.
# Paderborn configs go through scripts/run_combinations.py (all 10 combinations, D220), the others
# through scripts/pretrain.py followed by scripts/evaluate.py.
# Skips a config that has already finished, so the queue can be restarted after an interruption
# (an unfinished Paderborn config is run again from combination 0).
# Everything printed goes to runs\queue_d256_<queue>.log as well. Set Windows not to sleep first.

param([ValidateSet("gpu", "cpu")][string]$Queue = "gpu")

$queues = @{
    gpu = @(
        "tfc_paderborn_positives_lambda0",                                        # missing arm (D227), ~25 min
        "ts2vec_hust", "tstcc_hust", "simmtm_hust", "patchtst_hust", "frera_hust", "tloss_hust",
        "ts2vec_paderborn", "tstcc_paderborn", "simmtm_paderborn", "patchtst_paderborn",
        "tsjepa_hust",                                                            # ~3 h
        "tloss_paderborn", "frera_paderborn",                                     # ~7-10 h, ~15 h
        "tsjepa_paderborn"                                                        # ~28 h
    )
    cpu = @(
        "fft_paderborn", "envelope_paderborn", "catch22_paderborn", "minirocket_paderborn",  # reruns with D255 inputs
        "fft_hust", "envelope_hust", "catch22_hust", "minirocket_hust",
        "fft_xjtu_sy_stage", "envelope_xjtu_sy_stage", "catch22_xjtu_sy_stage", "minirocket_xjtu_sy_stage"
    )
}

$log = "runs\queue_d256_$Queue.log"
foreach ($name in $queues[$Queue]) {
    $combinations = $name -like "*_paderborn*"
    # finished = for Paderborn the combinations table exists; otherwise seed 0 stored and the run evaluated
    if ($combinations) {
        $done = Get-ChildItem runs -File -Filter "*_$($name)_combinations.csv" -ErrorAction SilentlyContinue
    } else {
        $done = Get-ChildItem runs -Directory -Filter "*_$name" -ErrorAction SilentlyContinue |
                Where-Object { (Test-Path (Join-Path $_.FullName "seed_0\Z_test.npy")) -and
                               (Test-Path (Join-Path $_.FullName "results.csv")) }
    }
    if ($done) { "skip $name (finished: $($done[0].Name))" | Tee-Object -FilePath $log -Append; continue }

    "=== $name  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
    # --no-sync: use the environment as it is installed; uv does not try to install changed dependencies first
    if ($combinations) {
        uv run --no-sync python scripts/run_combinations.py "configs/$name.yaml" --seeds 0 2>&1 | Tee-Object -FilePath $log -Append | Out-Host
        if ($LASTEXITCODE -ne 0) { "FAILED $name" | Tee-Object -FilePath $log -Append }
        continue
    }
    uv run --no-sync python scripts/pretrain.py "configs/$name.yaml" --seeds 0 2>&1 | Tee-Object -FilePath $log -Append |
        Tee-Object -Variable out | Out-Host  # shown, logged and kept in $out
    $match = $out | Select-String "^Done: (.+)$"
    if (-not $match) { "FAILED pretraining $name" | Tee-Object -FilePath $log -Append; continue }
    $folder = $match[0].Matches[0].Groups[1].Value  # this config's run folder
    uv run --no-sync python scripts/evaluate.py $folder 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
    if ($LASTEXITCODE -ne 0) { "FAILED evaluating $name ($folder)" | Tee-Object -FilePath $log -Append }
    else { "evaluated $folder" | Tee-Object -FilePath $log -Append }
}
"=== $Queue queue done  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
