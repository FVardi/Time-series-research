# Runs the XJTU-SY configs under D266 (12 windows of 2,560 points per minute, mean per minute for the probes),
# seed 0 (D239). Two queues, which can run at the same time in two terminals from the repository root:
#     powershell -ExecutionPolicy Bypass -File scripts\queue_xjtu.ps1 -Queue gpu
#     powershell -ExecutionPolicy Bypass -File scripts\queue_xjtu.ps1 -Queue cpu
# gpu: the self-supervised methods on the RUL and health-stage tasks, cheapest first, then the TF-C variants.
# cpu: the RUL baselines (the stage baselines are in scripts\queue_d256.ps1 -Queue cpu).
# Each config: scripts/pretrain.py, then scripts/evaluate.py. Finished configs are skipped on a restart.
# Everything printed goes to runs\queue_xjtu_<queue>.log as well. Set Windows not to sleep first.

param([ValidateSet("gpu", "cpu")][string]$Queue = "gpu")

$queues = @{
    gpu = @(
        "ts2vec_xjtu_sy", "ts2vec_xjtu_sy_stage", "tfc_xjtu_sy", "tfc_xjtu_sy_stage",
        "tloss_xjtu_sy", "tloss_xjtu_sy_stage", "tsjepa_xjtu_sy", "tsjepa_xjtu_sy_stage",
        "tstcc_xjtu_sy", "tstcc_xjtu_sy_stage", "simmtm_xjtu_sy", "simmtm_xjtu_sy_stage",
        "patchtst_xjtu_sy", "patchtst_xjtu_sy_stage",
        "tfc_xjtu_sy_paper", "tfc_xjtu_sy_left_out", "tfc_xjtu_sy_left_out_x3", "tfc_xjtu_sy_left_out_lambda0", "tfc_xjtu_sy_positives_lambda0",
        "tfc_xjtu_sy_stage_paper", "tfc_xjtu_sy_stage_left_out", "tfc_xjtu_sy_stage_left_out_x3", "tfc_xjtu_sy_stage_left_out_lambda0", "tfc_xjtu_sy_stage_positives_lambda0"
    )
    cpu = @("fft_xjtu_sy", "envelope_xjtu_sy", "catch22_xjtu_sy", "minirocket_xjtu_sy")
}

$log = "runs\queue_xjtu_$Queue.log"
foreach ($name in $queues[$Queue]) {
    # finished = seed 0 stored and the run evaluated
    $done = Get-ChildItem runs -Directory -Filter "*_$name" -ErrorAction SilentlyContinue |
            Where-Object { (Test-Path (Join-Path $_.FullName "seed_0\Z_test.npy")) -and
                           (Test-Path (Join-Path $_.FullName "results.csv")) }
    if ($done) { "skip $name (finished: $($done[0].Name))" | Tee-Object -FilePath $log -Append; continue }

    "=== $name  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
    # --no-sync: use the environment as it is installed; uv does not try to install changed dependencies first
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
