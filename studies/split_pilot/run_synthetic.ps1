# Runs the synthetic pilot configs (D198, D199, D204): pretraining, then the standard evaluation.
# Start from the repository root:
#     powershell -ExecutionPolicy Bypass -File studies\split_pilot\run_synthetic.ps1
# By default the minimal set of D204, in this order: variant A, then B1, then C1 (all synthetic splits),
# with seed 0 only (pretrain.py --seeds 0, D201). Other variants or seeds, e.g.  -Variants B10,C10 -Seeds 0,1,2
# Skips a config whose run already finished (last seed stored and evaluated), so it can be restarted after an interruption.
# Everything printed goes to runs\split_pilot_synthetic.log as well.

param([string[]]$Variants = @("A", "B1", "C1"), [string[]]$Seeds = @("0"))

# With 'powershell -File', a list like B1live,C100 arrives as one string: split it at the commas.
$Variants = $Variants | ForEach-Object { $_ -split "," } | Where-Object { $_ }
$Seeds = $Seeds | ForEach-Object { $_ -split "," } | Where-Object { $_ }

$log = "runs\split_pilot_synthetic.log"
foreach ($variant in $Variants) {
    $configs = Get-ChildItem "studies\split_pilot\configs\pilot_$($variant)_synthetic_*.yaml" -ErrorAction SilentlyContinue | Sort-Object Name
    if (-not $configs) { "NO CONFIGS for variant '$variant'" | Tee-Object -FilePath $log -Append; continue }
    foreach ($cfg in $configs) {
        $name = $cfg.BaseName
        # finished = the last requested seed is stored and the run was evaluated (a run with more seeds counts too)
        $last = $Seeds[-1]
        $done = Get-ChildItem runs -Directory -Filter "*_$name" -ErrorAction SilentlyContinue |
                Where-Object { (Test-Path (Join-Path $_.FullName "seed_$last\Z_test.npy")) -and
                               (Test-Path (Join-Path $_.FullName "results.csv")) }
        if ($done) { "skip $name (finished: $($done[0].Name))" | Tee-Object -FilePath $log -Append; continue }

        "=== $name  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
        # --no-sync: use the environment as it is installed; uv does not try to install changed dependencies first
        $out = uv run --no-sync python scripts/pretrain.py $cfg.FullName --seeds @Seeds 2>&1 | Tee-Object -FilePath $log -Append
        $match = $out | Select-String "^Done: (.+)$"
        if (-not $match) { "FAILED pretraining $name" | Tee-Object -FilePath $log -Append; continue }
        $folder = $match[0].Matches[0].Groups[1].Value  # this config's run folder (set anew for every config)
        uv run --no-sync python scripts/evaluate.py $folder 2>&1 | Tee-Object -FilePath $log -Append | Out-Null
        "evaluated $folder" | Tee-Object -FilePath $log -Append
    }
}
"=== all done  $(Get-Date -Format s)" | Tee-Object -FilePath $log -Append
