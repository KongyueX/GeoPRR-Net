param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)]
    [ValidateSet('Audit','Warm','Full','Cache','Controls','Evaluate','Summarize')][string]$Phase,
    [int[]]$Seeds = @(20262020,20262021,20262022)
)
$ErrorActionPreference = 'Stop'
$ConfigPath = (Resolve-Path $Config).Path
Set-Location (Split-Path $PSScriptRoot -Parent)
$Settings = Get-Content -Raw $ConfigPath | ConvertFrom-Json
$Python = $Settings.python
if (-not (Test-Path $Python)) { throw "Python does not exist: $Python" }
$RunRoot = $Settings.output_root
if (-not $RunRoot) { throw 'Set output_root in the run configuration.' }
function Required($Value, [string]$Name) {
    if ($null -eq $Value -or "$Value" -eq '') { throw "Fill $Name from the training computer's actual configuration." }
    return $Value
}
function Invoke-Experiment([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $($Arguments -join ' ')" }
}
$Device = Required $Settings.device 'device'
$Workers = [int]$Settings.workers
$Cohorts = @('syncg','rf100','industrial')
$Variants = @('no_geometry_fusion','no_relational_transport','no_polar_evidence','no_geometry_fixed_routing')
$Methods = @('full','base_only','polar_only','relational_only','uniform','fixed_prior','linear_gate','scalar_gate','dual_view')
$Policy = $Settings.availability_policy
$Tau = $Settings.smooth_max_temperature
if ($Phase -in @('Full','Cache')) { $Policy = Required $Policy 'availability_policy' }
if ($Phase -in @('Full','Controls')) { $Tau = Required $Tau 'smooth_max_temperature' }
if ($Phase -eq 'Summarize') {
    foreach ($Cohort in $Cohorts) {
        $SummaryMethods = $Methods
        if ($Cohort -eq 'syncg') { $SummaryMethods = $Methods + $Variants }
        $Files = @()
        foreach ($Seed in @(20262020,20262021,20262022)) {
            foreach ($Method in $SummaryMethods) { $Files += "$RunRoot/$Seed/predictions/$Cohort/$Method.csv" }
        }
        $BootstrapSeed = @{syncg=20260907;rf100=20261007;industrial=20261107}[$Cohort]
        Invoke-Experiment (@('-m','experiments.revision_diagnostics','summarize','--predictions') + $Files + @('--reference','full','--bootstrap-seed',"$BootstrapSeed",'--output-dir',"$RunRoot/summary/$Cohort"))
    }
    exit 0
}
foreach ($Seed in $Seeds) {
    if ($Seed -notin @(20262020,20262021,20262022)) { throw "Unexpected source seed: $Seed" }
    $Entry = $Settings.seeds.PSObject.Properties["$Seed"].Value
    if ($null -eq $Entry) { throw "Missing configuration for seed $Seed" }
    $SeedRoot = "$RunRoot/$Seed"
    $WarmCheckpoint = "$SeedRoot/warm.pt"
    $FullCheckpoint = "$SeedRoot/full/full/ema.pt"
    $TrainCache = "$SeedRoot/full/source_train_cache_ema_polar.pt"
    switch ($Phase) {
        'Audit' {
            $Foundation = Required $Entry.foundation "seeds.$Seed.foundation"
            $Bank = Required $Entry.source_r2mt "seeds.$Seed.source_r2mt"
            Invoke-Experiment @('-m','experiments.revision_diagnostics','audit','--checkpoints',$Foundation,$Bank,'--output',"$SeedRoot/input_audit.json")
        }
        'Warm' {
            Invoke-Experiment @('-m','experiments.revision_full_training','warm-polar',
                '--foundation',(Required $Entry.foundation "seeds.$Seed.foundation"),
                '--official-train-manifest',(Required $Settings.source_train_manifest 'source_train_manifest'),
                '--warm-manifest',(Required $Settings.warm.manifest 'warm.manifest'),
                '--expected-warm-images',"$($Settings.warm.expected_images)",
                '--epochs',"$($Settings.warm.epochs)",
                '--learning-rate',"$(Required $Settings.warm.learning_rate 'warm.learning_rate')",
                '--weight-decay',"$(Required $Settings.warm.weight_decay 'warm.weight_decay')",
                '--seed',"$Seed",'--output',$WarmCheckpoint,'--device',$Device,'--workers',"$Workers")
        }
        'Full' {
            Invoke-Experiment @('-m','experiments.train_unified_pointer_reader',
                '--population-mode','official_full',
                '--official-train-manifest',(Required $Settings.source_train_manifest 'source_train_manifest'),
                '--source-r2mt',(Required $Entry.source_r2mt "seeds.$Seed.source_r2mt"),
                '--warm-polar',$WarmCheckpoint,'--availability-policy',$Policy,
                '--polar-epochs','10','--router-stage-one-epochs','40','--router-stage-two-epochs','120',
                '--smooth-max-temperature',"$Tau",'--seed',"$Seed",'--device',$Device,
                '--workers',"$Workers",'--output-root',"$SeedRoot/full")
        }
        'Cache' {
            foreach ($Cohort in $Cohorts) {
                $Manifest = Required $Settings.evaluation_manifests.PSObject.Properties[$Cohort].Value "evaluation_manifests.$Cohort"
                Invoke-Experiment @('-m','experiments.revision_cache','--checkpoint',$FullCheckpoint,
                    '--manifest',$Manifest,'--cohort',$Cohort,'--seed',"$Seed",'--device',$Device,
                    '--workers',"$Workers",'--availability-policy',$Policy,'--output',"$SeedRoot/cache/$Cohort.pt")
            }
            foreach ($Variant in $Variants) {
                Invoke-Experiment @('-m','experiments.revision_cache','--checkpoint',"$SeedRoot/full/$Variant/ema.pt",
                    '--manifest',$Settings.evaluation_manifests.syncg,'--cohort','syncg','--seed',"$Seed",'--device',$Device,
                    '--workers',"$Workers",'--availability-policy',$Policy,'--output',"$SeedRoot/cache/syncg_$Variant.pt")
            }
        }
        'Controls' {
            foreach ($Method in @('linear_gate','scalar_gate','dual_view')) {
                $Arguments = @('-m','experiments.revision_controls','train','--cache',$TrainCache,
                    '--method',$Method,'--seed',"$Seed",'--device',$Device,'--output',"$SeedRoot/controls/$Method")
                if ($Method -ne 'dual_view') { $Arguments += @('--smooth-max-temperature',"$Tau") }
                Invoke-Experiment $Arguments
            }
        }
        'Evaluate' {
            foreach ($Cohort in $Cohorts) {
                foreach ($Method in $Methods) {
                    $Arguments = @('-m','experiments.revision_controls','evaluate',
                        '--cache',"$SeedRoot/cache/$Cohort.pt",'--method',$Method,'--seed',"$Seed",'--device',$Device,
                        '--output-csv',"$SeedRoot/predictions/$Cohort/$Method.csv")
                    if ($Method -eq 'dual_view') { $Arguments += @('--checkpoint',"$SeedRoot/controls/$Method/terminal.pt") }
                    if ($Method -in @('linear_gate','scalar_gate')) { $Arguments += @('--checkpoint',"$SeedRoot/controls/$Method/ema.pt") }
                    Invoke-Experiment $Arguments
                }
            }
            foreach ($Variant in $Variants) {
                Invoke-Experiment @('-m','experiments.revision_controls','evaluate',
                    '--cache',"$SeedRoot/cache/syncg_$Variant.pt",'--method','full','--seed',"$Seed",'--device',$Device,
                    '--output-csv',"$SeedRoot/predictions/syncg/$Variant.csv")
            }
        }
    }
}
