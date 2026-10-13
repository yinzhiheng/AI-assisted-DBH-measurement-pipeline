[CmdletBinding()]
param(
    [ValidateSet('all', 'MobileSAMv2', 'SAM3')][string]$Method = 'all',
    [ValidateSet('all', 'natural', 'plotted')][string]$Dataset = 'all',
    [ValidateRange(0, 2147483647)][int]$MaxFiles = 0,
    [string]$OutputDir = '',
    [string]$DataDir = 'data',
    [string]$SegmentationDir = 'output/SAM3SegResults',
    [switch]$SaveMasks
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$methods = if ($Method -eq 'all') { @('MobileSAMv2', 'SAM3') } else { @($Method) }
$datasets = if ($Dataset -eq 'all') { @('plotted', 'natural') } else { @($Dataset) }
if (!$OutputDir) { $OutputDir = Join-Path $projectRoot 'output/DBHResults' }
elseif (![IO.Path]::IsPathRooted($OutputDir)) { $OutputDir = Join-Path $projectRoot $OutputDir }
if (![IO.Path]::IsPathRooted($DataDir)) { $DataDir = Join-Path $projectRoot $DataDir }
if (![IO.Path]::IsPathRooted($SegmentationDir)) { $SegmentationDir = Join-Path $projectRoot $SegmentationDir }
foreach ($methodName in $methods) {
    if ($methodName -eq 'MobileSAMv2') {
        $modelFolder = Join-Path $projectRoot 'model/MobileSAMv2/onnx'
        foreach ($modelFile in @('mobile_sam_preprocess.onnx', 'mobile_sam.onnx')) {
            if (!(Test-Path -LiteralPath (Join-Path $modelFolder $modelFile))) {
                throw "Missing MobileSAMv2 model: $modelFile"
            }
        }
    }
    foreach ($datasetName in $datasets) {
        $programDataset = if ($datasetName -eq 'plotted') { 'Plot' } else { 'Natural' }
        $csvDataset = if ($datasetName -eq 'plotted') { 'plot' } else { 'natural' }
        $exePath = Join-Path $projectRoot "codes/DBH/build/Release/batchProcess${programDataset}DatasetBy${methodName}.exe"
        if (!(Test-Path -LiteralPath $exePath)) { throw "Build codes/DBH/DBHCalculationBatch.sln first: $exePath" }
        if ($methodName -eq 'SAM3') {
            $selectionPath = Join-Path $SegmentationDir 'reports/target_selection.json'
            $selectionRecords = $null
            if (Test-Path -LiteralPath $selectionPath) {
                $selection = Get-Content -LiteralPath $selectionPath -Raw -Encoding UTF8 | ConvertFrom-Json
                $selectionRecords = $selection.records.$datasetName
            }
            $rawDataset = if ($datasetName -eq 'plotted') { 'plottedField' } else { 'naturalField' }
            $depthFiles = @(Get-ChildItem -LiteralPath (Join-Path $DataDir "rawdata/depth_$rawDataset") -Filter 'combine-*.png' -File | Sort-Object Name)
            if ($MaxFiles -gt 0) { $depthFiles = @($depthFiles | Select-Object -First $MaxFiles) }
            foreach ($depthFile in $depthFiles) {
                $sampleId = $depthFile.BaseName.Substring(8)
                $maskFile = Join-Path $SegmentationDir "targetTreeMasks/$datasetName/${sampleId}_selected_mask.png"
                if (!(Test-Path -LiteralPath $maskFile)) { throw "Missing SAM31 target mask. Run segmentation/selection first: $maskFile" }
                if (Test-Path -LiteralPath $selectionPath) {
                    $record = if ($null -ne $selectionRecords) { $selectionRecords.PSObject.Properties[$sampleId] } else { $null }
                    if ($null -eq $record -or $record.Value.status -ne 'selected') {
                        throw "No successful target selection recorded for $sampleId. Resolve the selection error before running DBH."
                    }
                }
            }
        }
        $methodOutput = Join-Path $OutputDir $methodName
        New-Item -ItemType Directory -Path $methodOutput -Force | Out-Null
        $csvPath = Join-Path $methodOutput "results_${methodName}_${csvDataset}.csv"
        $runArgs = @($projectRoot, $csvPath, [string]$MaxFiles, $DataDir)
        if ($methodName -eq 'SAM3') { $runArgs += (Join-Path $SegmentationDir 'targetTreeMasks') }
        if ($SaveMasks -and $methodName -eq 'MobileSAMv2') {
            $runArgs += @('--save-masks', '--debug-dir', (Join-Path $methodOutput "masks/$datasetName"))
        }
        Write-Host "Running $methodName on $datasetName (MaxFiles=$MaxFiles; 0 means all)."
        & $exePath @runArgs
        if ($LASTEXITCODE -ne 0) { throw "Batch failed: $exePath, exit $LASTEXITCODE" }
        Write-Host "Saved $csvPath"
    }
}
