function Invoke-PipelineStage([hashtable]$Context) {
    Invoke-PythonCheck $Context 'Packages'
    # Candidate filenames are long. Keep this subtree shallow on Windows.
    $segmentation = Join-Path $Context.RunRoot 'sam'
    $results = Join-Path $Context.StageOutput 'DBHResults'
    $device = if ($Context.Device -eq 'cpu') { 'cpu' } else { 'cuda:0' }
    Invoke-Checked $Context.Python @('-B','-X','utf8',(Join-Path $Context.Root 'codes/segmentation/batchProcess_SAM31_CPU.py'),
        '--dataset','all','--limit',[string]$Context.SampleCount,'--warmup','0','--device',$device,'--precision',$Context.Precision,
        '--data-dir',$Context.DataDir,'--output-dir',$segmentation)
    & (Join-Path $Context.Root 'codes/DBH/run/Run-DBH.ps1') -Method all -Dataset all -MaxFiles $Context.SampleCount -DataDir $Context.DataDir -SegmentationDir $segmentation -OutputDir $results -SaveMasks
    Assert-Measurements $results @('SAM3','MobileSAMv2') $Context.SampleCount
    Invoke-PythonCheck $Context 'Notebook' @('--results-dir',$results)
}
