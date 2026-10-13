function Invoke-ExportStage([hashtable]$Context) {
    Invoke-PythonCheck $Context 'Packages'
    $models = Join-Path $Context.StageOutput 'onnx'
    if ($Context.CheckOnly) { $models = Join-Path $Context.Root 'model/MobileSAMv2/onnx' }
    else {
        Invoke-Checked $Context.Python @('-B','-X','utf8',(Join-Path $Context.Root 'codes/segmentation/MobileSAMv2_export/export_dbh_onnx.py'),'--output-dir',$models)
    }
    Invoke-PythonCheck $Context 'Export' @('--model-dir',$models)
}
