function Invoke-TorchStage([hashtable]$Context) {
    Invoke-PythonCheck $Context 'Python'
    if (!(Test-VcRuntime)) { throw 'Run the System stage before installing native Python packages.' }
    if (!$Context.CheckOnly) {
        if (!(Test-Path -LiteralPath $Context.Uv)) { throw 'Run the Python stage first to install uv.' }
        Invoke-Checked $Context.Uv @('pip','install','--python',$Context.Python,'--index-url',
            "https://download.pytorch.org/whl/$($Context.Device)","torch==$($Context.TorchVersion)","torchvision==$($Context.VisionVersion)")
    }
    Invoke-PythonCheck $Context 'Torch'
    Check-Dependencies $Context
}
