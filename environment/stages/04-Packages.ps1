function Invoke-PackagesStage([hashtable]$Context) {
    Invoke-PythonCheck $Context 'Torch'
    if (!$Context.CheckOnly) {
        $constraints = Join-Path $Context.StageOutput 'torch-constraints.txt'
        @("torch==$($Context.TorchVersion)","torchvision==$($Context.VisionVersion)") | Set-Content -LiteralPath $constraints -Encoding ASCII
        $arguments = @('pip','install','--python',$Context.Python,'--index-url','https://pypi.org/simple','--constraint',$constraints)
        foreach ($recipe in $Context.Requirements) { $arguments += @('-r',(Join-Path $Context.EnvironmentRoot $recipe)) }
        Invoke-Checked $Context.Uv $arguments
    }
    Check-Dependencies $Context
    Invoke-PythonCheck $Context 'Packages'
    $freeze = & $Context.Uv pip freeze --python $Context.Python
    if ($LASTEXITCODE -ne 0) { throw 'Could not record installed package versions.' }
    $freeze | Set-Content -LiteralPath (Join-Path $Context.StageOutput 'installed-packages.txt') -Encoding UTF8
}
