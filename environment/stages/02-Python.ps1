function Invoke-PythonStage([hashtable]$Context) {
    if (!$Context.CheckOnly) {
        Install-Uv $Context
        if (!(Test-Path -LiteralPath $Context.Python)) {
            if (Test-Path -LiteralPath $Context.Venv) { throw "Incomplete environment: $($Context.Venv). Rename it before retrying; setup will not delete it." }
            Invoke-Checked $Context.Uv @('venv','--managed-python','--python',$Context.PythonVersion,$Context.Venv)
        }
    }
    Invoke-PythonCheck $Context 'Python'
}
