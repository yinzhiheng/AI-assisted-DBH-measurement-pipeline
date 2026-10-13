#requires -Version 5.1
[CmdletBinding()]
param(
    [ValidateSet('All','System','Python','Torch','Packages','Cpp','Pipeline','Export')][string]$Stage = 'All',
    [ValidateSet('Full','Measurement','Python')][string]$Profile = 'Full',
    [ValidateSet('cpu','cu126')][string]$Device = 'cpu',
    [ValidateSet('sam31','export')][string]$Environment = 'sam31',
    [string]$VenvPath = '',
    [string]$DataDir = 'data',
    [ValidateRange(1,10)][int]$SampleCount = 1,
    [ValidateSet('float32','float16','bfloat16')][string]$Precision = 'float32',
    [switch]$CheckOnly,
    [switch]$IncludeExport,
    [switch]$InstallSystemPrerequisites,
    [switch]$SkipBuild,
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
Import-Module (Join-Path $PSHOME 'Modules/Microsoft.PowerShell.Utility/Microsoft.PowerShell.Utility.psd1')
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if ($env:OS -ne 'Windows_NT' -or ![Environment]::Is64BitProcess -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'Setup supports native Windows x64 only.'
}
if ($Environment -eq 'export' -and $Device -ne 'cpu') { throw 'The optional export environment uses CPU. Omit -Device cu126.' }
if ($Environment -eq 'export' -and $Stage -in @('Cpp','Pipeline')) { throw 'Cpp/Pipeline use -Environment sam31.' }
if ($Environment -eq 'sam31' -and $Stage -eq 'Export') { throw 'Use -Environment export for the Export stage.' }
if ($IncludeExport -and ($Stage -ne 'All' -or $Environment -ne 'sam31' -or $Profile -eq 'Measurement' -or $VenvPath)) {
    throw '-IncludeExport requires All, sam31, Full/Python, and the default environment paths.'
}
if ($Device -eq 'cpu' -and $Precision -ne 'float32') { throw 'CPU inference uses float32.' }
. (Join-Path $PSScriptRoot 'stages/Common.ps1')
foreach ($file in Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'stages') -Filter '*.ps1' | Where-Object Name -Match '^\d' | Sort-Object Name) { . $file.FullName }
$stageFunctions = @{
    System='Invoke-SystemStage'; Python='Invoke-PythonStage'; Torch='Invoke-TorchStage';
    Packages='Invoke-PackagesStage'; Cpp='Invoke-CppStage'; Pipeline='Invoke-PipelineStage'; Export='Invoke-ExportStage'
}
$sequence = if ($Stage -ne 'All') { @($Stage) }
elseif ($Environment -eq 'export') { @('System','Python','Torch','Packages','Export') }
elseif ($Profile -eq 'Measurement') { @('System','Cpp') }
elseif ($Profile -eq 'Python') { @('System','Python','Torch','Packages') }
else { @('System','Python','Torch','Packages','Cpp','Pipeline') }
$runRoot = Join-Path $repoRoot ('output/setup/' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff') + '-' + [guid]::NewGuid().ToString('N').Substring(0,6))
function New-StageContext([string]$Name, [string]$TargetDevice, [string]$CustomVenv) {
    $isExport = $Name -eq 'export'
    $venvName = if ($isExport) { 'mobilesamv2-export' } else { "sam31-$TargetDevice" }
    $venv = if ($CustomVenv) { $CustomVenv } else { ".venvs/$venvName" }
    if (![IO.Path]::IsPathRooted($venv)) { $venv = Join-Path $repoRoot $venv }
    $data = if ([IO.Path]::IsPathRooted($DataDir)) { $DataDir } else { Join-Path $repoRoot $DataDir }
    return @{
        Root=$repoRoot; RunRoot=$runRoot; EnvironmentRoot=$PSScriptRoot; Tools=(Join-Path $PSScriptRoot '.tools');
        Uv=(Join-Path $PSScriptRoot '.tools/uv-0.13.0/uv.exe'); EnvironmentName=$Name; Device=$TargetDevice;
        Venv=[IO.Path]::GetFullPath($venv); Python=(Join-Path ([IO.Path]::GetFullPath($venv)) 'Scripts/python.exe');
        PythonVersion=$(if ($isExport) { '3.10' } else { '3.13' });
        TorchVersion=$(if ($isExport) { '2.5.1+cpu' } else { "2.7.0+$TargetDevice" });
        VisionVersion=$(if ($isExport) { '0.20.1+cpu' } else { "0.22.0+$TargetDevice" });
        Requirements=$(if ($isExport) { @('python/mobilesamv2_export/requirements.txt') } else { @('python/sam31_cpu/requirements.txt','python/visualization/requirements.txt') });
        WithCpp=(!$isExport -and $Profile -ne 'Python'); CheckOnly=[bool]$CheckOnly;
        InstallSystemPrerequisites=[bool]$InstallSystemPrerequisites; SkipBuild=[bool]$SkipBuild;
        DataDir=[IO.Path]::GetFullPath($data); SampleCount=$SampleCount; Precision=$Precision; StageOutput=''
    }
}
$context = New-StageContext $Environment $Device $VenvPath
Write-Host "Repository: $repoRoot"
Write-Host "Stages: $($sequence -join ' -> '); environment: $Environment/$Device; check only: $CheckOnly"
Write-Host "Python: $($context.Python)"
if ($Plan) {
    if ($IncludeExport) { Write-Host 'Then: export/System -> Python -> Torch -> Packages -> Export' }
    Write-Host 'Preview only. No installation, checks or files are created.'
    return
}
$originalEnvironment = @{}
foreach ($name in @('UV_CACHE_DIR','UV_PYTHON_INSTALL_DIR','UV_PYTHON_INSTALL_REGISTRY','UV_PYTHON_INSTALL_BIN','UV_NO_CONFIG','UV_LINK_MODE','PYTHONDONTWRITEBYTECODE','PYTHONUTF8','PYTHONHOME','PYTHONPATH')) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name,'Process')
}
$stageResults = [Collections.Generic.List[object]]::new()
function Invoke-LoggedStage([hashtable]$Context, [string]$Name) {
    $Context.StageOutput = Join-Path $runRoot "$($Context.EnvironmentName)-$($Context.Device)/$Name"
    New-Item -ItemType Directory -Path $Context.StageOutput -Force | Out-Null
    $log = Join-Path $Context.StageOutput 'stage.log'
    Start-Transcript -LiteralPath $log | Out-Null
    $result = [ordered]@{ stage=$Name; environment=$Context.EnvironmentName; device=$Context.Device;
        check_only=$Context.CheckOnly; python=$Context.Python; started=(Get-Date).ToString('o');
        status='running'; error=$null; output=$Context.StageOutput; log=$log }
    try {
        Write-Host "Starting $Name ($($Context.EnvironmentName)/$($Context.Device))."
        & $stageFunctions[$Name] $Context
        $result.status = 'passed'
        Write-Host "PASS: $Name"
    } catch {
        $result.status = 'failed'
        $result.error = ($_ | Out-String)
        Write-Host "FAIL: $Name. $($_.Exception.Message)"
        Write-Host "See $log"
        throw
    } finally {
        if ($result.status -eq 'running') {
            $result.status = 'interrupted'
            $result.error = 'The stage was interrupted before completion.'
        }
        $result['finished'] = (Get-Date).ToString('o')
        $stageResults.Add([pscustomobject]$result)
        $result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Context.StageOutput 'status.json') -Encoding UTF8
        ConvertTo-Json -InputObject @($stageResults.ToArray()) -Depth 5 | Set-Content -LiteralPath (Join-Path $runRoot 'summary.json') -Encoding UTF8
        Stop-Transcript | Out-Null
    }
}
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.cache/uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.tools/python'
    $env:UV_PYTHON_INSTALL_REGISTRY = '0'; $env:UV_PYTHON_INSTALL_BIN = '0'
    $env:UV_NO_CONFIG = '1'; $env:UV_LINK_MODE = 'hardlink'
    $env:PYTHONDONTWRITEBYTECODE = '1'; $env:PYTHONUTF8 = '1'
    $env:PYTHONHOME = $null; $env:PYTHONPATH = $null
    foreach ($name in $sequence) { Invoke-LoggedStage $context $name }
    if ($IncludeExport) {
        $exportContext = New-StageContext 'export' 'cpu' ''
        foreach ($name in @('System','Python','Torch','Packages','Export')) { Invoke-LoggedStage $exportContext $name }
    }
    Write-Host "All requested stages passed. Results: $runRoot"
} finally {
    foreach ($name in $originalEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$originalEnvironment[$name],'Process') }
}
