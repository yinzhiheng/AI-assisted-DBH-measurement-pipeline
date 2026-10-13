[CmdletBinding()]
param([string]$MSBuildPath = '')
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
if (!$MSBuildPath) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { throw 'Install Visual Studio 2022 with Desktop development with C++.' }
    $MSBuildPath = & $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1
}
if (!$MSBuildPath) { throw 'MSBuild was not found.' }
# Some launchers pass both Path and PATH. MSBuild's case-insensitive environment
# dictionary rejects that duplicate; normalize the current process before spawning it.
$buildSearchPath = [Environment]::GetEnvironmentVariable('PATH', 'Process')
[Environment]::SetEnvironmentVariable('PATH', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $null, 'Process')
[Environment]::SetEnvironmentVariable('Path', $buildSearchPath, 'Process')
& $MSBuildPath (Join-Path $projectRoot 'codes/DBH/DBHCalculationBatch.sln') '/p:Configuration=Release' '/p:Platform=x64' '/m:2' '/nr:false' '/verbosity:minimal'
if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
