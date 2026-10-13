function Invoke-Checked([string]$Program, [string[]]$Arguments) {
    & $Program @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Command failed (exit $LASTEXITCODE): $Program $($Arguments -join ' ')" }
}

function Assert-Files([hashtable]$Context, [string[]]$Paths) {
    foreach ($path in $Paths) {
        if (!(Test-Path -LiteralPath (Join-Path $Context.Root $path) -PathType Leaf)) {
            throw "Missing bundled file: $path. Copy the complete project distribution."
        }
    }
}

function Get-BuildTools {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { return $null }
    return (& $vswhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find 'MSBuild\**\Bin\MSBuild.exe' | Select-Object -First 1)
}

function Test-WindowsSdk {
    $key = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots' -ErrorAction SilentlyContinue
    if (!$key -or !$key.PSObject.Properties['KitsRoot10']) { return $false }
    $include = Join-Path $key.KitsRoot10 'Include'
    if (!(Test-Path -LiteralPath $include)) { return $false }
    return @((Get-ChildItem -LiteralPath $include -Directory) | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'um/Windows.h') }).Count -gt 0
}

function Test-VcRuntime {
    $runtime = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    return ($null -ne $runtime -and $runtime.Installed -eq 1 -and [version]$runtime.Version.TrimStart('v') -ge [version]'14.44.35211')
}

function Install-MicrosoftPackage([hashtable]$Context, [string]$Url, [string]$Name, [string]$Arguments) {
    $download = Join-Path $Context.Tools $Name
    New-Item -ItemType Directory -Path $Context.Tools -Force | Out-Null
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $download
    $signature = Get-AuthenticodeSignature -LiteralPath $download
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw "Invalid Microsoft installer signature: $download"
    }
    $process = Start-Process -FilePath $download -ArgumentList $Arguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -in @(3010,1641)) { throw 'Restart Windows, then repeat this stage.' }
    if ($process.ExitCode -ne 0) { throw "$Name failed with exit code $($process.ExitCode). See Windows TEMP installer logs." }
}

function Install-Uv([hashtable]$Context) {
    if (Test-Path -LiteralPath $Context.Uv) {
        $installed = & $Context.Uv --version
        if ($LASTEXITCODE -ne 0 -or $installed -notmatch '^uv 0\.13\.0( |$)') { throw "Unexpected uv: $installed" }
        return
    }
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/astral-sh/uv/releases/tags/0.13.0'
    $asset = @($release.assets | Where-Object name -eq 'uv-x86_64-pc-windows-msvc.zip')
    if ($asset.Count -ne 1 -or $asset[0].digest -notmatch '^sha256:([0-9a-f]{64})$') { throw 'The official uv release has no usable SHA-256 digest.' }
    $expected = $Matches[1]
    $url = 'https://github.com/astral-sh/uv/releases/download/0.13.0/uv-x86_64-pc-windows-msvc.zip'
    if ($asset[0].browser_download_url -ne $url) { throw 'Unexpected uv download URL.' }
    New-Item -ItemType Directory -Path $Context.Tools -Force | Out-Null
    $archive = Join-Path $Context.Tools 'uv-0.13.0.zip'
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expected) { throw 'uv download checksum mismatch.' }
    Expand-Archive -LiteralPath $archive -DestinationPath (Split-Path $Context.Uv) -Force
    Invoke-Checked $Context.Uv @('--version')
}

function Invoke-PythonCheck([hashtable]$Context, [string]$Kind, [string[]]$Extra = @()) {
    if (!(Test-Path -LiteralPath $Context.Python)) { throw 'Python environment missing. Run the Python stage first.' }
    $arguments = @('-B','-X','utf8',(Join-Path $Context.EnvironmentRoot 'checks/check_python.py'),
        '--stage',$Kind,'--environment',$Context.EnvironmentName,'--device',$Context.Device,
        '--venv',$Context.Venv,'--output-dir',$Context.StageOutput)
    Invoke-Checked $Context.Python ($arguments + $Extra)
}

function Check-Dependencies([hashtable]$Context) {
    if (!(Test-Path -LiteralPath $Context.Uv)) { throw 'uv is missing. Run the Python stage first.' }
    Invoke-Checked $Context.Uv @('pip','check','--python',$Context.Python)
}

function Assert-Measurements([string]$Directory, [string[]]$Methods, [int]$Count) {
    foreach ($method in $Methods) {
        foreach ($dataset in @('natural','plot')) {
            $path = Join-Path $Directory "$method/results_${method}_${dataset}.csv"
            $rows = @(Import-Csv -LiteralPath $path)
            if ($rows.Count -ne $Count) { throw "$path has $($rows.Count) rows; expected $Count." }
            foreach ($row in $rows) {
                $dbh = [double]::Parse($row.dbhSAM_UV_cm, [Globalization.CultureInfo]::InvariantCulture)
                if ([double]::IsNaN($dbh) -or [double]::IsInfinity($dbh) -or $dbh -le 0) { throw "Invalid DBH in $path for $($row.uniqueName)." }
            }
            if (@($rows.uniqueName | Select-Object -Unique).Count -ne $Count) { throw "Duplicate sample IDs in $path" }
            Write-Host "PASS: $method/$dataset produced $Count positive finite measurement(s)."
        }
    }
}
