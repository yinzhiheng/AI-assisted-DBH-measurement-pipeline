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
    $paths = @(& $vswhere -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
    if ($LASTEXITCODE -ne 0) { throw 'vswhere failed while locating VS 2022 C++ tools.' }
    foreach ($path in $paths) {
        $msbuild = Join-Path $path 'MSBuild/Current/Bin/MSBuild.exe'
        $versionFile = Join-Path $path 'VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt'
        if (!(Test-Path -LiteralPath $msbuild -PathType Leaf) -or !(Test-Path -LiteralPath $versionFile -PathType Leaf)) { continue }
        $version = (Get-Content -LiteralPath $versionFile -Raw).Trim()
        if ($version -notmatch '^14\.[34]\d\.\d+$') { continue }
        $toolRoot = Join-Path $path "VC/Tools/MSVC/$version"
        $missing = @('bin/Hostx64/x64/cl.exe','bin/Hostx64/x64/link.exe','include/vector','lib/x64/libcmt.lib') |
            Where-Object { !(Test-Path -LiteralPath (Join-Path $toolRoot $_) -PathType Leaf) }
        if (!$missing) { return $msbuild }
    }
    return $null
}

function Get-Vs2022Instance {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere)) { return $null }
    # Include installations that have not yet acquired the C++ workload.
    $json = & $vswhere -latest -all -products '*' -version '[17.0,18.0)' -format json
    if ($LASTEXITCODE -ne 0) { throw 'vswhere failed while locating an existing VS 2022 installation.' }
    return ($json | ConvertFrom-Json | Select-Object -First 1)
}

function Get-WindowsSdkRoots {
    foreach ($registryPath in @('HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots',
            'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots')) {
        $key = Get-ItemProperty -LiteralPath $registryPath -ErrorAction SilentlyContinue
        if ($key -and $key.PSObject.Properties['KitsRoot10'] -and $key.KitsRoot10) { $key.KitsRoot10 }
    }
    Join-Path ${env:ProgramFiles(x86)} 'Windows Kits/10'
}

function Get-WindowsSdk {
    foreach ($root in (Get-WindowsSdkRoots | Select-Object -Unique)) {
        $include = Join-Path $root 'Include'
        if (!(Test-Path -LiteralPath $include -PathType Container)) { continue }
        foreach ($directory in (Get-ChildItem -LiteralPath $include -Directory | Where-Object Name -Match '^10\.0\.\d+\.0$' | Sort-Object { [version]$_.Name } -Descending)) {
            $version = $directory.Name
            $required = @("Include/$version/um/Windows.h", "Include/$version/shared/sdkddkver.h",
                "Include/$version/ucrt/stdio.h", "Lib/$version/um/x64/kernel32.lib",
                "Lib/$version/ucrt/x64/ucrt.lib", "bin/$version/x64/rc.exe")
            $missing = @($required | Where-Object { !(Test-Path -LiteralPath (Join-Path $root $_) -PathType Leaf) })
            if ($missing.Count -eq 0) { return [pscustomobject]@{ Root=$root; Version=$version } }
        }
    }
    return $null
}

function Test-WindowsSdk { return $null -ne (Get-WindowsSdk) }

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-VcRuntime {
    $runtime = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64' -ErrorAction SilentlyContinue
    return ($null -ne $runtime -and $runtime.Installed -eq 1 -and [version]$runtime.Version.TrimStart('v') -ge [version]'14.44.35211')
}

function Install-MicrosoftPackage([hashtable]$Context, [string]$Url, [string]$Name, [string]$Arguments) {
    if ($Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { throw 'System installation requires -InstallSystemPrerequisites without -CheckOnly.' }
    $download = Join-Path $Context.Tools $Name
    New-Item -ItemType Directory -Path $Context.Tools -Force | Out-Null
    Write-Host "Downloading Microsoft installer: $Name"
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $download
    $signature = Get-AuthenticodeSignature -LiteralPath $download
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw "Invalid Microsoft installer signature: $download"
    }
    $startOptions = @{ FilePath=$download; ArgumentList=$Arguments; PassThru=$true; WindowStyle='Hidden'; WorkingDirectory=$Context.Tools }
    if (!(Test-Administrator)) {
        Write-Host 'Approve the Windows administrator (UAC) prompt to install system prerequisites.'
        $startOptions.Verb = 'RunAs'
    }
    Write-Host "Installing $Name. This can take several minutes; keep this window open."
    Write-Host "Installer arguments: $Arguments"
    $process = Start-Process @startOptions
    # The VS bootstrapper receives --wait, so it stays alive until its installer finishes.
    while (!$process.WaitForExit(15000)) { Write-Host "Still installing $Name ..." }
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    Write-Host "$Name finished with exit code $exitCode."
    if ($exitCode -in @(3010,1641)) { throw 'Installation succeeded but Windows needs a restart. Restart Windows, then repeat this stage.' }
    if ($exitCode -ne 0) {
        $hint = switch ($exitCode) {
            740 { 'Administrator access is required. Run PowerShell as administrator and repeat this stage.' }
            1001 { 'Another Visual Studio installer is running. Wait for it to finish, then repeat this stage.' }
            1618 { 'Another installation is running. Wait for it to finish, then repeat this stage.' }
            1602 { 'Installation was canceled. Repeat this stage and allow installation to finish.' }
            5003 { 'A download failed. Check the network connection, then repeat this stage.' }
            default { 'Inspect the Microsoft installer logs for the failure details.' }
        }
        throw "$Name failed with exit code $exitCode. $hint Logs: $env:TEMP\dd_*.log (or the administrator account's TEMP folder if different credentials were used)."
    }
}

function Install-Vs2022BuildTools([hashtable]$Context) {
    $instance = Get-Vs2022Instance
    $name = 'vs_BuildTools.exe'
    $workload = 'Microsoft.VisualStudio.Workload.VCTools'
    $sdk = 'Microsoft.VisualStudio.Component.Windows11SDK.26100'
    $arguments = '--quiet --wait --norestart'
    if ($instance) {
        $installers = @{
            'Microsoft.VisualStudio.Product.BuildTools'='vs_BuildTools.exe'
            'Microsoft.VisualStudio.Product.Community'='vs_Community.exe'
            'Microsoft.VisualStudio.Product.Professional'='vs_Professional.exe'
            'Microsoft.VisualStudio.Product.Enterprise'='vs_Enterprise.exe'
        }
        $name = $installers[$instance.productId]
        if (!$name) { throw "Unsupported VS 2022 product: $($instance.productId)" }
        if ($instance.productId -ne 'Microsoft.VisualStudio.Product.BuildTools') { $workload = 'Microsoft.VisualStudio.Workload.NativeDesktop' }
        # Older VS 2022 catalogs do not contain the 26100 SDK. The projects target
        # Windows SDK 10.0 and also support 19041, available throughout VS 2022.
        if ([version]$instance.installationVersion -lt [version]'17.12') { $sdk = 'Microsoft.VisualStudio.Component.Windows10SDK.19041' }
        $path = $instance.installationPath.TrimEnd('\')
        $channel = $instance.channelId
        if ($path.Contains('"') -or $channel -notmatch '^VisualStudio\.17\.[A-Za-z0-9.]+$') { throw 'Unexpected VS 2022 installation path/channel.' }
        $arguments = "modify --installPath `"$path`" --channelId $channel $arguments"
        Write-Host "Adding C++ build components to existing VS 2022: $path"
    } else {
        Write-Host 'Installing standalone Visual Studio 2022 Build Tools (the full Visual Studio IDE is not required).'
    }
    $arguments += " --add $workload --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add $sdk"
    Install-MicrosoftPackage $Context "https://aka.ms/vs/17/release/$name" $name $arguments
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
