#requires -Version 5.1
# Optional add-on for the staged Setup-Windows.ps1 dispatcher.
# Keep Common.ps1, 01-System.ps1 and every other original file unchanged.
# The dispatcher loads this file last. Original successful checks remain authoritative.
if (Get-Variable -Name DbhRecoveryOriginalSystem -Scope Script -ErrorAction SilentlyContinue) { return }
if (!(Test-Path Function:Test-WindowsSdk) -or !(Test-Path Function:Invoke-SystemStage)) {
    throw 'Run environment/Setup-Windows.cmd; do not run 99-SystemRecovery.ps1 directly.'
}
$script:DbhRecoveryOriginalSdk = ${function:Test-WindowsSdk}
$script:DbhRecoveryOriginalSystem = ${function:Invoke-SystemStage}

function Get-DbhRecoverySdkRoots {
    foreach ($keyPath in @('HKLM:\SOFTWARE\Microsoft\Windows Kits\Installed Roots',
            'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows Kits\Installed Roots')) {
        $key = Get-ItemProperty -LiteralPath $keyPath -ErrorAction SilentlyContinue
        if ($key -and $key.PSObject.Properties['KitsRoot10'] -and $key.KitsRoot10) { $key.KitsRoot10 }
    }
    foreach ($base in @(${env:ProgramFiles(x86)}, $env:ProgramFiles)) {
        if ($base) { Join-Path $base 'Windows Kits/10' }
    }
}

function Get-DbhRecoverySdk {
    foreach ($root in (Get-DbhRecoverySdkRoots | Select-Object -Unique)) {
        $include = Join-Path $root 'Include'
        if (!(Test-Path -LiteralPath $include -PathType Container)) { continue }
        $versions = Get-ChildItem -LiteralPath $include -Directory |
            Where-Object Name -Match '^10\.0\.\d+\.0$' | Sort-Object { [version]$_.Name } -Descending
        foreach ($directory in $versions) {
            $version = $directory.Name
            # A fallback must contain usable x64 SDK files, not just an empty directory.
            $required = @("Include/$version/um/Windows.h", "Include/$version/ucrt/stdio.h",
                "Lib/$version/um/x64/kernel32.lib", "Lib/$version/ucrt/x64/ucrt.lib",
                "bin/$version/x64/rc.exe")
            $missing = @($required | Where-Object { !(Test-Path -LiteralPath (Join-Path $root $_) -PathType Leaf) })
            if (!$missing.Count) { return [pscustomobject]@{ Root=$root; Version=$version } }
        }
    }
    return $null
}

function Test-WindowsSdk {
    # Preserve the original result on computers where setup already works.
    if (& $script:DbhRecoveryOriginalSdk) { return $true }
    $sdk = Get-DbhRecoverySdk
    if ($sdk) {
        Write-Host "SDK fallback: found Windows SDK $($sdk.Version) in $($sdk.Root)"
        return $true
    }
    return $false
}

function Get-DbhRecoveryBuildToolsInstances {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    if (!(Test-Path -LiteralPath $vswhere -PathType Leaf)) { return }
    # Only standalone VS 2022 Build Tools are eligible for automatic recovery.
    # Do not repair a user's Community/Professional/Enterprise IDE installation.
    $json = & $vswhere -all -products Microsoft.VisualStudio.Product.BuildTools -version '[17.0,18.0)' -format json
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect Build Tools: vswhere exited with $LASTEXITCODE." }
    $instances = ($json -join [Environment]::NewLine) | ConvertFrom-Json
    return $instances
}

function Save-DbhRecoveryInstallerLogs([hashtable]$Context, [string]$Label) {
    $destination = Join-Path $Context.StageOutput "recovery-logs/$Label"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    $logs = Get-ChildItem -LiteralPath $env:TEMP -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -eq '.log' -and
            ($_.Name.StartsWith('dd_setup_') -or $_.Name.StartsWith('dd_bootstrapper_') -or $_.Name.StartsWith('dd_client_')) } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 30
    foreach ($log in $logs) {
        try { Copy-Item -LiteralPath $log.FullName -Destination $destination -Force -ErrorAction Stop }
        catch { Write-Warning "Could not copy installer log $($log.FullName): $($_.Exception.Message)" }
    }
    Write-Host "Available Microsoft installer logs copied to: $destination"
    Write-Host 'If UAC used another administrator account, additional installer logs may be in that account TEMP folder.'
}

function Test-DbhRecoveryAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-DbhRecoveryInstaller([hashtable]$Context, $Instance,
        [ValidateSet('repair','modify')][string]$Operation) {
    if ($Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { throw 'Recovery installation is not authorized in this run.' }
    if ($Instance.productId -ne 'Microsoft.VisualStudio.Product.BuildTools') { throw 'Automatic recovery is limited to standalone Build Tools.' }
    $path = $Instance.installationPath.TrimEnd('\')
    $channel = $Instance.channelId
    if (![IO.Path]::IsPathRooted($path) -or $path -match '["\r\n]' -or $channel -notmatch '^VisualStudio\.17\.[A-Za-z0-9.]+$') {
        throw 'Unexpected Build Tools installation path or channel. Recovery stopped.'
    }
    $arguments = "$Operation --installPath `"$path`" --channelId $channel --quiet --wait --norestart"
    if ($Operation -eq 'modify') {
        $arguments += ' --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
        if (!(Test-WindowsSdk)) {
            $sdk = if ([version]$Instance.installationVersion -ge [version]'17.12') {
                'Microsoft.VisualStudio.Component.Windows11SDK.26100'
            } else { 'Microsoft.VisualStudio.Component.Windows10SDK.19041' }
            $arguments += " --add $sdk"
        }
    }
    Save-DbhRecoveryInstallerLogs $Context "before-$Operation"
    try {
        $directory = Join-Path $Context.Tools 'system-recovery'
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
        $download = Join-Path $directory 'vs_BuildTools.exe'
        Write-Host 'Downloading the official VS 2022 Build Tools bootstrapper for recovery...'
        Invoke-WebRequest -UseBasicParsing -Uri 'https://aka.ms/vs/17/release/vs_BuildTools.exe' -OutFile $download
        $signature = Get-AuthenticodeSignature -LiteralPath $download
        if ($signature.Status -ne 'Valid' -or !$signature.SignerCertificate -or
                $signature.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
            throw 'The downloaded Build Tools installer does not have a valid Microsoft signature.'
        }
        $options = @{ FilePath=$download; ArgumentList=$arguments; PassThru=$true;
            WindowStyle='Hidden'; WorkingDirectory=$directory; ErrorAction='Stop' }
        if (!(Test-DbhRecoveryAdministrator)) {
            Write-Host 'Approve the Windows administrator prompt to recover Build Tools.'
            $options.Verb = 'RunAs'
        }
        Write-Host "Recovery operation: $arguments"
        $process = Start-Process @options
        try {
            # --wait belongs to the bootstrapper, and keeps it alive until setup finishes.
            while (!$process.WaitForExit(15000)) { Write-Host "Build Tools $Operation is still running..." }
            $process.WaitForExit()
            $exitCode = $process.ExitCode
        } finally { $process.Dispose() }
        Write-Host "Build Tools $Operation exit code: $exitCode"
        if ($exitCode -in @(3010,1641)) { throw 'Microsoft requested a restart. Restart Windows, then repeat the same System command.' }
        if ($null -eq $exitCode -or $exitCode -ne 0) {
            $hint = switch ($exitCode) {
                740 { 'Administrator access is required.' }
                1001 { 'Another Visual Studio installer is running; let it finish before retrying.' }
                1618 { 'Another installation is running; let it finish before retrying.' }
                1602 { 'Installation was canceled.' }
                5003 { 'A package download failed; check the network/proxy connection.' }
                default { 'The underlying installer failure needs attention.' }
            }
            throw "Build Tools $Operation failed (exit $exitCode). $hint See recovery-logs in this stage output."
        }
    } finally {
        try { Save-DbhRecoveryInstallerLogs $Context "after-$Operation" }
        catch { Write-Warning "Could not collect installer logs: $($_.Exception.Message)" }
    }
}

function Invoke-DbhBuildToolsRecovery([hashtable]$Context) {
    if (!$Context.WithCpp -or $Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { return }
    if (Get-BuildTools) { return } # A usable compiler already exists: never repair another instance.
    $instance = Get-DbhRecoveryBuildToolsInstances | Sort-Object { [version]$_.installationVersion } -Descending | Select-Object -First 1
    if (!$instance) { return } # No existing Build Tools: keep the original fresh-install behavior.
    if ($instance.isRebootRequired) { throw 'Build Tools requires a Windows restart. Restart, then repeat the same System command.' }
    if (!$instance.isComplete) {
        Write-Host "Conditional recovery: incomplete Build Tools installation at $($instance.installationPath)."
        Invoke-DbhRecoveryInstaller $Context $instance 'repair'
        $path = $instance.installationPath
        $instance = Get-DbhRecoveryBuildToolsInstances | Where-Object installationPath -EQ $path | Select-Object -First 1
        if ($instance -and $instance.isRebootRequired) { throw 'Build Tools repair requires a restart. Restart, then repeat the same System command.' }
        if (!$instance -or !$instance.isComplete) {
            throw 'Build Tools still reports an incomplete installation after repair. Recovery stopped; see recovery-logs in this stage output.'
        }
    }
    if (!(Get-BuildTools) -or !(Test-WindowsSdk)) {
        Write-Host 'Conditional recovery: adding the required C++/SDK components to the existing Build Tools installation.'
        Invoke-DbhRecoveryInstaller $Context $instance 'modify'
    }
    if (!(Get-BuildTools)) { throw 'A complete VS 2022 C++ toolset is still not detectable after recovery. See recovery-logs in this stage output.' }
    if (!(Test-WindowsSdk)) { throw 'Windows SDK files are still missing after recovery. See recovery-logs in this stage output.' }
}

function Invoke-SystemStage([hashtable]$Context) {
    Invoke-DbhBuildToolsRecovery $Context
    # Run the original stage, including its runtime, compiler, SDK and GPU checks.
    & $script:DbhRecoveryOriginalSystem $Context
}
