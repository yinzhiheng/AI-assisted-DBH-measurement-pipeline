function Invoke-SystemStage([hashtable]$Context) {
    if (!(Test-VcRuntime)) {
        if ($Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { throw 'VC++ v14 x64 runtime >=14.44.35211 is missing. Run System with -InstallSystemPrerequisites.' }
        Install-MicrosoftPackage $Context 'https://aka.ms/vs/17/release/vc_redist.x64.exe' 'vc_redist.x64.exe' '/install /quiet /norestart'
        if (!(Test-VcRuntime)) { throw 'VC++ x64 runtime is still missing.' }
    }
    Write-Host 'PASS: Microsoft VC++ x64 runtime.'
    if ($Context.WithCpp) {
        $msbuild = Get-BuildTools
        $sdk = Get-WindowsSdk
        if (!$msbuild -or !$sdk) {
            if (!$msbuild) { Write-Host 'Missing: VS 2022 MSBuild and MSVC v143 x64 compiler/libraries.' }
            if (!$sdk) { Write-Host 'Missing: complete Windows 10/11 SDK headers, x64 libraries and resource compiler.' }
            if ($Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { throw 'Run System with -InstallSystemPrerequisites to install the missing C++ build components automatically.' }
            Install-Vs2022BuildTools $Context
            $msbuild = Get-BuildTools
            $sdk = Get-WindowsSdk
            $missing = @()
            if (!$msbuild) { $missing += 'VS 2022 MSBuild/MSVC v143 x64' }
            if (!$sdk) { $missing += 'complete Windows SDK (headers, x64 libraries and rc.exe)' }
            if ($missing.Count) { throw "Microsoft installer returned success, but verification still cannot find: $($missing -join '; '). Check $env:TEMP\dd_*.log and this stage's stage.log. If Windows has a pending restart, restart and repeat System." }
        }
        Write-Host "PASS: VS 2022 C++ tools. MSBuild: $msbuild"
        Write-Host "PASS: Windows SDK $($sdk.Version). Location: $($sdk.Root)"
    }
    if ($Context.Device -eq 'cu126') {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        $smiPath = if ($smi) { $smi.Source } else { Join-Path $env:SystemRoot 'System32/nvidia-smi.exe' }
        if (!(Test-Path -LiteralPath $smiPath)) { throw 'NVIDIA driver/nvidia-smi missing. Install an NVIDIA driver or choose -Device cpu.' }
        Invoke-Checked $smiPath @('--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader')
        Write-Host 'GPU/driver detected. The Torch stage must still pass actual CUDA execution.'
    }
}
