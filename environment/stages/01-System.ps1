function Invoke-SystemStage([hashtable]$Context) {
    if (!(Test-VcRuntime)) {
        if ($Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { throw 'VC++ v14 x64 runtime >=14.44.35211 is missing. Run System with -InstallSystemPrerequisites.' }
        Install-MicrosoftPackage $Context 'https://aka.ms/vs/17/release/vc_redist.x64.exe' 'vc_redist.x64.exe' '/install /quiet /norestart'
        if (!(Test-VcRuntime)) { throw 'VC++ x64 runtime is still missing.' }
    }
    Write-Host 'PASS: Microsoft VC++ x64 runtime.'
    if ($Context.WithCpp) {
        if (!(Get-BuildTools) -or !(Test-WindowsSdk)) {
            if ($Context.CheckOnly -or !$Context.InstallSystemPrerequisites) { throw 'VS 2022 C++ tools or Windows SDK missing. Run System with -InstallSystemPrerequisites.' }
            Install-MicrosoftPackage $Context 'https://aka.ms/vs/17/release/vs_BuildTools.exe' 'vs_BuildTools.exe' '--quiet --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'
            if (!(Get-BuildTools) -or !(Test-WindowsSdk)) { throw 'VS 2022 tools/Windows SDK are still missing.' }
        }
        Write-Host "PASS: VS 2022 C++ tools and Windows SDK. MSBuild: $(Get-BuildTools)"
    }
    if ($Context.Device -eq 'cu126') {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        $smiPath = if ($smi) { $smi.Source } else { Join-Path $env:SystemRoot 'System32/nvidia-smi.exe' }
        if (!(Test-Path -LiteralPath $smiPath)) { throw 'NVIDIA driver/nvidia-smi missing. Install an NVIDIA driver or choose -Device cpu.' }
        Invoke-Checked $smiPath @('--query-gpu=name,driver_version,memory.total,memory.free','--format=csv,noheader')
        Write-Host 'GPU/driver detected. The Torch stage must still pass actual CUDA execution.'
    }
}
