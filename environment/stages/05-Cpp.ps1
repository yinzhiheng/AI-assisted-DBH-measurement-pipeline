function Invoke-CppStage([hashtable]$Context) {
    Assert-Files $Context @('environment/cpp/opencv/bin/opencv_world4110.dll',
        'environment/cpp/onnxruntime/lib/onnxruntime.dll','environment/cpp/onnxruntime/lib/onnxruntime_providers_shared.dll',
        'environment/cpp/opencv/include/opencv2/opencv.hpp','environment/cpp/opencv/lib/opencv_world4110.lib',
        'environment/cpp/onnxruntime/include/onnxruntime_cxx_api.h','environment/cpp/onnxruntime/lib/onnxruntime.lib',
        'model/MobileSAMv2/onnx/mobile_sam.onnx','model/MobileSAMv2/onnx/mobile_sam_preprocess.onnx')
    if (!(Test-VcRuntime)) { throw 'Run the System stage before Cpp.' }
    if (!$Context.CheckOnly -and !$Context.SkipBuild) {
        if (!(Get-BuildTools) -or !(Test-WindowsSdk)) { throw 'Run the System stage before Cpp.' }
        & (Join-Path $Context.Root 'codes/DBH/run/Build-DBH.ps1') -MSBuildPath (Get-BuildTools)
    }
    $release = Join-Path $Context.Root 'codes/DBH/build/Release'
    foreach ($dll in @('environment/cpp/opencv/bin/opencv_world4110.dll','environment/cpp/onnxruntime/lib/onnxruntime.dll','environment/cpp/onnxruntime/lib/onnxruntime_providers_shared.dll')) {
        $source = Join-Path $Context.Root $dll
        $destination = Join-Path $release ([IO.Path]::GetFileName($dll))
        if (!$Context.CheckOnly) { Copy-Item -LiteralPath $source -Destination $destination -Force }
        if (!(Test-Path -LiteralPath $destination) -or (Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $destination).Hash) {
            throw "Missing or mismatched runtime DLL: $destination. Rerun Cpp without -CheckOnly."
        }
    }
    foreach ($dataset in @('Natural','Plot')) {
        foreach ($method in @('SAM3','MobileSAMv2')) {
            Assert-Files $Context @("codes/DBH/build/Release/batchProcess${dataset}DatasetBy${method}.exe")
        }
    }
    $results = Join-Path $Context.StageOutput 'DBHResults'
    & (Join-Path $Context.Root 'codes/DBH/run/Run-DBH.ps1') -Method MobileSAMv2 -Dataset all -MaxFiles 1 -DataDir $Context.DataDir -OutputDir $results
    Assert-Measurements $results @('MobileSAMv2') 1
}
