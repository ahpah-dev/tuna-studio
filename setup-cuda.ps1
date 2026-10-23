[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$appDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$appPython = Join-Path $appDirectory '.venv\Scripts\python.exe'
$requirementsFile = Join-Path $appDirectory 'requirements-train.txt'

if (-not (Test-Path -LiteralPath $appPython -PathType Leaf)) {
    & python -m venv (Join-Path $appDirectory '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the app Python environment.' }
}

# CUDA 13.0 wheels include support for NVIDIA Blackwell (RTX 50 series).
# The NVIDIA driver must support CUDA 13.0; a separate CUDA Toolkit is not needed.
$uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
if ($null -ne $uvCommand) {
    & $uvCommand.Source pip install --python $appPython --torch-backend cu130 'torch==2.14.1+cu130' -r $requirementsFile
    if ($LASTEXITCODE -ne 0) { throw 'CUDA dependency installation failed.' }
} else {
    & $appPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip in the app environment.' }
    & $appPython -m pip install 'torch==2.14.1+cu130' --index-url 'https://download.pytorch.org/whl/cu130'
    if ($LASTEXITCODE -ne 0) { throw 'CUDA-enabled PyTorch installation failed.' }
    & $appPython -m pip install -r $requirementsFile
    if ($LASTEXITCODE -ne 0) { throw 'Training dependency installation failed.' }
}

& $appPython -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable; check the NVIDIA driver."; x=torch.randn(64,64,device="cuda",requires_grad=True); (x@x).sum().backward(); torch.cuda.synchronize(); print("CUDA verified:",torch.cuda.get_device_name(0),"| torch",torch.__version__,"| CUDA",torch.version.cuda)'
if ($LASTEXITCODE -ne 0) { throw 'Installed dependencies, but the real CUDA operation failed.' }
Write-Host 'CUDA setup complete. Restart Tuna Studio with .\start-tuna.ps1.'
