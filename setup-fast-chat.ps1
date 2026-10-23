[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$appDirectory = [System.IO.Path]::GetFullPath($PSScriptRoot)
$appPython = Join-Path $appDirectory '.venv\Scripts\python.exe'
$requirementsFile = Join-Path $appDirectory 'requirements-fast-chat.txt'
if (-not (Test-Path -LiteralPath $appPython -PathType Leaf)) {
    throw 'Run .\setup-cuda.ps1 first to create the app CUDA environment.'
}

& $appPython -c 'import sys,torch; assert sys.platform == "win32", "This installer is for Windows."; assert (3,10) <= sys.version_info[:2] < (3,15), "Use Python 3.10 to 3.14."; assert torch.__version__.split("+")[0].startswith("2.14."), "These Triton wheels target PyTorch 2.14; run setup-cuda.ps1 first."; assert torch.cuda.is_available(), "CUDA is unavailable in the app environment."'
if ($LASTEXITCODE -ne 0) { throw 'The existing runtime does not match the tested fast-chat environment.' }

$uvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
if ($null -ne $uvCommand) {
    & $uvCommand.Source pip install --python $appPython -r $requirementsFile
} else {
    & $appPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { throw 'Could not prepare pip in the app environment.' }
    & $appPython -m pip install -r $requirementsFile
}
if ($LASTEXITCODE -ne 0) { throw 'Optional fast-chat dependency installation failed.' }

$kernelCheck = @'
import torch
import triton
from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
q = torch.zeros((1, 1, 16, 128), device='cuda', dtype=dtype)
g = torch.full((1, 1, 16), -.1, device='cuda', dtype=torch.float32)
beta = torch.ones((1, 1, 16), device='cuda', dtype=dtype)
with torch.inference_mode():
    output, state = fused_recurrent_gated_delta_rule(q, q, q, g=g, beta=beta,
        output_final_state=True, use_qk_l2norm_in_kernel=True)
torch.cuda.synchronize()
assert torch.isfinite(output).all() and torch.isfinite(state).all(), 'Kernel returned invalid values'
print('Fast chat kernel verified:', torch.cuda.get_device_name(0), '| Triton', triton.__version__)
'@
& $appPython -c $kernelCheck
if ($LASTEXITCODE -ne 0) { throw 'Optional packages installed, but the real GPU kernel check failed.' }
Write-Host 'Fast chat setup complete. Unload and reload your chat model to use the new kernels.'
