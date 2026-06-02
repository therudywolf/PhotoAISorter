# Install PyTorch with CUDA 12.6 into project .venv (RTX 40xx etc.)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Сначала создайте .venv (run.bat)."
}
& $Py -m pip install --upgrade pip
# --force-reinstall --no-deps: a CPU torch of the same version already satisfies
# torch>=..., so a plain install is a no-op and never fetches the CUDA wheel.
# --no-deps keeps numpy/pillow (absent from the pytorch index) intact.
& $Py -m pip install -r (Join-Path $Root "requirements-gpu.txt") --force-reinstall --no-deps
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
