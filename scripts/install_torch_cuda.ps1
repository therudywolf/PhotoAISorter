# Install PyTorch with CUDA 12.4 into project .venv (RTX 40xx etc.)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Error "Сначала создайте .venv (run.bat)."
}
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Root "requirements-gpu.txt")
& $Py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
