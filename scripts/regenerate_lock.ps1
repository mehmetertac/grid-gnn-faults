# Regenerate requirements.lock (universal pins + merged wheel hashes for CI).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

uv pip compile pyproject.toml --universal --python-version 3.11 --extra dev --extra grid --only-binary :all: --generate-hashes --extra-index-url https://download.pytorch.org/whl/cpu -o requirements.lock

uv pip compile pyproject.toml --python-version 3.11 --python-platform x86_64-pc-windows-msvc --extra dev --extra grid --only-binary :all: --generate-hashes --extra-index-url https://download.pytorch.org/whl/cpu -o requirements.win.lock
uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-pc-windows-msvc --extra dev --extra grid --only-binary :all: --generate-hashes --extra-index-url https://download.pytorch.org/whl/cpu -o requirements.win312.lock
uv pip compile pyproject.toml --python-version 3.11 --python-platform x86_64-unknown-linux-gnu --extra dev --extra grid --only-binary :all: --generate-hashes --extra-index-url https://download.pytorch.org/whl/cpu -o requirements.linux311.lock
uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --extra dev --extra grid --only-binary :all: --generate-hashes --extra-index-url https://download.pytorch.org/whl/cpu -o requirements.linux312.lock

python scripts/merge_lock_hashes.py