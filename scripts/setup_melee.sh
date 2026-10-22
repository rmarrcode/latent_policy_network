#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
"$PYTHON_BIN" -m pip install --editable '.[melee]'
mkdir -p .cache/slippi
target=.cache/slippi/Slippi_Online-x86_64-ExiAI.AppImage
if [[ ! -f "$target" ]]; then
    curl --fail --location --retry 3 \
        https://github.com/vladfi1/slippi-Ishiiruka/releases/download/exi-ai-0.2.0/Slippi_Online-x86_64-ExiAI.AppImage \
        --output "$target.download"
    mv "$target.download" "$target"
fi
echo '87e9ef6d80ed03354a1647d0616016dbc91399aa9e86a69ae5a398edd0a0c2bd  .cache/slippi/Slippi_Online-x86_64-ExiAI.AppImage' | sha256sum --check
chmod +x "$target"
(
    cd .cache/slippi
    ./Slippi_Online-x86_64-ExiAI.AppImage --appimage-extract > extract.log
)
"$PYTHON_BIN" -c 'from melee.console import get_dolphin_version; print(get_dolphin_version(".cache/slippi/squashfs-root/AppRun"))'
