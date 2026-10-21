#!/usr/bin/env bash
# Run a script with the Isaac Sim Python.  Usage:  scripts/isaac.sh sim/run_sorter.py --perception gt
# Set ISAACSIM_PYTHON if the Isaac Sim venv is not at ../env_isaacsim (next to this repo).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default="$here/../../env_isaacsim/Scripts/python.exe"
[ -x "$default" ] || default="$here/../../env_isaacsim/bin/python"
py="${ISAACSIM_PYTHON:-$default}"
if [ ! -e "$py" ]; then
  echo "Isaac Sim Python not found: $py  (set ISAACSIM_PYTHON)" >&2
  exit 1
fi
OMNI_KIT_ACCEPT_EULA=YES PYTHONUTF8=1 exec "$py" "$@"
