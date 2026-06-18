#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

chmod +x scripts/git-hooks/post-checkout
git config core.hooksPath scripts/git-hooks

echo "Git hooks enabled: core.hooksPath=scripts/git-hooks"
echo "Branch switches will now auto-clean __pycache__ and empty robosuite/ dirs."
