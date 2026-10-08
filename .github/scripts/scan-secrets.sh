#!/usr/bin/env bash
set -euo pipefail
# Official release, pinned archive digest; no third-party action or license key.
version=8.30.1
archive="gitleaks_${version}_linux_x64.tar.gz"
tool_dir="$(mktemp -d)"
trap 'rm -rf "$tool_dir"' EXIT
curl --fail --silent --show-error --location --retry 3 \
  "https://github.com/gitleaks/gitleaks/releases/download/v${version}/${archive}" \
  --output "$tool_dir/$archive"
printf '%s  %s\n' '551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb' "$tool_dir/$archive" | sha256sum --check
tar -xzf "$tool_dir/$archive" -C "$tool_dir" gitleaks
"$tool_dir/gitleaks" git --redact --log-opts='--all' .
