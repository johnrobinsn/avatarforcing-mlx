#!/usr/bin/env bash
# Fetch the original AvatarForcing code (reference for the equivalence tests) at the
# commit this port was made against.
set -euo pipefail
cd "$(dirname "$0")/.."
SHA=507e3046ad539c938e0412ac7a74cf6e3ba2a61c
if [ ! -d third_party/AvatarForcing/.git ]; then
  git clone https://github.com/TaekyungKi/AvatarForcing.git third_party/AvatarForcing
fi
git -C third_party/AvatarForcing fetch -q origin
git -C third_party/AvatarForcing checkout -q "$SHA"
echo "third_party/AvatarForcing @ $(git -C third_party/AvatarForcing rev-parse --short HEAD)"
