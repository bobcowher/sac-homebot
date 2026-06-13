#!/usr/bin/env bash
set -euo pipefail

# Downloads the latest checkpoints for the Q-Homebot project from Beekeeper.
# For the world-model run this pulls world_model.pt, actor.pt, critic.pt (+best.pt).
BEEKEEPER_HOST="http://lab.local:5000"
PROJECT="Q-Homebot"
DEST="$(dirname "$0")/checkpoints"

mkdir -p "$DEST"

echo "Downloading checkpoints from $BEEKEEPER_HOST..."
curl -fsSL "${BEEKEEPER_HOST}/api/v1/projects/${PROJECT}/files/checkpoints?zip=1" \
    -o /tmp/beekeeper_checkpoints.zip

echo "Extracting to $DEST..."
unzip -o /tmp/beekeeper_checkpoints.zip -d "$DEST"
rm /tmp/beekeeper_checkpoints.zip

echo "Done. Models are in $DEST"
ls -lh "$DEST"
