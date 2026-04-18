#!/usr/bin/env bash
# zenoh/setup.sh
#
# Downloads and installs the Zenoh ROS 2 DDS bridge binary.
# Run this once on BOTH the assigned GPU server and your local laptop before using
# start_zenoh_bridge.sh or connect_zenoh_bridge.sh.
#
# Usage:
#   chmod +x zenoh/setup.sh && ./zenoh/setup.sh
#
# To force a re-download of an existing binary, pass --force:
#   ./zenoh/setup.sh --force

set -Eeuo pipefail

ZENOH_VERSION="1.9.0"
ARCH="x86_64-unknown-linux-gnu"
ZIP_NAME="zenoh-plugin-ros2dds-${ZENOH_VERSION}-${ARCH}-standalone.zip"
DOWNLOAD_URL="https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download/${ZENOH_VERSION}/${ZIP_NAME}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_BIN="${SCRIPT_DIR}/zenoh-bridge-ros2dds"
PLUGIN_LIB="${SCRIPT_DIR}/libzenoh_plugin_ros2dds.so"
FORCE=false

info()  { printf '[INFO] %s\n' "$*"; }
error() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

for arg in "$@"; do
  case "$arg" in
    --force) FORCE=true ;;
    -h|--help)
      echo "Usage: $0 [--force]"
      echo "  Downloads zenoh-bridge-ros2dds v${ZENOH_VERSION} into this directory."
      echo "  --force   Re-download even if the binary already exists."
      exit 0
      ;;
    *) error "Unknown argument: ${arg}" ;;
  esac
done

if [[ -x "$BRIDGE_BIN" && "$FORCE" == false ]]; then
  info "zenoh-bridge-ros2dds already present at ${BRIDGE_BIN}."
  info "Run with --force to re-download."
  exit 0
fi

# Prefer curl, fall back to wget.
if command -v curl >/dev/null 2>&1; then
  info "Downloading Zenoh bridge v${ZENOH_VERSION} via curl..."
  curl -fsSL "$DOWNLOAD_URL" -o "${SCRIPT_DIR}/${ZIP_NAME}"
elif command -v wget >/dev/null 2>&1; then
  info "Downloading Zenoh bridge v${ZENOH_VERSION} via wget..."
  wget -q "$DOWNLOAD_URL" -O "${SCRIPT_DIR}/${ZIP_NAME}"
else
  error "Neither curl nor wget is available. Install one and retry."
fi

command -v unzip >/dev/null 2>&1 || error "unzip is not installed. Run: sudo apt-get install -y unzip"

info "Extracting binaries..."
# Extract only the two files we need, flat into the script directory.
unzip -j -o "${SCRIPT_DIR}/${ZIP_NAME}" \
  "zenoh-bridge-ros2dds" \
  "libzenoh_plugin_ros2dds.so" \
  -d "${SCRIPT_DIR}/"

chmod +x "$BRIDGE_BIN" "$PLUGIN_LIB"
rm -f "${SCRIPT_DIR}/${ZIP_NAME}"

info "Done. Installed:"
info "  ${BRIDGE_BIN}"
info "  ${PLUGIN_LIB}"
info ""
info "Next steps:"
info "  GPU server :  source /opt/ros/<distro>/setup.bash && ./zenoh/start_zenoh_bridge.sh"
info "  Laptop     :  ./zenoh/connect_zenoh_bridge.sh <GPU_IP> [EXTERNAL_PORT]"
