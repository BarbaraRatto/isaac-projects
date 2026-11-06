#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_NAME=$(basename "$0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Minimal cloud startup/system services may not export HOME.
HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"
[[ -n "$HOME" ]] || { printf 'Cannot determine the current user home directory.\n' >&2; exit 1; }
export HOME

# Defaults; can be overridden with environment variables.
ISAAC_IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:5.1.0}"
ROS_INSTALL_VARIANT="${ROS_INSTALL_VARIANT:-ros-base}"   # ros-base | desktop
ISAAC_ROS_ENABLE="${ISAAC_ROS_ENABLE:-1}"
ISAAC_ROS_DISTRO="${ISAAC_ROS_DISTRO:-auto}"
ISAAC_ROS_INSTALL_VARIANT="${ISAAC_ROS_INSTALL_VARIANT:-$ROS_INSTALL_VARIANT}"
ISAAC_ROS_IMAGE="${ISAAC_ROS_IMAGE:-}"
CONTAINER_NAME="${CONTAINER_NAME:-isaac-sim}"
ISAAC_HOST_ROOT="${ISAAC_HOST_ROOT:-$HOME/docker/isaac-sim}"
WEBRTC_SIGNAL_PORT="${WEBRTC_SIGNAL_PORT:-49100}"
WEBRTC_STREAM_PORT="${WEBRTC_STREAM_PORT:-47998}"
ALLOWED_CLIENT_IP="${ALLOWED_CLIENT_IP:-}"               # optional; only used if ufw is active
ISAAC_EXTRA_ARGS="${ISAAC_EXTRA_ARGS:-}"
PRIVACY_USERID="${PRIVACY_USERID:-}"
START_TIMEOUT_SEC="${START_TIMEOUT_SEC:-30}"
if [[ -z "${ROS_DOMAIN_ID:-}" && -r /etc/isaac-projects/ros.env ]]; then
  # shellcheck disable=SC1091
  source /etc/isaac-projects/ros.env
fi
ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-}"
HOST_WORKSPACE_ROOT="${HOST_WORKSPACE_ROOT:-$SCRIPT_DIR}"
CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-/workspace/isaac-projects}"
CONTAINER_UID="${CONTAINER_UID:-}"
CONTAINER_GID="${CONTAINER_GID:-}"
ZENOH_BRIDGE_CONTEXT="${ZENOH_BRIDGE_CONTEXT:-auto}"
ISAACLAB_ENABLE="${ISAACLAB_ENABLE:-1}"
ISAACLAB_PATH="${ISAACLAB_PATH:-external/IsaacLab}"
ISAACLAB_REPO_URL="${ISAACLAB_REPO_URL:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_REF="${ISAACLAB_REF:-v2.3.2}"
ISAACLAB_FRAMEWORKS="${ISAACLAB_FRAMEWORKS:-rsl_rl}"
ISAACLAB_IMAGE="${ISAACLAB_IMAGE:-}"
VSCODE_REMOTE_ENABLE="${VSCODE_REMOTE_ENABLE:-1}"
JUPYTER_ENABLE="${JUPYTER_ENABLE:-1}"
STUDENT_EXTRA_TOOLS="${STUDENT_EXTRA_TOOLS:-}"
# Vast full-VM preparation and the desktop are the normal student deployment.
# Non-Vast/prepared hosts can set VAST_VM_PREPARE=0 or use --no-vast-vm.
VAST_VM_PREPARE="${VAST_VM_PREPARE:-1}"
TIGERVNC_ENABLE="${TIGERVNC_ENABLE:-$VAST_VM_PREPARE}"
TIGERVNC_DISPLAY="${TIGERVNC_DISPLAY:-1}"
if [[ -z "${TIGERVNC_PORT:-}" && -r "$HOME/.vnc/isaac-projects-port" ]]; then
  read -r TIGERVNC_PORT < "$HOME/.vnc/isaac-projects-port"
fi
TIGERVNC_PORT="${TIGERVNC_PORT:-5901}"
TIGERVNC_GEOMETRY="${TIGERVNC_GEOMETRY:-1920x1080}"
TIGERVNC_DEPTH="${TIGERVNC_DEPTH:-24}"
TIGERVNC_LOCALHOST="${TIGERVNC_LOCALHOST:-0}"
TIGERVNC_DESKTOP="${TIGERVNC_DESKTOP:-gnome}"
TIGERVNC_TERMINAL="${TIGERVNC_TERMINAL:-gnome-terminal}"
TIGERVNC_PASSWORD="${TIGERVNC_PASSWORD:-}"
TIGERVNC_AUTOSTART="${TIGERVNC_AUTOSTART:-$VAST_VM_PREPARE}"
APT_LOCK_TIMEOUT="${APT_LOCK_TIMEOUT:-900}"
NVIDIA_MIN_DRIVER_VERSION="${NVIDIA_MIN_DRIVER_VERSION:-580.65.06}"
INSTALL_STAGE_LABEL=""
INSTALL_STDERR_FD=""
VERBOSE=0
LOG_FILE="${LOG_FILE:-}"
ACTIVE_LOG_FILE=""
INVOCATION_STRING="$SCRIPT_NAME"
DOCKER_GROUP_RELOGIN_REQUIRED=0
TIGERVNC_XSTARTUP_UPDATED=0
ISAAC_RUNTIME_IMAGE=""
RESOLVED_ISAAC_ROS_DISTRO=""

COLOR_RESET=""
COLOR_BOLD=""
COLOR_BLUE=""
COLOR_GREEN=""
COLOR_YELLOW=""
COLOR_RED=""
COLOR_CYAN=""
COLOR_DIM=""
UI_IS_TTY=0
UI_SUPPORTS_UNICODE=0

setup_colors() {
  local charset_hint
  if [[ -t 1 ]]; then
    UI_IS_TTY=1
  fi

  charset_hint="${LC_ALL:-${LC_CTYPE:-${LANG:-}}}"
  if command -v locale >/dev/null 2>&1; then
    charset_hint="$(locale charmap 2>/dev/null || printf '%s' "$charset_hint")"
  fi
  case "$charset_hint" in
    *UTF-8*|*utf8*|*UTF8*)
      UI_SUPPORTS_UNICODE=1
      ;;
  esac

  if [[ "$UI_IS_TTY" -eq 1 && -z "${NO_COLOR:-}" ]]; then
    COLOR_RESET=$'\033[0m'
    COLOR_BOLD=$'\033[1m'
    COLOR_BLUE=$'\033[34m'
    COLOR_GREEN=$'\033[32m'
    COLOR_YELLOW=$'\033[33m'
    COLOR_RED=$'\033[31m'
    COLOR_CYAN=$'\033[36m'
    COLOR_DIM=$'\033[2m'
  fi
}

clear_progress_line() {
  if [[ "$UI_IS_TTY" -eq 1 ]]; then
    printf '\r\033[2K'
  fi
}

show_progress_line() {
  if [[ "$UI_IS_TTY" -eq 1 ]]; then
    printf '\r\033[2K%s' "$1"
  fi
}

print_message() {
  local color="$1"
  local label="$2"
  local stream="$3"
  shift 3
  clear_progress_line
  if [[ "$stream" == "stderr" ]]; then
    printf '%b[%s]%b %s\n' "$color" "$label" "$COLOR_RESET" "$*" >&2
  else
    printf '%b[%s]%b %s\n' "$color" "$label" "$COLOR_RESET" "$*"
  fi
}

info()    { print_message "$COLOR_BLUE" "INFO" stdout "$*"; }
success() { print_message "$COLOR_GREEN" "OK" stdout "$*"; }
warn()    { print_message "$COLOR_YELLOW" "WARN" stderr "$*"; }
error()   { print_message "$COLOR_RED" "ERROR" stderr "$*"; exit 1; }

unexpected_error() {
  local exit_code=$?
  local line="$1"
  # Commands may contain credentials; report the location without echoing them.
  local message="${SCRIPT_NAME}: line ${line}: command failed (exit ${exit_code})"
  if [[ -n "$ACTIVE_LOG_FILE" ]]; then
    message="${message}. See ${ACTIVE_LOG_FILE}"
  fi
  print_message "$COLOR_RED" "ERROR" stderr "$message"
  exit "$exit_code"
}

trap 'unexpected_error "${LINENO}" "${BASH_COMMAND}"' ERR

installer_exit() {
  local result="$1"
  # EXIT can run while a failed stage's redirections are still active.
  # Restore the original terminal stream before printing failure details.
  if [[ -n "$INSTALL_STDERR_FD" ]]; then
    exec 1>&"$INSTALL_STDERR_FD" 2>&"$INSTALL_STDERR_FD"
  fi
  stop_install_spinner
  # Exit 42 is the documented successful host-preparation/reboot boundary.
  if [[ "$result" -eq 42 ]]; then return 0; fi
  if [[ "$result" -ne 0 && -n "$INSTALL_STAGE_LABEL" ]]; then
    print_message "$COLOR_RED" "ERROR" stderr "Failed during '${INSTALL_STAGE_LABEL}' (exit ${result}). Full log: ${ACTIVE_LOG_FILE}"
    if [[ "$VERBOSE" -eq 0 && -r "$ACTIVE_LOG_FILE" ]]; then
      tail -n 25 "$ACTIVE_LOG_FILE" >&2 || true
    fi
  fi
}

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

timestamp_compact_utc() {
  date -u +%Y%m%dT%H%M%SZ
}

format_duration() {
  local seconds="$1"
  if (( seconds < 60 )); then
    printf '%ss' "$seconds"
  else
    printf '%sm%02ss' "$((seconds / 60))" "$((seconds % 60))"
  fi
}

repeat_char() {
  local char="$1"
  local count="$2"
  local output=""
  local i
  for ((i = 0; i < count; i++)); do
    output+="$char"
  done
  printf '%s' "$output"
}

spinner_frame() {
  local index="$1"
  if [[ "$UI_SUPPORTS_UNICODE" -eq 1 ]]; then
    local frames=("⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏")
    printf '%s' "${frames[$((index % ${#frames[@]}))]}"
  else
    local frames=("-" "\\" "|" "/")
    printf '%s' "${frames[$((index % ${#frames[@]}))]}"
  fi
}

status_glyph() {
  local phase="$1"
  case "$phase" in
    success)
      if [[ "$UI_SUPPORTS_UNICODE" -eq 1 ]]; then
        printf '✓'
      else
        printf 'OK'
      fi
      ;;
    failure)
      if [[ "$UI_SUPPORTS_UNICODE" -eq 1 ]]; then
        printf '✕'
      else
        printf '!!'
      fi
      ;;
  esac
}

render_progress_bar() {
  local completed="$1"
  local total="$2"
  local phase="${3:-complete}"
  local spinner_index="${4:-0}"
  local width=26
  local filled=0
  local fill_char empty_char active_char
  if (( total > 0 )); then
    filled=$((completed * width / total))
  fi

  if [[ "$UI_SUPPORTS_UNICODE" -eq 1 ]]; then
    fill_char="█"
    empty_char="░"
    if (( spinner_index % 2 == 0 )); then
      active_char="▓"
    else
      active_char="▒"
    fi
  else
    fill_char="="
    empty_char="-"
    active_char=">"
  fi

  local empty=$((width - filled))
  local bar
  bar="$(repeat_char "$fill_char" "$filled")"
  if [[ "$phase" == "active" && filled < width ]]; then
    bar+="$active_char"
    bar+="$(repeat_char "$empty_char" "$((width - filled - 1))")"
  else
    bar+="$(repeat_char "$empty_char" "$empty")"
  fi
  printf '%s[%s]%s' "$COLOR_CYAN" "$bar" "$COLOR_RESET"
}

render_install_status_line() {
  local phase="$1"
  local total="$2"
  local index="$3"
  local label="$4"
  local elapsed_seconds="$5"
  local spinner_index="${6:-0}"
  local mark mark_color bar progress_count

  case "$phase" in
    active)
      mark="$(spinner_frame "$spinner_index")"
      mark_color="$COLOR_BLUE"
      progress_count=$((index - 1))
      bar="$(render_progress_bar "$progress_count" "$total" active "$spinner_index")"
      ;;
    *)
      mark="$(spinner_frame "$spinner_index")"
      mark_color="$COLOR_BLUE"
      progress_count=$((index - 1))
      bar="$(render_progress_bar "$progress_count" "$total" active "$spinner_index")"
      ;;
  esac

  printf '%b%s%b %b[%d/%d]%b %s  %s  %b%s%b' \
    "$mark_color" "$mark" "$COLOR_RESET" \
    "$COLOR_DIM" "$index" "$total" "$COLOR_RESET" \
    "$label" \
    "$bar" \
    "$COLOR_DIM" "$(format_duration "$elapsed_seconds")" "$COLOR_RESET"
}

render_install_step_summary_line() {
  local phase="$1"
  local total="$2"
  local index="$3"
  local label="$4"
  local elapsed_seconds="$5"
  local mark mark_color

  case "$phase" in
    success)
      mark="$(status_glyph success)"
      mark_color="$COLOR_GREEN"
      ;;
    failure)
      mark="$(status_glyph failure)"
      mark_color="$COLOR_RED"
      ;;
    *)
      if [[ "$UI_SUPPORTS_UNICODE" -eq 1 ]]; then
        mark="•"
      else
        mark=">"
      fi
      mark_color="$COLOR_BLUE"
      ;;
  esac

  printf '%b%s%b %b[%d/%d]%b %s  %b%s%b' \
    "$mark_color" "$mark" "$COLOR_RESET" \
    "$COLOR_DIM" "$index" "$total" "$COLOR_RESET" \
    "$label" \
    "$COLOR_DIM" "$(format_duration "$elapsed_seconds")" "$COLOR_RESET"
}

use_live_progress_ui() {
  [[ "$VERBOSE" -eq 0 && "$UI_IS_TTY" -eq 1 ]]
}

print_live_status_line() {
  local stream="${2:-stdout}"
  clear_progress_line
  if [[ "$stream" == "stderr" ]]; then
    printf '%s\n' "$1" >&2
  else
    printf '%s\n' "$1"
  fi
}

start_install_spinner() {
  local total="$1"
  local index="$2"
  local label="$3"

  if ! use_live_progress_ui; then
    return 0
  fi

  (
    trap 'exit 0' TERM INT
    local spinner_index=0
    local started_at=$SECONDS
    while :; do
      show_progress_line "$(render_install_status_line active "$total" "$index" "$label" "$((SECONDS - started_at))" "$spinner_index")"
      sleep 0.12
      spinner_index=$((spinner_index + 1))
    done
  ) &
  INSTALL_SPINNER_PID=$!
}

stop_install_spinner() {
  local spinner_pid="${INSTALL_SPINNER_PID:-}"
  if [[ -n "$spinner_pid" ]]; then
    kill "$spinner_pid" 2>/dev/null || true
    wait "$spinner_pid" 2>/dev/null || true
    INSTALL_SPINNER_PID=""
  fi
  clear_progress_line
}

apply_vast_vm_profile() {
  # Retained for compatibility with earlier explicit --vast-vm invocations.
  VAST_VM_PREPARE=1
  TIGERVNC_ENABLE=1
  TIGERVNC_AUTOSTART=1
  TIGERVNC_DESKTOP=gnome
  TIGERVNC_TERMINAL=gnome-terminal
}

apply_non_vast_vm_profile() {
  VAST_VM_PREPARE=0
  TIGERVNC_AUTOSTART=0
}

parse_common_flags() {
  local -a parsed=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --vast-vm)
        apply_vast_vm_profile
        ;;
      --no-vast-vm)
        apply_non_vast_vm_profile
        ;;
      --verbose|-v)
        VERBOSE=1
        ;;
      --log-file)
        [[ $# -ge 2 ]] || error "--log-file requires a path."
        LOG_FILE="$2"
        shift
        ;;
      *)
        parsed+=("$1")
        ;;
    esac
    shift
  done
  PARSED_ARGS=("${parsed[@]}")
}

parse_leading_common_flags() {
  local -a remaining=("$@")
  while [[ ${#remaining[@]} -gt 0 ]]; do
    case "${remaining[0]}" in
      --vast-vm)
        apply_vast_vm_profile
        remaining=("${remaining[@]:1}")
        ;;
      --no-vast-vm)
        apply_non_vast_vm_profile
        remaining=("${remaining[@]:1}")
        ;;
      --verbose|-v)
        VERBOSE=1
        remaining=("${remaining[@]:1}")
        ;;
      --log-file)
        [[ ${#remaining[@]} -ge 2 ]] || error "--log-file requires a path."
        LOG_FILE="${remaining[1]}"
        remaining=("${remaining[@]:2}")
        ;;
      *)
        break
        ;;
    esac
  done
  PARSED_ARGS=("${remaining[@]}")
}

prepare_log_file() {
  local label="$1"
  if [[ -n "$ACTIVE_LOG_FILE" ]]; then
    return 0
  fi

  if [[ -n "$LOG_FILE" ]]; then
    ACTIVE_LOG_FILE="$LOG_FILE"
  else
    local log_root
    log_root="$(home_dir)/.local/state/isaac-projects/logs"
    ACTIVE_LOG_FILE="${log_root}/${label}_$(timestamp_compact_utc).log"
  fi

  mkdir -p "$(dirname "$ACTIVE_LOG_FILE")"
  : >"$ACTIVE_LOG_FILE"
  {
    printf '[%s] Command: %s\n' "$(timestamp_utc)" "$INVOCATION_STRING"
    printf '[%s] Log mode: %s\n' "$(timestamp_utc)" "$([[ "$VERBOSE" -eq 1 ]] && printf 'verbose' || printf 'summary')"
  } >>"$ACTIVE_LOG_FILE"
}

show_install_log_hint() {
  local label="$1"
  info "${label} log: ${ACTIVE_LOG_FILE}"
  if [[ "$VERBOSE" -eq 0 ]]; then
    info "Use --verbose to stream the full installer output, or run: tail -f ${ACTIVE_LOG_FILE}"
  fi
}

run_logged() {
  local total="$1"
  local index="$2"
  local label="$3"
  shift 3

  {
    printf '\n[%s] %s\n' "$(timestamp_utc)" "$label"
  } >>"$ACTIVE_LOG_FILE"

  if [[ "$VERBOSE" -eq 1 ]]; then
    "$@" > >(tee -a "$ACTIVE_LOG_FILE") 2> >(tee -a "$ACTIVE_LOG_FILE" >&2)
  else
    start_install_spinner "$total" "$index" "$label"
    "$@" >>"$ACTIVE_LOG_FILE" 2>&1
    stop_install_spinner
  fi
}

run_install_step() {
  local total="$1"
  local index="$2"
  local label="$3"
  shift 3

  local started_at=$SECONDS
  if ! use_live_progress_ui; then
    info "[${index}/${total}] ${label}"
  fi

  # Do not put this call (or main) inside an if/!/|| condition: Bash would
  # disable errexit throughout every nested installer function.
  if [[ -z "$INSTALL_STDERR_FD" ]]; then exec {INSTALL_STDERR_FD}>&2; fi
  INSTALL_STAGE_LABEL="$label"
  run_logged "$total" "$index" "$label" "$@"
  INSTALL_STAGE_LABEL=""
  if use_live_progress_ui; then
    print_live_status_line "$(render_install_step_summary_line success "$total" "$index" "$label" "$((SECONDS - started_at))")"
  else
    success "[${index}/${total}] ${label}  $(format_duration "$((SECONDS - started_at))")"
  fi
}

usage() {
  cat <<USAGE
Usage:
  ${SCRIPT_NAME} bootstrap [--no-vast-vm] [--verbose] [--log-file <path>]
                                 Install/repair Docker, NVIDIA runtime, ROS 2, default student tools, and pull the Isaac Sim image.
                                 Also installs and starts TigerVNC when TIGERVNC_ENABLE=1.
  ${SCRIPT_NAME} bootstrap zenoh|bridge [--force]
                                 Download or refresh the Zenoh bridge binary under zenoh/.
  ${SCRIPT_NAME} start isaacsim      Start Isaac Sim with WebRTC.
  ${SCRIPT_NAME} start isaacsim --gui|--vnc
                                 Start native Isaac Sim GUI on the current X display.
  ${SCRIPT_NAME} start tigervnc|vnc  Install/start the TigerVNC desktop.
  ${SCRIPT_NAME} start zenoh|bridge [port] [--domain <id>] [--namespace <ns>] [--config <file>]
                                 Start the server-side Zenoh ROS 2 bridge.
  ${SCRIPT_NAME} start isaacsim --headless
                                 Start Isaac Sim without WebRTC flags.
  ${SCRIPT_NAME} run isaaclab [--path <workspace-subdir>] '<isaaclab.sh args>'
                                 Run Isaac Lab from the bootstrap-managed checkout/image. Omit --headless only from a GUI/VNC terminal.
  ${SCRIPT_NAME} run --py [--gui|--livestream public|private [--public-ip <ip>]] '<python.sh args>'
                                 Run an Isaac Sim python.sh command in a one-shot container. Add --gui from a GUI/VNC terminal to open the app on the current X display, or use --livestream for WebRTC.
  ${SCRIPT_NAME} run [--livestream public|private] [--enable-cameras] [--public-ip <ip>] -- <command>
                                 Run a one-shot command inside the Isaac Sim container image.
  ${SCRIPT_NAME} stop isaacsim       Stop the Isaac Sim container.
  ${SCRIPT_NAME} stop tigervnc|vnc   Stop the TigerVNC desktop.
  ${SCRIPT_NAME} stop zenoh|bridge   Stop server-side Zenoh ROS 2 bridge processes.
  ${SCRIPT_NAME} restart isaacsim    Restart Isaac Sim.
  ${SCRIPT_NAME} status              Show VM / Docker / Isaac Sim / ROS status.
  ${SCRIPT_NAME} doctor              Check bootstrap packages, GPU container access and enabled VNC desktop (no installs/restarts).
  ${SCRIPT_NAME} logs                Show Isaac Sim container logs.
  ${SCRIPT_NAME} shell               Open a shell in the running Isaac Sim container.
  ${SCRIPT_NAME} install all [--verbose] [--log-file <path>]
                                 Install/repair Docker, NVIDIA runtime, ROS 2, default student tools, and Isaac Sim image.
  ${SCRIPT_NAME} install ros2 [--verbose] [--log-file <path>]
                                 Install/repair ROS 2 only.
  ${SCRIPT_NAME} install docker [--verbose] [--log-file <path>]
                                 Install/repair Docker + NVIDIA container runtime only.
  ${SCRIPT_NAME} install zenoh|bridge [--force]
                                 Download or refresh the Zenoh bridge binary under zenoh/.
  ${SCRIPT_NAME} install tigervnc|vnc [--verbose] [--log-file <path>]
                                 Install/start the TigerVNC desktop.
  ${SCRIPT_NAME} check               Show listener checks and client-side test commands.
  ${SCRIPT_NAME} help                Show this help.

Optional environment variables:
  ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0
  ISAAC_ROS_ENABLE=1                  # when 1, bootstrap builds a ROS-enabled Isaac Sim runtime image
  ISAAC_ROS_DISTRO=auto|humble|jazzy   # auto selects from the Isaac Sim image Ubuntu version
  ISAAC_ROS_INSTALL_VARIANT=ros-base|desktop
  ISAAC_ROS_IMAGE=<tag>                # optional local tag for the ROS-enabled Isaac Sim image
  CONTAINER_NAME=isaac-sim
  ISAAC_HOST_ROOT=
  WEBRTC_SIGNAL_PORT=49100
  WEBRTC_STREAM_PORT=47998
  ROS_INSTALL_VARIANT=ros-base|desktop
  ROS_DOMAIN_ID=0
  HOST_WORKSPACE_ROOT=<repo-root>
  CONTAINER_WORKSPACE=/workspace/isaac-projects
  CONTAINER_UID=<host-uid>            # optional; defaults to current user
  CONTAINER_GID=<host-gid>            # optional; defaults to current user
  ZENOH_BRIDGE_CONTEXT=auto|isaac|host # auto uses the Isaac container ROS runtime when it is running
  ISAACLAB_ENABLE=0|1                 # when 1, bootstrap clones/builds a pinned Isaac Lab setup
  ISAACLAB_PATH=external/IsaacLab     # workspace-relative path that contains isaaclab.sh
  ISAACLAB_REPO_URL=https://github.com/isaac-sim/IsaacLab.git
  ISAACLAB_REF=v2.3.2
  ISAACLAB_FRAMEWORKS=rsl_rl|rl_games|sb3|skrl|robomimic|all|none
  ISAACLAB_IMAGE=<tag>                # optional local tag for the managed Isaac Lab image
  VSCODE_REMOTE_ENABLE=0|1            # default 1; installs SSH host prerequisites and VS Code CLI
  JUPYTER_ENABLE=0|1                  # default 1; installs JupyterLab + Notebook in a user-local venv
  STUDENT_EXTRA_TOOLS='tool ...'      # optional: tmux htop btop ncdu nvtop nvitop ripgrep git-lfs ffmpeg mosh
  ALLOWED_CLIENT_IP=<your-public-ip>   # only applied if ufw is already active
  TIGERVNC_ENABLE=0|1                  # when 1, bootstrap installs/starts TigerVNC
  TIGERVNC_AUTOSTART=0|1               # optional per-user/display system service
  TIGERVNC_DISPLAY=1
  TIGERVNC_PORT=5901
  TIGERVNC_GEOMETRY=1920x1080
  TIGERVNC_DEPTH=24
  TIGERVNC_LOCALHOST=0|1               # 0 listens on all interfaces; 1 binds localhost only
  TIGERVNC_DESKTOP=gnome|gnome-flashback|xfce
  TIGERVNC_TERMINAL=gnome-terminal
  TIGERVNC_PASSWORD=<8-char-password>  # optional; generated once when unset
  APT_LOCK_TIMEOUT=900                 # seconds to wait for existing package updates
  NVIDIA_MIN_DRIVER_VERSION=580.65.06
  VAST_VM_PREPARE=0|1                 # default 1; set 0 for a prepared non-Vast host
  PRIVACY_USERID=<email>
  ISAAC_EXTRA_ARGS='<extra Isaac Sim args>'
  NGC_API_KEY=<your-ngc-api-key>       # optional, only used if image pull requires login

Default: Vast Ubuntu 22.04 full VM / RTX 5090:
  ${SCRIPT_NAME} bootstrap
    Installs/repairs the 580-open driver and firmware-capable HWE kernel when needed,
    plus a persistent GNOME/TigerVNC desktop with direct IP/port/password access. Never use on a
    container rental or a shared workstation. Compatible healthy drivers are preserved.
    Exit 42 means an administrator must reboot, reconnect and rerun the same command.
  ${SCRIPT_NAME} doctor
    Checks the installed setup without installing packages or restarting services.
    Isaac Lab is included by default. Set ISAACLAB_ENABLE=0 to skip it.
  Expose TCP 5901 in the Vast template before creating a new VM (-p 5901:5901).
    Bootstrap prints the public IP, mapped VNC port and password-file location.
    Enter that IP and public port in your VNC viewer, then the generated password.
    TIGERVNC_LOCALHOST=1 remains an optional SSH-tunnel mode.
    Vast mode is the default for later commands too. Use --no-vast-vm for other prepared hosts.
    Readiness checks do not qualify an Isaac Sim scene; test GUI/input/clipboard separately.

Optional flags:
  --no-vast-vm                       Skip Vast host repair and persistent VNC service management.
  --vast-vm                          Explicitly select the default Vast settings (compatibility option).
  --verbose, -v                      Stream detailed bootstrap/install logs live.
  --log-file <path>                  Write bootstrap/install logs to a specific file.

Examples:
  ${SCRIPT_NAME} bootstrap
  TIGERVNC_ENABLE=1 ${SCRIPT_NAME} bootstrap
  STUDENT_EXTRA_TOOLS="tmux ncdu nvtop" ${SCRIPT_NAME} bootstrap
  ${SCRIPT_NAME} bootstrap --verbose
  ${SCRIPT_NAME} bootstrap zenoh
  ${SCRIPT_NAME} bootstrap bridge --force
  ${SCRIPT_NAME} start isaacsim
  ${SCRIPT_NAME} start isaacsim --gui
  ${SCRIPT_NAME} start tigervnc
  ${SCRIPT_NAME} start zenoh
  ${SCRIPT_NAME} stop zenoh
  ${SCRIPT_NAME} start bridge 7447 --domain <id>
  ${SCRIPT_NAME} start isaacsim --headless
  ISAACLAB_ENABLE=1 ${SCRIPT_NAME} bootstrap
  ${SCRIPT_NAME} run isaaclab '-p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless'
  ${SCRIPT_NAME} run isaaclab '-p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0'
  ${SCRIPT_NAME} run --py --gui 'standalone_examples/api/isaacsim.robot.policy.examples/anymal_standalone.py'
  ${SCRIPT_NAME} run --py --livestream public 'standalone_examples/api/isaacsim.robot.policy.examples/anymal_standalone.py'
  ${SCRIPT_NAME} run -- bash -lc 'cd projects/my-project && python train.py'
  ROS_INSTALL_VARIANT=desktop ${SCRIPT_NAME} install ros2
  ${SCRIPT_NAME} install docker --log-file /tmp/isaac-bootstrap.log
  ALLOWED_CLIENT_IP=203.0.113.5 ${SCRIPT_NAME} start isaacsim
  ISAAC_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.0-dev2 ${SCRIPT_NAME} restart isaacsim
USAGE
}

need_root() {
  if [[ ${EUID} -eq 0 ]]; then
    ROOT_PREFIX=()
  else
    command -v sudo >/dev/null 2>&1 || error "sudo is required when not running as root."
    ROOT_PREFIX=(sudo)
  fi
}

as_root() {
  need_root
  "${ROOT_PREFIX[@]}" "$@"
}

# Keep host checks embedded; deployment adds no helper files to the repository.
cloud_python_source() {
  cat <<'ISAAC_CLOUD_PYTHON'
#!/usr/bin/env python3
"""Embedded bootstrap checks; requires only the system Python standard library."""
import argparse
import contextlib
import errno
import fcntl
import os
import subprocess
import sys
import time
import http.client
import json
import socket
from pathlib import Path
import hashlib
import re
import shlex
import tempfile


LOCKS = ('/var/lib/dpkg/lock-frontend', '/var/lib/dpkg/lock', '/var/lib/apt/lists/lock', '/var/cache/apt/archives/lock')


def wait_for_locks(paths, deadline):
    # APT uses POSIX record locks, rather than flock. Never remove lock files.
    announced = False
    while True:
        busy = False
        for path in paths:
            try:
                fd = os.open(path, os.O_RDWR)
            except FileNotFoundError:
                continue
            try:
                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                busy = True
            finally:
                os.close(fd)
            if busy:
                break
        if not busy:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('Package manager is still busy. Let OS updates finish and rerun bootstrap; do not delete lock files.')
        if not announced:
            print('Waiting for existing package updates to finish...', flush=True)
            announced = True
        time.sleep(min(1, max(0, deadline - time.monotonic())))


def apt_wait_main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if args.timeout < 0 or not command:
        parser.error('Supply a nonnegative timeout and a command after --')
    wait_for_locks(LOCKS, time.monotonic() + args.timeout)
    # The package command takes its own locks after this bounded preflight.
    return subprocess.call(command)


class UnixHTTPConnection(http.client.HTTPConnection):

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect('/var/run/docker.sock')


def assert_idle(connection):
    connection.request('GET', '/containers/json?all=0')
    response = connection.getresponse()
    if response.status != 200:
        raise ValueError('Cannot establish Docker workload ownership; refusing restart.')
    containers = json.load(response)
    if not isinstance(containers, list):
        raise ValueError('Unexpected Docker response; refusing restart.')
    if containers:
        raise ValueError('Docker has running containers. Stop or coordinate those workloads before rerunning bootstrap; no restart was attempted.')


def gpu_users(proc=Path('/proc')):
    busy = set()
    for process in proc.glob('[0-9]*'):
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target.startswith(('/dev/nvidia', '/dev/dri/')):
                        busy.add(process.name)
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    return busy


def repair_text(original):
    lines = original.splitlines(keepends=True)
    output = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = re.match('^(PUBLIC_KEY|SSH_PUBLIC_KEY)=(.*)$', line.rstrip('\r\n'))
        if not match or match[2].startswith(("'", '"')):
            output.append(line)
            index += 1
            continue
        values = [match[2]]
        index += 1
        while index < len(lines) and re.match('^(ssh-|ecdsa-sha2-)', lines[index]):
            values.append(lines[index].rstrip('\r\n'))
            index += 1
        output.append(match[1] + '=' + shlex.quote('\n'.join(values)) + '\n')
    return ''.join(output)


def repair_file(path, backup_dir):
    if path.is_symlink():
        raise ValueError('Refusing a symlink for the provider environment file')
    original = path.read_text()
    updated = repair_text(original)
    if original == updated:
        return False
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(backup_dir, 0o700)
    digest = hashlib.sha256(original.encode()).hexdigest()[:16]
    backup = backup_dir / ('environment.' + digest)
    if not backup.exists():
        fd = os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            stream.write(original)
    stat = path.stat()
    fd, temporary = tempfile.mkstemp(prefix='.isaac-environment-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(updated)
        # Syntax-check only: provider environment values must never be executed.
        subprocess.run(['/bin/sh', '-n', temporary], check=True, capture_output=True)
        os.chmod(temporary, stat.st_mode & 0o777)
        if os.geteuid() == 0:
            os.chown(temporary, stat.st_uid, stat.st_gid)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def provider_tcp_port(internal_port, path=Path('/etc/environment')):
    # Read only the public port field; never source provider environment data.
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    port = None
    for line in lines:
        if not line.startswith(f'VAST_TCP_PORT_{internal_port}='):
            continue
        try:
            values = shlex.split(line.split('=', 1)[1], comments=True)
        except ValueError:
            continue
        if len(values) == 1 and values[0].isdigit() and 1 <= int(values[0]) <= 65535:
            port = int(values[0])
    return port


def provider_ssh_port(path=Path('/etc/environment')):
    return provider_tcp_port(22, path)


def probe(port):
    with socket.create_connection(('127.0.0.1', port), timeout=2) as connection:
        banner = b''
        while len(banner) < 12:
            data = connection.recv(12 - len(banner))
            if not data:
                break
            banner += data
        return re.fullmatch(b'RFB 003\\.00[378]\\n', banner) is not None

def cloud_host_main():
    if len(sys.argv) < 2:
        raise ValueError("Missing bootstrap helper action")
    action, arguments = sys.argv[1], sys.argv[2:]
    if action == "apt-wait":
        return apt_wait_main(arguments)
    if action == "docker-idle":
        with contextlib.closing(UnixHTTPConnection("localhost", timeout=10)) as connection:
            assert_idle(connection)
        return 0
    if action == "gpu-idle":
        try:
            users = gpu_users()
        except OSError:
            raise ValueError("Cannot establish GPU ownership; refusing driver replacement. Run host preparation as an administrator.") from None
        if users:
            raise ValueError("GPU/graphics devices are in use by PID(s) " + ", ".join(sorted(users)) + ". Coordinate stopping them before driver repair.")
        return 0
    if action == "environment-repair":
        parser = argparse.ArgumentParser(description="Quote provider public-key environment blocks without executing them")
        parser.add_argument("--path", type=Path, default=Path("/etc/environment"))
        parser.add_argument("--backup-dir", type=Path, default=Path("/var/lib/isaac-projects/environment-backups"))
        args = parser.parse_args(arguments)
        try:
            changed = repair_file(args.path, args.backup_dir)
        except (OSError, ValueError, subprocess.CalledProcessError):
            raise ValueError("Provider environment repair failed; original preserved. Inspect the file privately.") from None
        print("Provider environment repaired." if changed else "Provider environment already quoted.")
        return 0
    if action == "vnc-probe":
        try:
            return 0 if len(arguments) == 1 and probe(int(arguments[0])) else 1
        except (OSError, ValueError):
            return 1
    if action == "provider-ssh-port":
        port = provider_ssh_port()
        if port is None:
            return 1
        print(port)
        return 0
    if action == "provider-tcp-port":
        if len(arguments) != 1 or not arguments[0].isdigit() or not 1 <= int(arguments[0]) <= 65535:
            return 1
        port = provider_tcp_port(int(arguments[0]))
        if port is None:
            return 1
        print(port)
        return 0
    raise ValueError("Unknown bootstrap helper action")


if __name__ == "__main__":
    try:
        sys.exit(cloud_host_main())
    except (TimeoutError, OSError, ValueError, http.client.HTTPException) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
ISAAC_CLOUD_PYTHON
}

cloud_python() {
  cloud_python_source | /usr/bin/python3 - "$@"
}

cloud_python_as_root() {
  cloud_python_source | as_root env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l /usr/bin/python3 - "$@"
}

package_command() {
  cloud_python_as_root apt-wait --timeout "$APT_LOCK_TIMEOUT" -- "$@"
}

apt_get() {
  package_command apt-get -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" -o Acquire::Retries=3 "$@"
}

package_installed() {
  [[ "$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null)" == "install ok installed" ]]
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

is_truthy() {
  case "$1" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

as_current_user() {
  local target_user="${USER_NAME:-$(current_user)}"
  local target_home="${USER_HOME:-$(home_dir)}"

  if [[ ${EUID} -eq 0 && -n "${USER_NAME:-}" && "$USER_NAME" != "root" ]]; then
    runuser -u "$target_user" -- env HOME="$target_home" USER="$target_user" LOGNAME="$target_user" "$@"
  else
    env HOME="$target_home" USER="$target_user" LOGNAME="$target_user" "$@"
  fi
}

install_for_current_user() {
  local mode="$1"
  local src="$2"
  local dst="$3"
  local user_group
  user_group=$(id -gn "$USER_NAME")

  if [[ ${EUID} -eq 0 ]]; then
    install -o "$USER_NAME" -g "$user_group" -m "$mode" "$src" "$dst"
  else
    install -m "$mode" "$src" "$dst"
  fi
}

append_unique() {
  local -n target_array=$1
  local value="$2"
  local existing
  for existing in "${target_array[@]:-}"; do
    [[ "$existing" == "$value" ]] && return 0
  done
  target_array+=("$value")
}

ros_distro_for_ubuntu_version() {
  case "$1" in
    22.04) printf '%s\n' "humble" ;;
    24.04) printf '%s\n' "jazzy" ;;
    *) return 1 ;;
  esac
}

validate_ros_domain_id() {
  [[ "$ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] || error "ROS_DOMAIN_ID must be an integer between 0 and 232. Found: ${ROS_DOMAIN_ID}"
  local domain_id=$((10#$ROS_DOMAIN_ID))
  (( domain_id <= 232 )) || error "ROS_DOMAIN_ID must be between 0 and 232. Found: ${ROS_DOMAIN_ID}"
  ROS_DOMAIN_ID="$domain_id"
  export ROS_DOMAIN_ID
}

require_supported_host() {
  [[ -r /etc/os-release ]] || error "/etc/os-release not found. Unsupported host."
  # shellcheck disable=SC1091
  . /etc/os-release

  [[ "${ID:-}" == "ubuntu" ]] || error "This script only supports Ubuntu hosts. Found: ${PRETTY_NAME:-unknown}."

  ROS_DISTRO=$(ros_distro_for_ubuntu_version "${VERSION_ID:-}") || error "Unsupported Ubuntu version: ${VERSION_ID:-unknown}. This script intentionally supports Ubuntu 22.04 (ROS 2 Humble) and Ubuntu 24.04 (ROS 2 Jazzy) only."
  ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
  validate_ros_domain_id

  ARCH=$(dpkg --print-architecture)
  case "$ARCH" in
    amd64) ;;
    *) warn "Architecture is ${ARCH}. Isaac Sim container workflows are typically x86_64/amd64; continue only if that is intentional." ;;
  esac
}

current_user() {
  if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    printf '%s' "$SUDO_USER"
  else
    id -un
  fi
}

home_dir() {
  local user
  user=$(current_user)
  getent passwd "$user" | cut -d: -f6
}

init_paths() {
  USER_NAME=$(current_user)
  USER_HOME=$(home_dir)
  if [[ -z "${ISAAC_HOST_ROOT:-}" || "$ISAAC_HOST_ROOT" == "$HOME/docker/isaac-sim" ]]; then
    ISAAC_HOST_ROOT="${USER_HOME}/docker/isaac-sim"
  fi
  if [[ ! "$HOST_WORKSPACE_ROOT" = /* ]]; then
    HOST_WORKSPACE_ROOT="$(cd "$HOST_WORKSPACE_ROOT" && pwd)"
  fi
  [[ -d "$HOST_WORKSPACE_ROOT" ]] || error "HOST_WORKSPACE_ROOT does not exist: ${HOST_WORKSPACE_ROOT}"

  if [[ -z "$CONTAINER_UID" ]]; then
    CONTAINER_UID=$(id -u "$USER_NAME")
  fi
  if [[ -z "$CONTAINER_GID" ]]; then
    CONTAINER_GID=$(id -g "$USER_NAME")
  fi
}

get_docker_cmd() {
  if [[ "$EUID" -eq 0 ]] || { have_cmd docker && docker info >/dev/null 2>&1; }; then
    DOCKER=(docker)
  elif have_cmd docker; then
    DOCKER=(sudo docker)
  else
    DOCKER=(sudo docker)
  fi
}

docker_cmd() {
  get_docker_cmd
  "${DOCKER[@]}" "$@"
}

assert_local_docker_idle() {
  # A local systemd restart cannot repair a remote Docker context.
  local endpoint
  endpoint=$(docker context inspect --format '{{.Endpoints.docker.Host}}')
  [[ "${DOCKER_HOST:-$endpoint}" == "unix:///var/run/docker.sock" || \
     "${DOCKER_HOST:-$endpoint}" == "unix:///run/docker.sock" ]] || \
    error "Automatic Docker repair requires the default local Unix socket. Check DOCKER_HOST and the Docker context."
  cloud_python_as_root docker-idle
}

restart_docker_safely() {
  assert_local_docker_idle
  info "No running containers; restarting the local Docker service."
  as_root systemctl restart docker
}

require_docker_runtime() {
  have_cmd docker || error "Docker is not installed. Run: ./${SCRIPT_NAME} bootstrap"
  get_docker_cmd
  docker_cmd version >/dev/null 2>&1 || error "Docker is installed but not usable by this user. Run: ./${SCRIPT_NAME} bootstrap"
}

ensure_common_apt_bits() {
  apt_get update -y
  apt_get install -y ca-certificates curl gnupg software-properties-common lsb-release iproute2 net-tools netcat-openbsd rsync
}

ensure_nvidia_driver_present() {
  have_cmd nvidia-smi || error "nvidia-smi not found. Install the NVIDIA GPU driver first, or use a GPU image that already has it."
  if ! nvidia-smi >/dev/null 2>&1; then
    error "nvidia-smi exists but the driver/GPU is not healthy. Fix that before using Isaac Sim containers."
  fi
  local versions version
  versions=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
  [[ -n "$versions" ]] || error "No NVIDIA GPU detected."
  while IFS= read -r version; do
    version="${version//[[:space:]]/}"
    dpkg --compare-versions "$version" ge "$NVIDIA_MIN_DRIVER_VERSION" || \
      error "NVIDIA driver ${version} is too old; need ${NVIDIA_MIN_DRIVER_VERSION} or later. Use a prepared GPU host, or the explicit --vast-vm mode on a new VM."
  done <<< "$versions"
}

ensure_docker_installed() {
  if have_cmd docker && docker --version >/dev/null 2>&1; then
    info "Docker already installed: $(docker --version)"
  else
    info "Installing Docker Engine from Docker's apt repository..."
    ensure_common_apt_bits

    # Remove conflicting packages if present.
    local -a conflicts=()
    local pkg
    for pkg in docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc; do
      if package_installed "$pkg"; then conflicts+=("$pkg"); fi
    done
    if [[ ${#conflicts[@]} -gt 0 ]]; then apt_get remove -y "${conflicts[@]}"; fi

    as_root install -m 0755 -d /etc/apt/keyrings
    as_root curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    as_root chmod a+r /etc/apt/keyrings/docker.asc

    local codename arch
    codename=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
    arch=$(dpkg --print-architecture)
    as_root bash -lc "cat > /etc/apt/sources.list.d/docker.sources <<DOCKERREPO
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${codename}
Components: stable
Architectures: ${arch}
Signed-By: /etc/apt/keyrings/docker.asc
DOCKERREPO"

    apt_get update -y
    apt_get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    as_root systemctl enable --now docker
  fi

  # Let the current user use docker in future shells.
  local user
  user=$(current_user)
  if getent group docker >/dev/null 2>&1; then
    if ! id -nG "$user" | tr ' ' '\n' | grep -qx docker; then
      as_root usermod -aG docker "$user" || true
      DOCKER_GROUP_RELOGIN_REQUIRED=1
      warn "Added ${user} to docker group. A new shell/login is needed before plain 'docker' works without sudo."
    fi
  fi

  get_docker_cmd
  if ! docker_cmd version >/dev/null 2>&1; then
    warn "Docker client/daemon check failed; checking whether a safe restart is possible."
    restart_docker_safely
  fi
  docker_cmd version >/dev/null
}

ensure_nvidia_container_runtime() {
  ensure_nvidia_driver_present
  ensure_docker_installed

  local ok=0
  if docker_cmd run --rm --gpus all ubuntu:24.04 nvidia-smi >/dev/null 2>&1; then
    ok=1
  fi

  if [[ $ok -eq 1 ]]; then
    info "NVIDIA Container Toolkit already working with Docker."
    return 0
  fi

  info "Installing/configuring NVIDIA Container Toolkit..."
  assert_local_docker_idle
  ensure_common_apt_bits

  as_root bash -o pipefail -c 'curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg'
  as_root bash -o pipefail -c 'curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" > /etc/apt/sources.list.d/nvidia-container-toolkit.list'
  apt_get update -y
  apt_get install -y nvidia-container-toolkit
  as_root nvidia-ctk runtime configure --runtime=docker
  restart_docker_safely

  if ! docker_cmd run --rm --gpus all ubuntu:24.04 nvidia-smi >/dev/null 2>&1; then
    error "Docker still cannot see the GPU after configuring NVIDIA Container Toolkit. Check driver health, reboot if needed, then retry."
  fi

  info "NVIDIA Container Toolkit is working."
}

verify_nvidia_container_runtime() {
  ensure_nvidia_driver_present
  require_docker_runtime

  if ! docker_cmd run --rm --gpus all ubuntu:24.04 nvidia-smi >/dev/null 2>&1; then
    error "Docker cannot see the GPU. Run: ./${SCRIPT_NAME} bootstrap"
  fi
}

install_ros_apt_source() {
  ensure_common_apt_bits
  apt_get install -y software-properties-common
  package_command add-apt-repository -y --no-update universe

  local tmpdeb version codename
  tmpdeb=/tmp/ros2-apt-source.deb
  version=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F 'tag_name' | awk -F'"' '{print $4}')
  [[ -n "$version" ]] || error "Failed to resolve ros2-apt-source release version."
  codename=$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")
  curl -fL -o "$tmpdeb" "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${version}/ros2-apt-source_${version}.${codename}_all.deb"
  package_command dpkg -i "$tmpdeb"
}

ensure_ros2_installed() {
  require_supported_host
  init_paths

  if [[ -f "$ROS_SETUP" ]]; then
    info "ROS 2 ${ROS_DISTRO} already installed."
  else
    info "Installing ROS 2 ${ROS_DISTRO} (${ROS_INSTALL_VARIANT})..."
    install_ros_apt_source
    apt_get update -y

    if [[ "$ROS_DISTRO" == "humble" ]]; then
      # Safer than a full distro upgrade on remote GPU VMs: update the packages ROS docs explicitly warn about.
      apt_get install -y systemd systemd-sysv libsystemd0 udev
    fi

    local ros_pkg
    case "$ROS_INSTALL_VARIANT" in
      ros-base) ros_pkg="ros-${ROS_DISTRO}-ros-base" ;;
      desktop)  ros_pkg="ros-${ROS_DISTRO}-desktop" ;;
      *) error "ROS_INSTALL_VARIANT must be 'ros-base' or 'desktop'." ;;
    esac

    local rosdep_pkg=""
    if apt-cache show python3-rosdep >/dev/null 2>&1; then
      rosdep_pkg="python3-rosdep"
    elif apt-cache show python3-rosdep2 >/dev/null 2>&1; then
      rosdep_pkg="python3-rosdep2"
    fi

    apt_get install -y "$ros_pkg" ros-dev-tools python3-colcon-common-extensions ${rosdep_pkg:+$rosdep_pkg}
  fi

  if [[ ! -f "$ROS_SETUP" ]]; then
    error "ROS setup script not found at ${ROS_SETUP} after installation."
  fi

  if have_cmd rosdep; then
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
      as_root rosdep init || true
    fi
    rosdep update || true
  fi

  ensure_ros_domain_env_file
  ensure_ros_setup_in_bashrc
  ensure_ros_profile_script
}

ensure_ros_domain_env_file() {
  local env_dir="/etc/isaac-projects"
  local env_path="${env_dir}/ros.env"
  local tmp_file

  tmp_file=$(mktemp)
  cat >"$tmp_file" <<EOF_DOMAIN
# Managed by isaac_vmctl.sh. Sets the default ROS 2 DDS domain for this machine.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID}"
EOF_DOMAIN

  as_root install -d -m 0755 "$env_dir"
  as_root install -m 0644 "$tmp_file" "$env_path"
  rm -f "$tmp_file"
  info "Ensured ROS_DOMAIN_ID=${ROS_DOMAIN_ID} is sourced from ${env_path}."
}

ensure_ros_setup_in_bashrc() {
  local bashrc_path="${USER_HOME}/.bashrc"
  local marker_begin="# >>> isaac-projects ROS 2 setup >>>"
  local marker_end="# <<< isaac-projects ROS 2 setup <<<"
  local tmp_file user_group

  mkdir -p "$USER_HOME"
  touch "$bashrc_path"

  tmp_file=$(mktemp)
  if grep -Fqx "$marker_begin" "$bashrc_path"; then
    awk -v begin="$marker_begin" -v end="$marker_end" '
      $0 == begin { skip=1; next }
      $0 == end { skip=0; next }
      !skip { print }
    ' "$bashrc_path" >"$tmp_file"
  else
    cp "$bashrc_path" "$tmp_file"
  fi

  {
    printf '\n%s\n' "$marker_begin"
    printf 'if [ -z "${ROS_DOMAIN_ID:-}" ] && [ -r /etc/isaac-projects/ros.env ]; then\n'
    printf '  . /etc/isaac-projects/ros.env\n'
    printf 'fi\n'
    printf 'export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-%s}"\n' "$ROS_DOMAIN_ID"
    printf 'if [ "${ROS_DISTRO:-}" != "%s" ] && [ -f "%s" ]; then\n' "$ROS_DISTRO" "$ROS_SETUP"
    printf '  __isaac_projects_nounset_was_enabled=0\n'
    printf '  case $- in *u*) __isaac_projects_nounset_was_enabled=1; set +u ;; esac\n'
    printf '  source "%s"\n' "$ROS_SETUP"
    printf '  if [ "$__isaac_projects_nounset_was_enabled" = "1" ]; then set -u; fi\n'
    printf '  unset __isaac_projects_nounset_was_enabled\n'
    printf 'fi\n'
    printf '%s\n' "$marker_end"
  } >>"$tmp_file"

  if [[ ${EUID} -eq 0 ]]; then
    user_group=$(id -gn "$USER_NAME")
    install -o "$USER_NAME" -g "$user_group" -m 0644 "$tmp_file" "$bashrc_path"
  else
    install -m 0644 "$tmp_file" "$bashrc_path"
  fi
  rm -f "$tmp_file"
  info "Ensured ROS 2 setup is sourced from ${bashrc_path}."
}

ensure_ros_profile_script() {
  local profile_path="/etc/profile.d/isaac-projects-ros2.sh"
  local tmp_file marker_begin marker_end bash_block shell_file

  tmp_file=$(mktemp)
  cat >"$tmp_file" <<EOF_PROFILE
# Source ROS 2 and the project DDS domain for login shells on the GPU host.
if [ -z "\${ROS_DOMAIN_ID:-}" ] && [ -r /etc/isaac-projects/ros.env ]; then
  . /etc/isaac-projects/ros.env
fi
export ROS_DOMAIN_ID="\${ROS_DOMAIN_ID:-${ROS_DOMAIN_ID}}"
if [ "\${ROS_DISTRO:-}" != "${ROS_DISTRO}" ] && [ -f "${ROS_SETUP%setup.bash}setup.sh" ]; then
  __isaac_projects_nounset_was_enabled=0
  case \$- in *u*) __isaac_projects_nounset_was_enabled=1; set +u ;; esac
  . "${ROS_SETUP%setup.bash}setup.sh"
  if [ "\$__isaac_projects_nounset_was_enabled" = "1" ]; then set -u; fi
  unset __isaac_projects_nounset_was_enabled
fi
EOF_PROFILE
  as_root install -m 0644 "$tmp_file" "$profile_path"
  rm -f "$tmp_file"
  info "Ensured ROS 2 setup is sourced from ${profile_path}."

  marker_begin="# >>> isaac-projects ROS 2 setup >>>"
  marker_end="# <<< isaac-projects ROS 2 setup <<<"
  bash_block="if [ -z \"\${ROS_DOMAIN_ID:-}\" ] && [ -r /etc/isaac-projects/ros.env ]; then
  . /etc/isaac-projects/ros.env
fi
export ROS_DOMAIN_ID=\"\${ROS_DOMAIN_ID:-${ROS_DOMAIN_ID}}\"
if [ \"\${ROS_DISTRO:-}\" != \"${ROS_DISTRO}\" ] && [ -f \"${ROS_SETUP}\" ]; then
  __isaac_projects_nounset_was_enabled=0
  case \$- in *u*) __isaac_projects_nounset_was_enabled=1; set +u ;; esac
  source \"${ROS_SETUP}\"
  if [ \"\$__isaac_projects_nounset_was_enabled\" = \"1\" ]; then set -u; fi
  unset __isaac_projects_nounset_was_enabled
fi"
  for shell_file in /etc/bash.bashrc /etc/skel/.bashrc; do
    [[ -f "$shell_file" ]] || as_root touch "$shell_file"
    tmp_file=$(mktemp)
    awk -v begin="$marker_begin" -v end="$marker_end" '
      $0 == begin { skip=1; next }
      $0 == end { skip=0; next }
      !skip { print }
    ' "$shell_file" >"$tmp_file"
    {
      printf '\n%s\n' "$marker_begin"
      printf '%s\n' "$bash_block"
      printf '%s\n' "$marker_end"
    } >>"$tmp_file"
    as_root install -m 0644 "$tmp_file" "$shell_file"
    rm -f "$tmp_file"
  done
  info "Ensured ROS 2 setup is sourced from system bash startup files."
}

ensure_shell_block() {
  local shell_file="$1"
  local marker_name="$2"
  local block_body="$3"
  local marker_begin="# >>> isaac-projects ${marker_name} >>>"
  local marker_end="# <<< isaac-projects ${marker_name} <<<"
  local tmp_file user_group

  mkdir -p "$(dirname "$shell_file")"
  touch "$shell_file"

  tmp_file=$(mktemp)
  if grep -Fqx "$marker_begin" "$shell_file"; then
    awk -v begin="$marker_begin" -v end="$marker_end" '
      $0 == begin { skip=1; next }
      $0 == end { skip=0; next }
      !skip { print }
    ' "$shell_file" >"$tmp_file"
  else
    cp "$shell_file" "$tmp_file"
  fi

  {
    printf '\n%s\n' "$marker_begin"
    printf '%s\n' "$block_body"
    printf '%s\n' "$marker_end"
  } >>"$tmp_file"

  if [[ ${EUID} -eq 0 ]]; then
    user_group=$(id -gn "$USER_NAME")
    install -o "$USER_NAME" -g "$user_group" -m 0644 "$tmp_file" "$shell_file"
  else
    install -m 0644 "$tmp_file" "$shell_file"
  fi
  rm -f "$tmp_file"
}

ensure_user_local_bin_dir() {
  local bin_dir="${USER_HOME}/.local/bin"
  local user_group
  user_group=$(id -gn "$USER_NAME")

  if [[ ${EUID} -eq 0 ]]; then
    install -d -o "$USER_NAME" -g "$user_group" -m 0755 "$bin_dir"
  else
    install -d -m 0755 "$bin_dir"
  fi
}

ensure_user_local_bin_in_shell_startup() {
  local path_block='if [ -d "$HOME/.local/bin" ]; then
  case ":$PATH:" in
    *:"$HOME/.local/bin":*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
  esac
fi'

  ensure_shell_block "${USER_HOME}/.profile" "user-local-bin" "$path_block"
  ensure_shell_block "${USER_HOME}/.bashrc" "user-local-bin" "$path_block"
}

install_user_command_wrapper() {
  local wrapper_path="$1"
  local target_path="$2"
  local tmp_file

  tmp_file=$(mktemp)
  cat >"$tmp_file" <<EOF_WRAPPER
#!/usr/bin/env bash
exec "${target_path}" "\$@"
EOF_WRAPPER
  install_for_current_user 0755 "$tmp_file" "$wrapper_path"
  rm -f "$tmp_file"
}

ensure_student_python_venv() {
  local base_dir="${USER_HOME}/.local/share/isaac-projects/python-tools"
  local user_group
  user_group=$(id -gn "$USER_NAME")

  STUDENT_PYTHON_VENV="${base_dir}/venv"
  if [[ -x "${STUDENT_PYTHON_VENV}/bin/python" ]]; then
    return 0
  fi

  if [[ ${EUID} -eq 0 ]]; then
    install -d -o "$USER_NAME" -g "$user_group" -m 0755 "$base_dir"
  else
    install -d -m 0755 "$base_dir"
  fi

  info "Creating Python tools environment at ${STUDENT_PYTHON_VENV}..."
  as_current_user python3 -m venv "$STUDENT_PYTHON_VENV"
  as_current_user "${STUDENT_PYTHON_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
}

ensure_vscode_cli_installed() {
  local install_path="${USER_HOME}/.local/bin/code"
  local cli_arch url archive tmp_dir

  if have_cmd code || [[ -x "$install_path" ]]; then
    info "VS Code CLI already available: $(command -v code 2>/dev/null || printf '%s' "$install_path")"
    return 0
  fi

  case "$ARCH" in
    amd64) cli_arch="x64" ;;
    arm64) cli_arch="arm64" ;;
    armhf|armel) cli_arch="armhf" ;;
    *)
      warn "Skipping VS Code CLI install on unsupported architecture ${ARCH}. Remote-SSH can still install VS Code Server on first connection."
      return 0
      ;;
  esac

  url="https://update.code.visualstudio.com/latest/cli-linux-${cli_arch}/stable"
  archive=$(mktemp --suffix=.tar.gz)
  tmp_dir=$(mktemp -d)

  info "Installing VS Code CLI from ${url}..."
  curl -fsSL -o "$archive" "$url"
  tar -xzf "$archive" -C "$tmp_dir"
  [[ -x "${tmp_dir}/code" ]] || error "VS Code CLI archive did not contain the expected 'code' binary."
  install_for_current_user 0755 "${tmp_dir}/code" "$install_path"
  rm -f "$archive"
  rm -rf "$tmp_dir"
  info "Installed VS Code CLI to ${install_path}."
}

ensure_jupyter_installed() {
  local jupyter_packages=(jupyterlab notebook ipykernel)

  ensure_student_python_venv
  if as_current_user "${STUDENT_PYTHON_VENV}/bin/python" -m pip show "${jupyter_packages[@]}" >/dev/null 2>&1; then
    info "JupyterLab and Notebook already installed in ${STUDENT_PYTHON_VENV}."
  else
    info "Installing JupyterLab and Notebook into ${STUDENT_PYTHON_VENV}..."
    as_current_user "${STUDENT_PYTHON_VENV}/bin/python" -m pip install --upgrade "${jupyter_packages[@]}"
  fi

  install_user_command_wrapper "${USER_HOME}/.local/bin/jupyter" "${STUDENT_PYTHON_VENV}/bin/jupyter"
  install_user_command_wrapper "${USER_HOME}/.local/bin/jupyter-lab" "${STUDENT_PYTHON_VENV}/bin/jupyter-lab"
  install_user_command_wrapper "${USER_HOME}/.local/bin/jupyter-notebook" "${STUDENT_PYTHON_VENV}/bin/jupyter-notebook"
  as_current_user "${STUDENT_PYTHON_VENV}/bin/python" -m ipykernel install --user --name isaac-projects-tools --display-name "Python (isaac-projects)" >/dev/null 2>&1 || true
}

ensure_student_tooling() {
  local apt_packages=()
  local python_packages=()
  local extra_tool normalized_tool raw_tools

  ensure_user_local_bin_dir
  ensure_user_local_bin_in_shell_startup

  if is_truthy "$VSCODE_REMOTE_ENABLE"; then
    append_unique apt_packages openssh-server
    append_unique apt_packages git
  fi
  if is_truthy "$JUPYTER_ENABLE"; then
    append_unique apt_packages python3
    append_unique apt_packages python3-venv
    append_unique apt_packages python3-pip
  fi

  raw_tools="${STUDENT_EXTRA_TOOLS//,/ }"
  for extra_tool in $raw_tools; do
    normalized_tool=$(printf '%s' "$extra_tool" | tr '[:upper:]' '[:lower:]')
    case "$normalized_tool" in
      tmux|htop|btop|ncdu|nvtop|ripgrep|ffmpeg|mosh)
        append_unique apt_packages "$normalized_tool"
        ;;
      git-lfs)
        append_unique apt_packages git-lfs
        ;;
      nvitop)
        append_unique apt_packages python3
        append_unique apt_packages python3-venv
        append_unique apt_packages python3-pip
        append_unique python_packages nvitop
        ;;
      "")
        ;;
      *)
        error "Unsupported STUDENT_EXTRA_TOOLS entry: ${extra_tool}. Supported values: tmux htop btop ncdu nvtop nvitop ripgrep git-lfs ffmpeg mosh"
        ;;
    esac
  done

  if (( ${#apt_packages[@]} > 0 )); then
    ensure_common_apt_bits
    apt_get install -y "${apt_packages[@]}"
  fi

  if is_truthy "$VSCODE_REMOTE_ENABLE"; then
    if have_cmd systemctl; then
      as_root systemctl enable --now ssh
    else
      warn "systemctl is not available; ensure the SSH service is running before using VS Code Remote-SSH."
    fi
    ensure_vscode_cli_installed
  fi

  if is_truthy "$JUPYTER_ENABLE"; then
    ensure_jupyter_installed
  fi

  if (( ${#python_packages[@]} > 0 )); then
    ensure_student_python_venv
    if as_current_user "${STUDENT_PYTHON_VENV}/bin/python" -m pip show "${python_packages[@]}" >/dev/null 2>&1; then
      info "Optional Python student tools already installed in ${STUDENT_PYTHON_VENV}."
    else
      info "Installing optional Python student tools into ${STUDENT_PYTHON_VENV}: ${python_packages[*]}"
      as_current_user "${STUDENT_PYTHON_VENV}/bin/python" -m pip install --upgrade "${python_packages[@]}"
    fi
    local python_tool
    for python_tool in "${python_packages[@]}"; do
      case "$python_tool" in
        nvitop)
          install_user_command_wrapper "${USER_HOME}/.local/bin/nvitop" "${STUDENT_PYTHON_VENV}/bin/nvitop"
          ;;
      esac
    done
  fi

  if [[ " ${apt_packages[*]} " == *" git-lfs "* ]]; then
    as_current_user git lfs install --skip-repo >/dev/null 2>&1 || true
  fi
}

ensure_isaac_dirs() {
  info "Preparing Isaac Sim host directories under ${ISAAC_HOST_ROOT}..."
  mkdir -p \
    "${ISAAC_HOST_ROOT}/cache/main/ov" \
    "${ISAAC_HOST_ROOT}/cache/main/warp" \
    "${ISAAC_HOST_ROOT}/cache/computecache" \
    "${ISAAC_HOST_ROOT}/config" \
    "${ISAAC_HOST_ROOT}/data/documents" \
    "${ISAAC_HOST_ROOT}/data/Kit" \
    "${ISAAC_HOST_ROOT}/logs" \
    "${ISAAC_HOST_ROOT}/pkg"
  as_root chown -R "${CONTAINER_UID}:${CONTAINER_GID}" "${ISAAC_HOST_ROOT}"
}

ensure_isaac_image() {
  if docker_cmd image inspect "$ISAAC_IMAGE" >/dev/null 2>&1; then
    info "Isaac Sim image already present: ${ISAAC_IMAGE}"
    return 0
  fi

  info "Pulling Isaac Sim image: ${ISAAC_IMAGE}"
  if docker_cmd pull "$ISAAC_IMAGE"; then
    return 0
  fi

  if [[ -n "${NGC_API_KEY:-}" ]]; then
    warn "Initial pull failed; trying nvcr.io login with NGC_API_KEY..."
    printf '%s' "$NGC_API_KEY" | docker_cmd login nvcr.io -u '$oauthtoken' --password-stdin
    docker_cmd pull "$ISAAC_IMAGE"
    return 0
  fi

  error "Failed to pull ${ISAAC_IMAGE}. If nvcr.io requires auth in your environment, export NGC_API_KEY and rerun."
}

validate_isaac_ros_config() {
  case "$ISAAC_ROS_DISTRO" in
    auto|humble|jazzy) ;;
    *) error "ISAAC_ROS_DISTRO must be 'auto', 'humble', or 'jazzy'." ;;
  esac

  case "$ISAAC_ROS_INSTALL_VARIANT" in
    ros-base|desktop) ;;
    *) error "ISAAC_ROS_INSTALL_VARIANT must be 'ros-base' or 'desktop'." ;;
  esac
}

resolve_isaac_ros_distro() {
  validate_isaac_ros_config

  if [[ -n "$RESOLVED_ISAAC_ROS_DISTRO" ]]; then
    printf '%s' "$RESOLVED_ISAAC_ROS_DISTRO"
    return 0
  fi

  local image_ubuntu_version expected_ros_distro
  image_ubuntu_version=$(docker_cmd run --rm --entrypoint bash "$ISAAC_IMAGE" -c '. /etc/os-release && printf "%s" "${VERSION_ID}"')
  expected_ros_distro=$(ros_distro_for_ubuntu_version "$image_ubuntu_version") || error "Cannot auto-select ROS 2 distro for ${ISAAC_IMAGE}: unsupported Ubuntu ${image_ubuntu_version}."

  if [[ "$ISAAC_ROS_DISTRO" == "auto" ]]; then
    RESOLVED_ISAAC_ROS_DISTRO="$expected_ros_distro"
  elif [[ "$ISAAC_ROS_DISTRO" == "$expected_ros_distro" ]]; then
    RESOLVED_ISAAC_ROS_DISTRO="$ISAAC_ROS_DISTRO"
  else
    error "ISAAC_ROS_DISTRO=${ISAAC_ROS_DISTRO} is incompatible with ${ISAAC_IMAGE} Ubuntu ${image_ubuntu_version}; use ${expected_ros_distro} or auto."
  fi

  printf '%s' "$RESOLVED_ISAAC_ROS_DISTRO"
}

resolve_isaac_runtime_image() {
  if ! is_truthy "$ISAAC_ROS_ENABLE"; then
    printf '%s' "$ISAAC_IMAGE"
    return 0
  fi

  local ros_distro base_tag variant_tag
  ros_distro=$(resolve_isaac_ros_distro)
  if [[ -n "$ISAAC_ROS_IMAGE" ]]; then
    printf '%s' "$ISAAC_ROS_IMAGE"
    return 0
  fi

  base_tag=$(sanitize_docker_component "${ISAAC_IMAGE##*:}")
  variant_tag=$(sanitize_docker_component "$ISAAC_ROS_INSTALL_VARIANT")
  printf 'rice/isaac-sim-ros2:%s-%s-%s' "$base_tag" "$ros_distro" "$variant_tag"
}

ensure_isaac_runtime_image() {
  ensure_isaac_image

  if ! is_truthy "$ISAAC_ROS_ENABLE"; then
    ISAAC_RUNTIME_IMAGE="$ISAAC_IMAGE"
    return 0
  fi

  validate_isaac_ros_config

  local ros_distro runtime_image dockerfile_path dockerfile_hash base_image_id
  ros_distro=$(resolve_isaac_ros_distro)
  runtime_image=$(resolve_isaac_runtime_image)
  dockerfile_path="${SCRIPT_DIR}/containers/isaac-sim-ros2.Dockerfile"
  [[ -f "$dockerfile_path" ]] || error "ROS-enabled Isaac Sim Dockerfile not found at ${dockerfile_path}"

  dockerfile_hash=$(sha256sum "$dockerfile_path" | awk '{print $1}')
  [[ -n "$dockerfile_hash" ]] || error "Unable to compute the Dockerfile hash for ${dockerfile_path}"
  base_image_id=$(docker_cmd image inspect "$ISAAC_IMAGE" --format '{{ .Id }}')

  if docker_cmd image inspect "$runtime_image" >/dev/null 2>&1; then
    local image_base image_base_id image_distro image_variant image_dockerfile_hash
    image_base=$(docker_cmd image inspect "$runtime_image" --format '{{ index .Config.Labels "rice.isaacsim.ros2.base_image" }}' 2>/dev/null || true)
    image_base_id=$(docker_cmd image inspect "$runtime_image" --format '{{ index .Config.Labels "rice.isaacsim.ros2.base_image_id" }}' 2>/dev/null || true)
    image_distro=$(docker_cmd image inspect "$runtime_image" --format '{{ index .Config.Labels "rice.isaacsim.ros2.distro" }}' 2>/dev/null || true)
    image_variant=$(docker_cmd image inspect "$runtime_image" --format '{{ index .Config.Labels "rice.isaacsim.ros2.install_variant" }}' 2>/dev/null || true)
    image_dockerfile_hash=$(docker_cmd image inspect "$runtime_image" --format '{{ index .Config.Labels "rice.isaacsim.ros2.dockerfile_hash" }}' 2>/dev/null || true)
    if [[ "$image_base" == "$ISAAC_IMAGE" && "$image_base_id" == "$base_image_id" && "$image_distro" == "$ros_distro" && "$image_variant" == "$ISAAC_ROS_INSTALL_VARIANT" && "$image_dockerfile_hash" == "$dockerfile_hash" ]]; then
      info "ROS-enabled Isaac Sim image already present: ${runtime_image} (container ROS 2 ${ros_distro}; host ROS 2 ${ROS_DISTRO:-unknown})"
      ISAAC_RUNTIME_IMAGE="$runtime_image"
      return 0
    fi
    warn "Existing ROS-enabled Isaac Sim image ${runtime_image} does not match the requested config; rebuilding."
  fi

  info "Building ROS-enabled Isaac Sim image ${runtime_image} from ${ISAAC_IMAGE} (container ROS 2 ${ros_distro}; host ROS 2 ${ROS_DISTRO:-unknown}; ${ISAAC_ROS_INSTALL_VARIANT})..."
  docker_cmd build \
    -f "$dockerfile_path" \
    --build-arg "ISAAC_IMAGE=${ISAAC_IMAGE}" \
    --build-arg "ISAAC_BASE_IMAGE_ID=${base_image_id}" \
    --build-arg "ISAAC_ROS_DISTRO=${ros_distro}" \
    --build-arg "ISAAC_ROS_INSTALL_VARIANT=${ISAAC_ROS_INSTALL_VARIANT}" \
    --build-arg "ISAAC_ROS_DOCKERFILE_HASH=${dockerfile_hash}" \
    -t "$runtime_image" \
    "${SCRIPT_DIR}/containers" || error "Failed to build ROS-enabled Isaac Sim image ${runtime_image}."

  ISAAC_RUNTIME_IMAGE="$runtime_image"
}

get_public_ip() {
  local ip
  ip=$(curl -4 -fsSL ifconfig.me 2>/dev/null || true)
  if [[ -z "$ip" ]]; then
    ip=$(curl -4 -fsSL https://api.ipify.org 2>/dev/null || true)
  fi
  [[ -n "$ip" ]] || error "Could not determine the VM public IPv4 address."
  printf '%s' "$ip"
}

configure_ufw_if_active() {
  if ! have_cmd ufw; then
    return 0
  fi

  local ufw_status
  need_root
  ufw_status=$("${ROOT_PREFIX[@]}" ufw status 2>/dev/null | head -n1 || true)
  if [[ "$ufw_status" != Status:\ active* ]]; then
    if [[ -n "$ALLOWED_CLIENT_IP" ]]; then
      warn "ALLOWED_CLIENT_IP is set, but ufw is not active. Not enabling or changing firewall automatically to avoid breaking SSH."
    fi
    return 0
  fi

  info "ufw is active; ensuring rules exist for SSH and Isaac Sim WebRTC..."
  if [[ -n "$ALLOWED_CLIENT_IP" ]]; then
    as_root ufw allow from "$ALLOWED_CLIENT_IP" to any port 22 proto tcp
    as_root ufw allow from "$ALLOWED_CLIENT_IP" to any port "$WEBRTC_SIGNAL_PORT" proto tcp
    as_root ufw allow from "$ALLOWED_CLIENT_IP" to any port "$WEBRTC_STREAM_PORT" proto udp
  else
    as_root ufw allow 22/tcp
    as_root ufw allow "${WEBRTC_SIGNAL_PORT}/tcp"
    as_root ufw allow "${WEBRTC_STREAM_PORT}/udp"
  fi
}

validate_tigervnc_config() {
  TIGERVNC_DISPLAY="${TIGERVNC_DISPLAY#:}"

  [[ "$TIGERVNC_DISPLAY" =~ ^[0-9]+$ ]] || error "TIGERVNC_DISPLAY must be a number, for example 1."
  [[ "$TIGERVNC_DISPLAY" -ge 1 && "$TIGERVNC_DISPLAY" -le 99 ]] || error "TIGERVNC_DISPLAY must be between 1 and 99."

  if [[ -z "$TIGERVNC_PORT" ]]; then
    TIGERVNC_PORT=$((5900 + TIGERVNC_DISPLAY))
  fi
  [[ "$TIGERVNC_PORT" =~ ^[0-9]+$ ]] || error "TIGERVNC_PORT must be a TCP port number."
  [[ "$TIGERVNC_PORT" -ge 1 && "$TIGERVNC_PORT" -le 65535 ]] || error "TIGERVNC_PORT must be between 1 and 65535."

  [[ "$TIGERVNC_GEOMETRY" =~ ^[0-9]+x[0-9]+$ ]] || error "TIGERVNC_GEOMETRY must look like 1920x1080."
  [[ "$TIGERVNC_DEPTH" =~ ^[0-9]+$ ]] || error "TIGERVNC_DEPTH must be a number."

  case "$TIGERVNC_DESKTOP" in
    gnome|ubuntu|ubuntu-gnome|gnome-flashback|xfce) ;;
    *) error "TIGERVNC_DESKTOP must be 'gnome', 'gnome-flashback', or 'xfce'." ;;
  esac
}

tigervnc_desktop_label() {
  case "$TIGERVNC_DESKTOP" in
    gnome|ubuntu|ubuntu-gnome) printf 'Ubuntu GNOME desktop with Yaru theme' ;;
    xfce) printf 'XFCE desktop with GNOME Terminal and Ubuntu Yaru theme' ;;
    gnome-flashback) printf 'GNOME Flashback with Ubuntu Yaru theme' ;;
    *) printf '%s' "$TIGERVNC_DESKTOP" ;;
  esac
}

ensure_tigervnc_desktop_installed() {
  local -a packages missing
  packages=(
    tigervnc-standalone-server
    tigervnc-common
    gnome-terminal
    terminator
    nautilus
    dconf-cli
    gsettings-desktop-schemas
    dbus-x11
    xauth
    x11-xserver-utils
    x11-utils
    xterm
    adwaita-icon-theme
    yaru-theme-gtk
    yaru-theme-icon
  )

  case "$TIGERVNC_DESKTOP" in
    gnome|ubuntu|ubuntu-gnome)
      packages+=(
        ubuntu-session
        gnome-session
        gnome-shell
        gnome-shell-extension-appindicator
        gnome-shell-extension-desktop-icons-ng
        gnome-shell-extension-ubuntu-dock
        gnome-control-center
        gnome-keyring
        xdg-desktop-portal
        xdg-desktop-portal-gnome
        yaru-theme-gnome-shell
        ubuntu-wallpapers
      )
      ;;
    gnome-flashback)
      packages+=(
        gnome-session-flashback
        metacity
      )
      ;;
    xfce)
      packages+=(
        xfce4
        xfce4-goodies
        xfce4-terminal
        thunar
      )
      ;;
  esac
  missing=()

  local pkg
  for pkg in "${packages[@]}"; do
    if ! package_installed "$pkg"; then
      missing+=("$pkg")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    info "TigerVNC desktop packages are already installed."
    return 0
  fi

  info "Installing TigerVNC desktop packages..."
  ensure_common_apt_bits
  package_command add-apt-repository -y --no-update universe
  apt_get update -y
  apt_get install -y "${missing[@]}"
}

ensure_tigervnc_user_files() {
  TIGERVNC_XSTARTUP_UPDATED=0

  local vnc_dir="${USER_HOME}/.vnc"
  local xstartup_path="${vnc_dir}/xstartup"
  local passwd_path="${vnc_dir}/passwd"
  local generated_password_path="${vnc_dir}/isaac-projects-vnc-password.txt"
  local user_group
  user_group=$(id -gn "$USER_NAME")

  mkdir -p "$vnc_dir"
  if [[ ${EUID} -eq 0 ]]; then
    chown "$USER_NAME:$user_group" "$vnc_dir"
  fi
  chmod 700 "$vnc_dir"

  local tmp_xstartup
  tmp_xstartup=$(mktemp)
  cat >"$tmp_xstartup" <<'EOF_XSTARTUP'
#!/usr/bin/env bash
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

TIGERVNC_DESKTOP="${TIGERVNC_DESKTOP:-gnome}"
TIGERVNC_TERMINAL="${TIGERVNC_TERMINAL:-gnome-terminal}"
TIGERVNC_GTK_THEME="${TIGERVNC_GTK_THEME:-Yaru}"
TIGERVNC_ICON_THEME="${TIGERVNC_ICON_THEME:-Yaru}"

export TIGERVNC_DESKTOP TIGERVNC_TERMINAL TIGERVNC_GTK_THEME TIGERVNC_ICON_THEME
export XDG_SESSION_TYPE=x11
export GTK_THEME="$TIGERVNC_GTK_THEME"
export GDK_BACKEND=x11
export QT_QPA_PLATFORM=xcb
export NO_AT_BRIDGE=1

display_id="${DISPLAY#:}"
display_id="${display_id%%.*}"
if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR:-}" || ! -w "${XDG_RUNTIME_DIR:-}" ]]; then
  export XDG_RUNTIME_DIR="/tmp/runtime-${USER:-$(id -un)}-vnc-${display_id:-0}"
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
fi

start_vnc_clipboard() {
  if command -v tigervncconfig >/dev/null 2>&1; then
    tigervncconfig -iconic >/dev/null 2>&1 &
  elif command -v vncconfig >/dev/null 2>&1; then
    vncconfig -iconic >/dev/null 2>&1 &
  fi
}

if [[ -r "$HOME/.profile" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/.profile"
fi

start_vnc_clipboard

case "$TIGERVNC_DESKTOP" in
  gnome|ubuntu|ubuntu-gnome)
    export XDG_CURRENT_DESKTOP=ubuntu:GNOME
    export XDG_SESSION_DESKTOP=ubuntu
    export DESKTOP_SESSION=ubuntu
    export GNOME_SHELL_SESSION_MODE=ubuntu
    export XDG_MENU_PREFIX=gnome-
    if command -v gnome-session >/dev/null 2>&1; then
      exec dbus-run-session -- bash -lc '
        run_gnome_settings() {
          gsettings set org.gnome.desktop.interface gtk-theme "${TIGERVNC_GTK_THEME:-Yaru}" >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.interface icon-theme "${TIGERVNC_ICON_THEME:-Yaru}" >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.interface cursor-theme "Yaru" >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.interface monospace-font-name "Ubuntu Mono 12" >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.interface color-scheme "prefer-light" >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.session idle-delay 0 >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.screensaver lock-enabled false >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.screensaver ubuntu-lock-on-suspend false >/dev/null 2>&1 || true
          gsettings set org.gnome.desktop.lockdown disable-lock-screen true >/dev/null 2>&1 || true
        }
        run_gnome_settings
        # Trust only the terminal launcher generated by this bootstrap.
        gio set "$HOME/Desktop/GNOME Terminal.desktop" metadata::trusted true >/dev/null 2>&1 || true
        exec gnome-session --builtin --session=ubuntu
      '
    fi
    ;;
  xfce)
    export XDG_CURRENT_DESKTOP=XFCE
    export XDG_SESSION_DESKTOP=xfce
    export DESKTOP_SESSION=xfce
    if command -v startxfce4 >/dev/null 2>&1; then
      exec dbus-run-session -- bash -lc '
        xfconf-query -c xsettings -p /Net/ThemeName -n -t string -s "${TIGERVNC_GTK_THEME:-Yaru}" >/dev/null 2>&1 || true
        xfconf-query -c xsettings -p /Net/IconThemeName -n -t string -s "${TIGERVNC_ICON_THEME:-Yaru}" >/dev/null 2>&1 || true
        xfconf-query -c xsettings -p /Gtk/MonospaceFontName -n -t string -s "Ubuntu Mono 12" >/dev/null 2>&1 || true
        xfconf-query -c xfce4-session -p /general/SaveOnExit -n -t bool -s false >/dev/null 2>&1 || true
        if command -v "${TIGERVNC_TERMINAL:-gnome-terminal}" >/dev/null 2>&1; then
          mkdir -p "$HOME/.local/share/applications"
          xdg-mime default org.gnome.Terminal.desktop x-scheme-handler/terminal >/dev/null 2>&1 || true
        fi
        exec startxfce4
      '
    fi
    ;;
  gnome-flashback)
    export XDG_CURRENT_DESKTOP=GNOME-Flashback:GNOME
    export XDG_SESSION_DESKTOP=gnome-flashback-metacity
    export DESKTOP_SESSION=gnome-flashback-metacity
    if command -v gnome-session >/dev/null 2>&1; then
      exec dbus-run-session -- bash -lc '
        gsettings set org.gnome.desktop.interface gtk-theme "${TIGERVNC_GTK_THEME:-Yaru}" >/dev/null 2>&1 || true
        gsettings set org.gnome.desktop.interface icon-theme "${TIGERVNC_ICON_THEME:-Yaru}" >/dev/null 2>&1 || true
        gsettings set org.gnome.desktop.session idle-delay 0 >/dev/null 2>&1 || true
        gsettings set org.gnome.desktop.screensaver lock-enabled false >/dev/null 2>&1 || true
        gsettings set org.gnome.desktop.screensaver ubuntu-lock-on-suspend false >/dev/null 2>&1 || true
        gsettings set org.gnome.desktop.lockdown disable-lock-screen true >/dev/null 2>&1 || true
        exec gnome-session --session=gnome-flashback-metacity
      '
    fi
    ;;
esac

if command -v "$TIGERVNC_TERMINAL" >/dev/null 2>&1; then
  exec "$TIGERVNC_TERMINAL"
fi
exec xterm
EOF_XSTARTUP
  if [[ ! -f "$xstartup_path" ]] || ! cmp -s "$tmp_xstartup" "$xstartup_path"; then
    TIGERVNC_XSTARTUP_UPDATED=1
  fi
  install_for_current_user 0755 "$tmp_xstartup" "$xstartup_path"
  rm -f "$tmp_xstartup"

  local xfce_config_dir="${USER_HOME}/.config/xfce4"
  local helpers_path="${xfce_config_dir}/helpers.rc"
  local desktop_dir="${USER_HOME}/Desktop"
  local terminal_desktop_path="${desktop_dir}/GNOME Terminal.desktop"
  mkdir -p "$xfce_config_dir" "$desktop_dir"
  if [[ ${EUID} -eq 0 ]]; then
    chown "$USER_NAME:$user_group" "${USER_HOME}/.config" "$xfce_config_dir" "$desktop_dir"
  fi

  local tmp_helpers
  tmp_helpers=$(mktemp)
  cat >"$tmp_helpers" <<'EOF_HELPERS'
TerminalEmulator=gnome-terminal
EOF_HELPERS
  install_for_current_user 0644 "$tmp_helpers" "$helpers_path"
  rm -f "$tmp_helpers"

  local tmp_terminal_desktop
  tmp_terminal_desktop=$(mktemp)
  cat >"$tmp_terminal_desktop" <<'EOF_TERMINAL_DESKTOP'
[Desktop Entry]
Version=1.0
Type=Application
Name=GNOME Terminal
Comment=Open a command line
Exec=gnome-terminal
Icon=org.gnome.Terminal
Terminal=false
Categories=System;TerminalEmulator;
EOF_TERMINAL_DESKTOP
  install_for_current_user 0755 "$tmp_terminal_desktop" "$terminal_desktop_path"
  rm -f "$tmp_terminal_desktop"

  local password generated_password tmp_pass tmp_plain
  generated_password=0
  if [[ -n "$TIGERVNC_PASSWORD" ]]; then
    password="$TIGERVNC_PASSWORD"
  elif [[ -f "$generated_password_path" ]]; then
    password=$(tr -d '\r\n' <"$generated_password_path")
  else
    password=$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n')
    generated_password=1
  fi

  [[ -n "$password" ]] || error "TigerVNC password is empty. Set TIGERVNC_PASSWORD and retry."
  if [[ ${#password} -gt 8 ]]; then
    warn "TigerVNC classic VNC authentication uses only the first 8 password characters."
    password="${password:0:8}"
  fi

  tmp_pass=$(mktemp)
  printf '%s\n' "$password" | vncpasswd -f >"$tmp_pass"
  install_for_current_user 0600 "$tmp_pass" "$passwd_path"
  rm -f "$tmp_pass"

  if [[ "$generated_password" -eq 1 ]]; then
    tmp_plain=$(mktemp)
    printf '%s\n' "$password" >"$tmp_plain"
    install_for_current_user 0600 "$tmp_plain" "$generated_password_path"
    rm -f "$tmp_plain"
    info "Generated TigerVNC password saved to ${generated_password_path}."
  elif [[ -f "$generated_password_path" && -z "$TIGERVNC_PASSWORD" ]]; then
    info "Using existing generated TigerVNC password from ${generated_password_path}."
  else
    info "Using TigerVNC password from TIGERVNC_PASSWORD."
  fi
}

configure_tigervnc_ufw_if_active() {
  # Loopback-only VNC is reached over SSH; no public firewall rule is needed.
  if is_truthy "$TIGERVNC_LOCALHOST"; then return 0; fi
  if ! have_cmd ufw; then
    return 0
  fi

  local ufw_status
  need_root
  ufw_status=$("${ROOT_PREFIX[@]}" ufw status 2>/dev/null | head -n1 || true)
  if [[ "$ufw_status" != Status:\ active* ]]; then
    if [[ -n "$ALLOWED_CLIENT_IP" ]]; then
      warn "ALLOWED_CLIENT_IP is set, but ufw is not active. Not enabling or changing firewall automatically to avoid breaking SSH."
    fi
    return 0
  fi

  info "ufw is active; ensuring a TigerVNC rule exists..."
  if [[ -n "$ALLOWED_CLIENT_IP" ]]; then
    as_root ufw allow from "$ALLOWED_CLIENT_IP" to any port "$TIGERVNC_PORT" proto tcp
  else
    as_root ufw allow "${TIGERVNC_PORT}/tcp"
  fi
}

tigervnc_port_listening() {
  ss -lnt | awk '{print $4}' | grep -Eq "[:.]${TIGERVNC_PORT}$"
}

webrtc_port_listening() {
  ss -lntu | grep -Eq ":(${WEBRTC_SIGNAL_PORT}|${WEBRTC_STREAM_PORT})\b"
}

tigervnc_port_owner_pids() {
  ss -lntp 2>/dev/null | awk -v port=":${TIGERVNC_PORT}" '
    $4 ~ port "$" && $0 ~ /Xtigervnc/ {
      line=$0
      while (match(line, /pid=[0-9]+/)) {
        print substr(line, RSTART + 4, RLENGTH - 4)
        line=substr(line, RSTART + RLENGTH)
      }
    }
  ' | sort -u
}

tigervnc_port_owned_by_xtigervnc() {
  [[ -n "$(tigervnc_port_owner_pids)" ]]
}

tigervnc_display_running() {
  have_cmd vncserver || return 1
  as_current_user vncserver -list 2>/dev/null | awk '{print $1}' | grep -Eq "^:?${TIGERVNC_DISPLAY}$"
}

stop_tigervnc_processes_on_port() {
  local -a pids=()
  local pid pid_list
  pid_list=$(tigervnc_port_owner_pids)
  [[ -n "$pid_list" ]] || return 0

  while IFS= read -r pid; do
    [[ -n "$pid" ]] && pids+=("$pid")
  done <<< "$pid_list"

  [[ ${#pids[@]} -gt 0 ]] || return 0

  info "Stopping existing TigerVNC listener on ${TIGERVNC_PORT}/tcp..."
  kill "${pids[@]}" 2>/dev/null || true

  local _
  for _ in $(seq 1 10); do
    tigervnc_port_listening || return 0
    sleep 1
  done

  warn "TigerVNC did not stop cleanly; forcing old listener on ${TIGERVNC_PORT}/tcp to exit."
  kill -9 "${pids[@]}" 2>/dev/null || true
}

start_tigervnc_server() {
  local display=":${TIGERVNC_DISPLAY}"
  local localhost_value="no"
  if is_truthy "$TIGERVNC_LOCALHOST"; then
    localhost_value="yes"
  fi

  if tigervnc_display_running && tigervnc_port_listening && [[ "$TIGERVNC_XSTARTUP_UPDATED" -eq 0 ]]; then
    info "TigerVNC already listening on ${TIGERVNC_PORT}/tcp."
    return 0
  fi

  if tigervnc_port_listening && ! tigervnc_port_owned_by_xtigervnc; then
    error "TCP ${TIGERVNC_PORT} is already in use. Choose a different TIGERVNC_PORT."
  fi

  if tigervnc_port_listening; then
    info "Restarting existing TigerVNC listener on ${TIGERVNC_PORT}/tcp."
  elif tigervnc_display_running && [[ "$TIGERVNC_XSTARTUP_UPDATED" -eq 1 ]]; then
    info "TigerVNC startup file changed; restarting display ${display}."
  fi
  as_current_user vncserver -kill "$display" >/dev/null 2>&1 || true
  if tigervnc_port_listening; then
    stop_tigervnc_processes_on_port
  fi

  info "Starting TigerVNC desktop on ${display} (${TIGERVNC_PORT}/tcp, ${TIGERVNC_GEOMETRY})..."
  as_current_user env \
    TIGERVNC_DESKTOP="$TIGERVNC_DESKTOP" \
    TIGERVNC_TERMINAL="$TIGERVNC_TERMINAL" \
    TIGERVNC_GTK_THEME="${TIGERVNC_GTK_THEME:-Yaru}" \
    TIGERVNC_ICON_THEME="${TIGERVNC_ICON_THEME:-Yaru}" \
    vncserver "$display" \
    -geometry "$TIGERVNC_GEOMETRY" \
    -depth "$TIGERVNC_DEPTH" \
    -rfbport "$TIGERVNC_PORT" \
    -localhost "$localhost_value"

  local _
  for _ in $(seq 1 15); do
    if tigervnc_port_listening; then
      info "TigerVNC is listening on ${TIGERVNC_PORT}/tcp."
      return 0
    fi
    sleep 1
  done

  warn "TigerVNC did not appear on ${TIGERVNC_PORT}/tcp yet. Check ${USER_HOME}/.vnc/*.log."
  return 1
}

tigervnc_session_failed() {
  local pid
  while IFS= read -r pid; do
    if grep -zFxq "DISPLAY=:${TIGERVNC_DISPLAY}" "/proc/${pid}/environ" 2>/dev/null || \
       grep -zFxq "DISPLAY=:${TIGERVNC_DISPLAY}.0" "/proc/${pid}/environ" 2>/dev/null; then
      return 0
    fi
  done < <(pgrep -u "$USER_NAME" -f '^(/[^ ]*/)?gnome-session-failed([[:space:]]|$)' || true)
  return 1
}

tigervnc_ready() {
  [[ -s "${USER_HOME}/.vnc/passwd" ]] || return 1
  tigervnc_display_running || return 1
  tigervnc_port_owned_by_xtigervnc || return 1
  if tigervnc_session_failed; then return 1; fi
  if is_truthy "$TIGERVNC_LOCALHOST"; then
    ss -lntH | awk -v port=":${TIGERVNC_PORT}" '
      $4 ~ port "$" { seen=1; if ($4 !~ /^(127\.0\.0\.1|\[?::1\]?):/) bad=1 }
      END { exit (!seen || bad) }
    ' || return 1
  fi
  cloud_python vnc-probe "$TIGERVNC_PORT" || return 1
  local wm_window
  wm_window=$(as_current_user env DISPLAY=":${TIGERVNC_DISPLAY}" XAUTHORITY="${USER_HOME}/.Xauthority" \
    xprop -root _NET_SUPPORTING_WM_CHECK 2>/dev/null | sed -nE 's/.*window id # (0x[0-9a-fA-F]+).*/\1/p') || return 1
  [[ "$wm_window" =~ ^0x[0-9a-fA-F]*[1-9a-fA-F][0-9a-fA-F]*$ ]] || return 1
  as_current_user env DISPLAY=":${TIGERVNC_DISPLAY}" XAUTHORITY="${USER_HOME}/.Xauthority" \
    xwininfo -id "$wm_window" >/dev/null 2>&1 || return 1
}

verify_tigervnc_ready() {
  have_cmd vncserver && have_cmd vncpasswd && have_cmd xprop || error "TigerVNC/X11 tools are missing after installation."
  local attempt stable=0
  for ((attempt=0; attempt<45; attempt++)); do
    if tigervnc_ready; then stable=$((stable + 1)); else stable=0; fi
    if [[ "$stable" -ge 3 ]]; then
      info "TigerVNC protocol, password file, listener and desktop window manager are ready."
      return 0
    fi
    sleep 1
  done
  error "TigerVNC desktop failed readiness checks. Inspect ${USER_HOME}/.vnc/*.log; bootstrap has not completed."
}

tigervnc_service_name() {
  printf 'isaac-projects-vnc-%s-%s.service' "$(id -u "$USER_NAME")" "$TIGERVNC_DISPLAY"
}

ensure_tigervnc_service() {
  local unit_name launcher tmp_launcher tmp_unit localhost_value=no
  unit_name=$(tigervnc_service_name)
  launcher="/usr/local/lib/isaac-projects/${unit_name%.service}.sh"
  is_truthy "$TIGERVNC_LOCALHOST" && localhost_value=yes
  tmp_launcher=$(mktemp)
  tmp_unit=$(mktemp)
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    printf 'export HOME=%q USER=%q LOGNAME=%q\n' "$USER_HOME" "$USER_NAME" "$USER_NAME"
    printf 'export TIGERVNC_DESKTOP=%q TIGERVNC_TERMINAL=%q\n' "$TIGERVNC_DESKTOP" "$TIGERVNC_TERMINAL"
    printf 'if [[ "${1:-}" == --stop ]]; then exec %q -kill %q; fi\n' "$(command -v vncserver)" ":${TIGERVNC_DISPLAY}"
    printf 'exec %q -fg %q -geometry %q -depth %q -rfbport %q -localhost %q -SecurityTypes VncAuth -rfbauth %q -xstartup %q\n' \
      "$(command -v vncserver)" ":${TIGERVNC_DISPLAY}" "$TIGERVNC_GEOMETRY" "$TIGERVNC_DEPTH" \
      "$TIGERVNC_PORT" "$localhost_value" "${USER_HOME}/.vnc/passwd" "${USER_HOME}/.vnc/xstartup"
  } >"$tmp_launcher"
  cat >"$tmp_unit" <<EOF_VNC_SERVICE
[Unit]
Description=Isaac Projects TigerVNC desktop (${USER_NAME}, display ${TIGERVNC_DISPLAY})
After=network.target

[Service]
Type=simple
User=$(id -u "$USER_NAME")
PAMName=login
Environment=DISPLAY=:${TIGERVNC_DISPLAY} XDG_SESSION_TYPE=x11 XDG_SESSION_CLASS=user
ExecStart=/usr/bin/bash ${launcher}
ExecStop=/usr/bin/bash ${launcher} --stop
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF_VNC_SERVICE
  if as_root systemctl is-active --quiet "$unit_name"; then
    if ! cmp -s "$tmp_launcher" "$launcher" || \
       ! cmp -s "$tmp_unit" "/etc/systemd/system/${unit_name}" || [[ "$TIGERVNC_XSTARTUP_UPDATED" -eq 1 ]]; then
      rm -f "$tmp_launcher" "$tmp_unit"
      error "Desktop configuration changed while ${unit_name} is running. Coordinate stopping it, then rerun bootstrap."
    fi
    rm -f "$tmp_launcher" "$tmp_unit"
    return 0
  fi
  if tigervnc_port_listening; then
    tigervnc_display_running && tigervnc_port_owned_by_xtigervnc || error "VNC port is owned by another session; refusing service takeover."
    as_current_user vncserver -kill ":${TIGERVNC_DISPLAY}"
  fi
  as_root install -d -m 0755 /usr/local/lib/isaac-projects
  as_root install -m 0755 "$tmp_launcher" "$launcher"
  as_root install -m 0644 "$tmp_unit" "/etc/systemd/system/${unit_name}"
  rm -f "$tmp_launcher" "$tmp_unit"
  as_root systemctl daemon-reload
  as_root systemctl enable --now "$unit_name"
  # Remember a provider-mapped custom port for later start/status/doctor calls.
  local tmp_port
  tmp_port=$(mktemp)
  printf '%s\n' "$TIGERVNC_PORT" >"$tmp_port"
  install_for_current_user 0600 "$tmp_port" "${USER_HOME}/.vnc/isaac-projects-port"
  rm -f "$tmp_port"
}

print_tigervnc_connection_summary() {
  validate_tigervnc_config

  local public_ip password_hint ssh_port public_port port_variable
  public_ip="${PUBLIC_IPADDR:-}"
  if [[ -z "$public_ip" ]]; then public_ip=$(get_public_ip 2>/dev/null || true); fi
  password_hint="${USER_HOME}/.vnc/isaac-projects-vnc-password.txt"
  ssh_port="${VNC_SSH_PORT:-${VAST_TCP_PORT_22:-}}"
  if [[ -z "$ssh_port" ]]; then ssh_port=$(cloud_python provider-ssh-port || true); fi
  if [[ -z "$ssh_port" ]]; then
    if is_truthy "$VAST_VM_PREPARE"; then ssh_port='<public-ssh-port>'; else ssh_port=22; fi
  fi

  echo
  echo "TigerVNC desktop:"
  if is_truthy "$TIGERVNC_LOCALHOST"; then
    echo "  SSH tunnel on your PC (substitute your server's public SSH host/port):"
    printf '    ssh -N -L 15901:127.0.0.1:%s -p %q %q\n' "$TIGERVNC_PORT" "$ssh_port" "${USER_NAME}@${public_ip:-<server-ip>}"
    echo "  Viewer:       127.0.0.1::15901 (TigerVNC explicit TCP port)"
  else
    port_variable="VAST_TCP_PORT_${TIGERVNC_PORT}"
    public_port="${!port_variable:-}"
    if [[ -z "$public_port" ]]; then public_port=$(cloud_python provider-tcp-port "$TIGERVNC_PORT" || true); fi
    if [[ ! "$public_port" =~ ^[0-9]+$ ]] || ((public_port < 1 || public_port > 65535)); then
      if is_truthy "$VAST_VM_PREPARE"; then
        public_port='<mapped-public-vnc-port>'
        warn "No public VNC mapping found. Expose TCP ${TIGERVNC_PORT} in the Vast template (-p ${TIGERVNC_PORT}:${TIGERVNC_PORT}); do not use the SSH port for VNC."
      else
        public_port="$TIGERVNC_PORT"
      fi
    fi
    echo "  IP address:   ${public_ip:-<server-ip>}"
    echo "  Public port:  ${public_port}"
    echo "  Target:       ${public_ip:-<server-ip>}::${public_port} (TigerVNC explicit TCP port)"
  fi
  echo "  Display:      :${TIGERVNC_DISPLAY}"
  echo "  Geometry:     ${TIGERVNC_GEOMETRY}"
  echo "  Desktop:      $(tigervnc_desktop_label)"
  if is_truthy "$TIGERVNC_AUTOSTART"; then echo "  Service:      $(tigervnc_service_name)"; fi
  if [[ -f "$password_hint" && -z "$TIGERVNC_PASSWORD" ]]; then
    echo "  Password:     ${password_hint}"
  else
    echo "  Password:     value supplied by TIGERVNC_PASSWORD"
  fi
}

ensure_tigervnc_ready() {
  require_supported_host
  init_paths
  validate_tigervnc_config
  ensure_tigervnc_desktop_installed
  ensure_tigervnc_user_files
  configure_tigervnc_ufw_if_active
  if is_truthy "$TIGERVNC_AUTOSTART"; then
    ensure_tigervnc_service
  else
    start_tigervnc_server
  fi
  verify_tigervnc_ready
}

stop_tigervnc_server() {
  require_supported_host
  init_paths
  validate_tigervnc_config

  local display=":${TIGERVNC_DISPLAY}"
  if as_root systemctl is-active --quiet "$(tigervnc_service_name)"; then
    as_root systemctl stop "$(tigervnc_service_name)"
    return 0
  fi
  if have_cmd vncserver; then
    info "Stopping TigerVNC display ${display}..."
    as_current_user vncserver -kill "$display" || warn "TigerVNC display ${display} was not running."
  else
    warn "vncserver is not installed."
  fi
}

append_extra_args() {
  local -n target=$1
  local -a extra_args=()

  if [[ -n "$ISAAC_EXTRA_ARGS" ]]; then
    # ISAAC_EXTRA_ARGS is intentionally simple: whitespace-separated flags.
    # For complex commands, use: isaac_vmctl.sh run -- bash -lc '...'
    read -r -a extra_args <<< "$ISAAC_EXTRA_ARGS"
    target+=("${extra_args[@]}")
  fi
}

build_isaac_command() {
  local mode=$1
  local public_ip=${2:-}
  local tag
  tag="${ISAAC_IMAGE##*:}"
  if [[ "$mode" == "gui" ]]; then
    ISAAC_CMD=("./isaac-sim.sh")
  else
    ISAAC_CMD=("./runheadless.sh")
  fi

  if [[ "$mode" == "gui" ]]; then
    ISAAC_CMD+=(--allow-root)
  fi

  if [[ "$mode" == "webrtc" ]]; then
    if [[ "$tag" == 5.* ]]; then
      ISAAC_CMD+=(
        "--/app/livestream/publicEndpointAddress=${public_ip}"
        "--/app/livestream/port=${WEBRTC_SIGNAL_PORT}"
      )
    else
      ISAAC_CMD+=(
        "--/exts/omni.kit.livestream.app/primaryStream/publicIp=${public_ip}"
        "--/exts/omni.kit.livestream.app/primaryStream/signalPort=${WEBRTC_SIGNAL_PORT}"
        "--/exts/omni.kit.livestream.app/primaryStream/streamPort=${WEBRTC_STREAM_PORT}"
      )
    fi
  fi

  append_extra_args ISAAC_CMD
}

parse_shell_words() {
  local raw_words="$1"
  have_cmd python3 || error "python3 is required to parse Isaac Lab command strings. Run '${SCRIPT_NAME} bootstrap' first."
  python3 -c 'import shlex, sys; [print(arg) for arg in shlex.split(sys.argv[1])]' "$raw_words"
}

sanitize_docker_component() {
  local value
  value=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  value=$(printf '%s' "$value" | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//')
  printf '%s' "${value:-default}"
}

resolve_isaaclab_image() {
  if [[ -n "$ISAACLAB_IMAGE" ]]; then
    printf '%s' "$ISAACLAB_IMAGE"
    return 0
  fi

  local base_tag ref_tag framework_tag
  base_tag=$(sanitize_docker_component "${ISAAC_IMAGE##*:}")
  ref_tag=$(sanitize_docker_component "$ISAACLAB_REF")
  framework_tag=$(sanitize_docker_component "$ISAACLAB_FRAMEWORKS")
  printf 'rice/isaac-lab:%s-%s-%s' "$base_tag" "$ref_tag" "$framework_tag"
}

isaaclab_args_request_install() {
  case "${1:-}" in
    --install|-i) return 0 ;;
    *) return 1 ;;
  esac
}

isaaclab_args_need_gui() {
  local joined_args=" $* "

  case "$joined_args" in
    *" --headless "*|*" --headless"|*"--headless "*) return 1 ;;
  esac

  case "$joined_args" in
    *" -p "*|*" --python "*|*" -s "*|*" --sim "*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_isaaclab_workspace() {
  [[ -n "$ISAACLAB_PATH" ]] || error "ISAACLAB_PATH must not be empty."
  [[ "$ISAACLAB_PATH" != /* ]] || error "ISAACLAB_PATH must be a path inside the mounted workspace, not an absolute path."
  [[ -n "$ISAACLAB_REPO_URL" ]] || error "ISAACLAB_REPO_URL must not be empty."
  [[ -n "$ISAACLAB_REF" ]] || error "ISAACLAB_REF must not be empty."
  [[ -n "$ISAACLAB_FRAMEWORKS" ]] || error "ISAACLAB_FRAMEWORKS must not be empty."
}

ensure_git_installed() {
  if have_cmd git; then
    return 0
  fi
  ensure_common_apt_bits
  apt_get install -y git
}

ensure_isaaclab_runtime_link() {
  local checkout_path="$1"
  local link_path="${checkout_path}/_isaac_sim"
  local link_target="/isaac-sim"

  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink "$link_path")" == "$link_target" ]]; then
      return 0
    fi
    rm -f "$link_path"
  elif [[ -e "$link_path" ]]; then
    warn "Isaac Lab checkout already has ${link_path}; leaving it in place."
    return 0
  fi

  as_current_user ln -s "$link_target" "$link_path"
  info "Prepared Isaac Lab runtime link: ${link_path} -> ${link_target}"
}

ensure_isaaclab_checkout() {
  validate_isaaclab_workspace
  ensure_git_installed

  local checkout_path="${HOST_WORKSPACE_ROOT}/${ISAACLAB_PATH}"
  local checkout_parent
  checkout_parent=$(dirname "$checkout_path")
  local user_group
  user_group=$(id -gn "$USER_NAME")
  if [[ ${EUID} -eq 0 ]]; then
    install -d -o "$USER_NAME" -g "$user_group" -m 0755 "$checkout_parent"
  else
    install -d -m 0755 "$checkout_parent"
  fi

  if [[ -e "$checkout_path" && ! -d "$checkout_path" ]]; then
    error "ISAACLAB_PATH points to a non-directory: ${checkout_path}"
  fi

  if [[ ! -d "${checkout_path}/.git" ]]; then
    if [[ -d "$checkout_path" ]] && [[ -n "$(find "$checkout_path" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      error "Isaac Lab checkout path exists but is not a git checkout: ${checkout_path}"
    fi
    info "Cloning Isaac Lab ${ISAACLAB_REF} into ${checkout_path}..."
    as_current_user git clone "$ISAACLAB_REPO_URL" "$checkout_path"
  fi

  [[ -d "${checkout_path}/.git" ]] || error "Isaac Lab checkout did not initialize correctly at ${checkout_path}"

  local dirty_checkout=""
  dirty_checkout=$(as_current_user git -C "$checkout_path" status --porcelain --untracked-files=no 2>/dev/null || true)
  if [[ -n "$dirty_checkout" ]]; then
    ensure_isaaclab_runtime_link "$checkout_path"
    warn "Isaac Lab checkout has local changes at ${checkout_path}; leaving the checkout as-is and building from the current workspace contents."
    return 0
  fi

  info "Syncing Isaac Lab checkout to ${ISAACLAB_REF}..."
  as_current_user git -C "$checkout_path" remote set-url origin "$ISAACLAB_REPO_URL" >/dev/null 2>&1 || true
  as_current_user git -C "$checkout_path" fetch --tags --force origin
  as_current_user git -C "$checkout_path" checkout --force "$ISAACLAB_REF"

  if as_current_user git -C "$checkout_path" rev-parse --verify "origin/${ISAACLAB_REF}" >/dev/null 2>&1; then
    as_current_user git -C "$checkout_path" reset --hard "origin/${ISAACLAB_REF}" >/dev/null
  fi

  local current_rev
  current_rev=$(as_current_user git -C "$checkout_path" rev-parse --short HEAD)
  ensure_isaaclab_runtime_link "$checkout_path"
  # This generated dependency is local installation data, not a project change.
  local local_exclude
  local_exclude=$(git -C "$HOST_WORKSPACE_ROOT" rev-parse --path-format=absolute --git-path info/exclude 2>/dev/null || true)
  if [[ -f "$local_exclude" ]] && ! grep -Fxq "/${ISAACLAB_PATH}/" "$local_exclude"; then
    as_current_user bash -c 'printf "\n%s\n" "$1" >> "$2"' _ "/${ISAACLAB_PATH}/" "$local_exclude"
  fi
  info "Isaac Lab checkout ready at ${checkout_path} (${current_rev})."
}

isaaclab_dockerfile_source() {
  # Keep the repository Dockerfile unchanged. Kit 5.1 bundles packaging 23;
  # replacing it breaks pip's vendored imports during Lab installation.
  if [[ "$ISAAC_IMAGE" != *:5.1.0 ]]; then cat "$1"; return; fi
  /usr/bin/python3 - "$1" <<'ISAACLAB_DOCKERFILE'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
anchor = "WORKDIR /workspace/isaac-projects\n"
if anchor not in source:
    raise SystemExit("Cannot locate the Isaac Lab Dockerfile installation section")
compatibility = r'''# Preserve Kit's packaging and satisfy Lab's Starlette/FastAPI combination.
ENV PIP_CONSTRAINT=/tmp/isaaclab-constraints.txt
RUN printf '%s\n' 'packaging==23.0' 'fastapi==0.121.3' > "$PIP_CONSTRAINT" && \
    /isaac-sim/python.sh -m pip install 'fastapi==0.121.3'

'''
print(source.replace(anchor, compatibility + anchor, 1), end="")
ISAACLAB_DOCKERFILE
}

ensure_isaaclab_image() {
  validate_isaaclab_workspace
  ensure_isaac_runtime_image
  ensure_isaaclab_checkout

  local checkout_path="${HOST_WORKSPACE_ROOT}/${ISAACLAB_PATH}"
  local helper_path="${checkout_path}/isaaclab.sh"
  [[ -x "$helper_path" || -f "$helper_path" ]] || error "Isaac Lab helper not found at ${helper_path}"
  local checkout_rev
  checkout_rev=$(as_current_user git -C "$checkout_path" rev-parse HEAD 2>/dev/null || true)
  [[ -n "$checkout_rev" ]] || error "Unable to determine the current Isaac Lab revision at ${checkout_path}"
  local dirty_checkout=""
  dirty_checkout=$(as_current_user git -C "$checkout_path" status --porcelain --untracked-files=no 2>/dev/null || true)
  local dockerfile_path="${HOST_WORKSPACE_ROOT}/containers/isaac-lab.Dockerfile"
  [[ -f "$dockerfile_path" ]] || error "Isaac Lab Dockerfile not found at ${dockerfile_path}"
  local dockerfile_hash dockerfile_source
  dockerfile_source=$(isaaclab_dockerfile_source "$dockerfile_path")
  dockerfile_hash=$(printf '%s\n' "$dockerfile_source" | sha256sum | awk '{print $1}')
  [[ -n "$dockerfile_hash" ]] || error "Unable to compute the Dockerfile hash for ${dockerfile_path}"

  local isaaclab_image isaaclab_base_image
  isaaclab_image=$(resolve_isaaclab_image)
  isaaclab_base_image="$ISAAC_RUNTIME_IMAGE"

  if docker_cmd image inspect "$isaaclab_image" >/dev/null 2>&1; then
    local image_ref image_frameworks image_path image_base image_checkout_rev image_dockerfile_hash
    image_ref=$(docker_cmd image inspect "$isaaclab_image" --format '{{ index .Config.Labels "rice.isaaclab.ref" }}' 2>/dev/null || true)
    image_frameworks=$(docker_cmd image inspect "$isaaclab_image" --format '{{ index .Config.Labels "rice.isaaclab.frameworks" }}' 2>/dev/null || true)
    image_path=$(docker_cmd image inspect "$isaaclab_image" --format '{{ index .Config.Labels "rice.isaaclab.path" }}' 2>/dev/null || true)
    image_base=$(docker_cmd image inspect "$isaaclab_image" --format '{{ index .Config.Labels "rice.isaaclab.base_image" }}' 2>/dev/null || true)
    image_checkout_rev=$(docker_cmd image inspect "$isaaclab_image" --format '{{ index .Config.Labels "rice.isaaclab.checkout_rev" }}' 2>/dev/null || true)
    image_dockerfile_hash=$(docker_cmd image inspect "$isaaclab_image" --format '{{ index .Config.Labels "rice.isaaclab.dockerfile_hash" }}' 2>/dev/null || true)
    if [[ -z "$dirty_checkout" && "$image_ref" == "$ISAACLAB_REF" && "$image_frameworks" == "$ISAACLAB_FRAMEWORKS" && "$image_path" == "$ISAACLAB_PATH" && "$image_base" == "$isaaclab_base_image" && "$image_checkout_rev" == "$checkout_rev" && "$image_dockerfile_hash" == "$dockerfile_hash" ]]; then
      info "Isaac Lab image already present: ${isaaclab_image}"
      return 0
    fi
    warn "Existing Isaac Lab image ${isaaclab_image} does not match the requested config; rebuilding."
  fi

  if [[ -n "$dirty_checkout" ]]; then
    warn "Isaac Lab checkout has local modifications at ${checkout_path}; building the managed image from the current workspace contents."
  fi
  info "Building Isaac Lab image ${isaaclab_image} from ${checkout_path} (${ISAACLAB_REF}, ${ISAACLAB_FRAMEWORKS}, ${checkout_rev:0:12})..."
  docker_cmd build \
    -f - \
    --build-arg "ISAAC_IMAGE=${isaaclab_base_image}" \
    --build-arg "ISAACLAB_PATH=${ISAACLAB_PATH}" \
    --build-arg "ISAACLAB_REF=${ISAACLAB_REF}" \
    --build-arg "ISAACLAB_FRAMEWORKS=${ISAACLAB_FRAMEWORKS}" \
    --build-arg "ISAACLAB_CHECKOUT_REV=${checkout_rev}" \
    --build-arg "ISAACLAB_DOCKERFILE_HASH=${dockerfile_hash}" \
    -t "$isaaclab_image" \
    "$HOST_WORKSPACE_ROOT" <<< "$dockerfile_source" || error "Failed to build Isaac Lab image ${isaaclab_image}."
}

prepare_gui_x11_access() {
  GUI_DISPLAY="${DISPLAY:-}"
  [[ -n "$GUI_DISPLAY" ]] || error "DISPLAY is not set. Run '${SCRIPT_NAME} start isaacsim --gui' from a terminal inside the GUI session, or export DISPLAY first."

  local resolved_xauthority="${XAUTHORITY:-}"
  if [[ -z "$resolved_xauthority" ]]; then
    resolved_xauthority="${HOME}/.Xauthority"
  fi
  GUI_XAUTH_SOURCE="$resolved_xauthority"
  GUI_XAUTH_FILE="${ISAAC_HOST_ROOT}/xauth/gui-$(current_user)-${GUI_DISPLAY//[:\/]/_}.xauth"
  GUI_XAUTH_CONTAINER_FILE="/tmp/.isaac-gui-${GUI_DISPLAY//[:\/]/_}.xauth"

  mkdir -p "$(dirname "$GUI_XAUTH_FILE")"
  : >"$GUI_XAUTH_FILE"
  as_root chown "${CONTAINER_UID}:${CONTAINER_GID}" "$GUI_XAUTH_FILE"
  chmod 0600 "$GUI_XAUTH_FILE"

  local xauth_copied=0
  if have_cmd xauth && [[ -f "$GUI_XAUTH_SOURCE" ]]; then
    local xauth_output
    xauth_output=$(env DISPLAY="$GUI_DISPLAY" XAUTHORITY="$GUI_XAUTH_SOURCE" xauth nlist "$GUI_DISPLAY" 2>/dev/null || true)
    if [[ -n "$xauth_output" ]]; then
      printf '%s\n' "$xauth_output" | sed -e 's/^..../ffff/' | xauth -f "$GUI_XAUTH_FILE" nmerge - >/dev/null 2>&1 || true
      if [[ -s "$GUI_XAUTH_FILE" ]]; then
        xauth_copied=1
      fi
    fi
  fi

  if [[ "$xauth_copied" -eq 1 ]]; then
    info "Prepared Xauthority for GUI display ${GUI_DISPLAY}."
    return 0
  fi

  warn "Could not copy Xauthority for ${GUI_DISPLAY}; allowing local X11 clients for this session."
  if env DISPLAY="$GUI_DISPLAY" XAUTHORITY="$GUI_XAUTH_SOURCE" xhost +local: >/dev/null 2>&1; then
    return 0
  fi
  if env DISPLAY="$GUI_DISPLAY" xhost +local: >/dev/null 2>&1; then
    return 0
  fi

  error "Unable to prepare X11 access for display ${GUI_DISPLAY}. Run this from the GUI desktop terminal and confirm X11 clients can open."
}

build_common_docker_args() {
  DOCKER_RUN_ARGS=(
    --gpus all
    --network=host
    --ipc=host
    -e ACCEPT_EULA=Y
    -e PRIVACY_CONSENT=Y
    # The ROS image is built as root; keep Kit's user files in the mounted
    # Isaac home even when Docker chooses /root for the selected UID.
    -e HOME=/isaac-sim
    -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
    -e "CONTAINER_WORKSPACE=${CONTAINER_WORKSPACE}"
    -v "${ISAAC_HOST_ROOT}/cache/main:/isaac-sim/.cache:rw"
    -v "${ISAAC_HOST_ROOT}/cache/computecache:/isaac-sim/.nv/ComputeCache:rw"
    -v "${ISAAC_HOST_ROOT}/logs:/isaac-sim/.nvidia-omniverse/logs:rw"
    -v "${ISAAC_HOST_ROOT}/config:/isaac-sim/.nvidia-omniverse/config:rw"
    -v "${ISAAC_HOST_ROOT}/data:/isaac-sim/.local/share/ov/data:rw"
    -v "${ISAAC_HOST_ROOT}/pkg:/isaac-sim/.local/share/ov/pkg:rw"
    -v "${HOST_WORKSPACE_ROOT}:${CONTAINER_WORKSPACE}:rw"
    -w "${CONTAINER_WORKSPACE}"
    -u "${CONTAINER_UID}:${CONTAINER_GID}"
  )

  if [[ -n "$PRIVACY_USERID" ]]; then
    DOCKER_RUN_ARGS+=(-e "PRIVACY_USERID=${PRIVACY_USERID}")
  fi
  if [[ -n "$RMW_IMPLEMENTATION" ]]; then
    DOCKER_RUN_ARGS+=(-e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}")
  fi
}

append_gui_docker_args() {
  DOCKER_RUN_ARGS+=(
    -e "DISPLAY=${GUI_DISPLAY}"
    -e "XAUTHORITY=${GUI_XAUTH_CONTAINER_FILE}"
    -e "QT_X11_NO_MITSHM=1"
    -e "__GLX_VENDOR_LIBRARY_NAME=nvidia"
    -e "NVIDIA_DRIVER_CAPABILITIES=all"
    -v "/tmp/.X11-unix:/tmp/.X11-unix:rw"
    -v "${GUI_XAUTH_FILE}:${GUI_XAUTH_CONTAINER_FILE}:ro"
  )
}

remove_existing_container() {
  if docker_cmd ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    info "Removing existing container named ${CONTAINER_NAME}..."
    docker_cmd rm -f "$CONTAINER_NAME" >/dev/null || true
  fi
}

make_ephemeral_container_name() {
  local prefix="$1"
  printf '%s-%s-%s' "$prefix" "$(date -u +%Y%m%d-%H%M%S)" "$$"
}

run_docker_attached() {
  local name_prefix="$1"
  shift

  local run_container_name
  run_container_name=$(make_ephemeral_container_name "$name_prefix")

  local docker_pid=""
  local run_exit=0
  local interrupted=0

  stop_run_container() {
    if [[ "$interrupted" -eq 0 ]]; then
      interrupted=1
      warn "Interrupt received. Stopping container ${run_container_name}..."
    fi
    docker_cmd stop -t 10 "$run_container_name" >/dev/null 2>&1 || true
  }

  get_docker_cmd
  trap 'stop_run_container' INT TERM
  "${DOCKER[@]}" run --rm --name "$run_container_name" "$@" &
  docker_pid=$!

  if wait "$docker_pid"; then
    run_exit=0
  else
    run_exit=$?
  fi

  trap - INT TERM

  if [[ "$interrupted" -eq 1 ]]; then
    if ! wait "$docker_pid" 2>/dev/null; then
      :
    fi
    return 130
  fi

  return "$run_exit"
}

start_isaacsim() {
  local mode="webrtc"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --headless) mode="headless"; shift ;;
      --webrtc)   mode="webrtc"; shift ;;
      --vnc|--desktop|--gui) mode="gui"; shift ;;
      *) error "Unknown start option: $1. Use --webrtc, --gui, or --headless." ;;
    esac
  done

  require_supported_host
  init_paths
  verify_nvidia_container_runtime
  ensure_isaac_dirs
  ensure_isaac_runtime_image
  if [[ "$mode" == "webrtc" ]]; then
    configure_ufw_if_active
  elif [[ "$mode" == "gui" ]]; then
    prepare_gui_x11_access
  fi

  local public_ip signal_ready ros_status container_ros_status
  public_ip="not needed"
  if [[ "$mode" == "webrtc" ]]; then
    public_ip=$(get_public_ip)
  fi
  if [[ -f "$ROS_SETUP" ]]; then
    ros_status="${ROS_DISTRO} (${ROS_SETUP})"
  else
    ros_status="not installed on host; run ./${SCRIPT_NAME} bootstrap if Zenoh/host ROS 2 is needed"
  fi
  container_ros_status="disabled"
  if is_truthy "$ISAAC_ROS_ENABLE"; then
    container_ros_status="$(resolve_isaac_ros_distro) (${ISAAC_RUNTIME_IMAGE})"
  fi

  build_isaac_command "$mode" "$public_ip"
  build_common_docker_args
  local docker_launch_args=(bash -c 'cd /isaac-sim && exec "$@"' _)
  if [[ "$mode" == "gui" ]]; then
    append_gui_docker_args
    DOCKER_RUN_ARGS+=(--entrypoint bash)
    docker_launch_args=(-c 'cd /isaac-sim && exec "$@"' _)
  fi
  remove_existing_container

  info "Starting ${CONTAINER_NAME} from ${ISAAC_RUNTIME_IMAGE} (${mode})..."
  docker_cmd run -d \
    --name "$CONTAINER_NAME" \
    "${DOCKER_RUN_ARGS[@]}" \
    "$ISAAC_RUNTIME_IMAGE" \
    "${docker_launch_args[@]}" "${ISAAC_CMD[@]}"

  if [[ "$mode" == "headless" ]]; then
    cat <<EOF_SUMMARY

Isaac Sim headless start requested.

Container:      ${CONTAINER_NAME}
Image:          ${ISAAC_RUNTIME_IMAGE}
Workspace:      ${HOST_WORKSPACE_ROOT} -> ${CONTAINER_WORKSPACE}
Host ROS 2:     ${ros_status}
Container ROS 2: ${container_ros_status}
ROS_DOMAIN_ID:  ${ROS_DOMAIN_ID}

Useful commands:
  ${SCRIPT_NAME} status
  ${SCRIPT_NAME} logs
  ${SCRIPT_NAME} shell
EOF_SUMMARY
    return 0
  fi

  if [[ "$mode" == "gui" ]]; then
    cat <<EOF_SUMMARY

Isaac Sim GUI start requested.

Container:      ${CONTAINER_NAME}
Image:          ${ISAAC_RUNTIME_IMAGE}
Workspace:      ${HOST_WORKSPACE_ROOT} -> ${CONTAINER_WORKSPACE}
X display:      ${GUI_DISPLAY}
Host ROS 2:     ${ros_status}
Container ROS 2: ${container_ros_status}
ROS_DOMAIN_ID:  ${ROS_DOMAIN_ID}

Useful commands:
  ${SCRIPT_NAME} status
  ${SCRIPT_NAME} logs
  ${SCRIPT_NAME} shell
  ${SCRIPT_NAME} check

Isaac Sim should open on the current X display. When using TigerVNC, run this from the terminal inside that desktop.
EOF_SUMMARY
    return 0
  fi

  info "Container launched. Waiting up to ${START_TIMEOUT_SEC}s for the WebRTC signal port..."
  sleep 5
  docker_cmd logs --tail 80 "$CONTAINER_NAME" || true

  signal_ready=0
  for _ in $(seq 1 "$START_TIMEOUT_SEC"); do
    if docker_cmd ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      :
    else
      error "Container exited during startup. Run '${SCRIPT_NAME} logs' to inspect the failure."
    fi
    if ss -lnt | awk '{print $4}' | grep -Eq "[:.]${WEBRTC_SIGNAL_PORT}$"; then
      signal_ready=1
      break
    fi
    sleep 1
  done

  if [[ "$signal_ready" -eq 1 ]]; then
    info "WebRTC signal port ${WEBRTC_SIGNAL_PORT}/tcp is listening."
  else
    warn "Timed out waiting for ${WEBRTC_SIGNAL_PORT}/tcp. Isaac Sim may still be starting; run '${SCRIPT_NAME} logs' or '${SCRIPT_NAME} check' next."
  fi

  cat <<EOF_SUMMARY

Isaac Sim start requested.

Public IP:      ${public_ip}
Signal port:    ${WEBRTC_SIGNAL_PORT}/tcp
Media port:     ${WEBRTC_STREAM_PORT}/udp
Container:      ${CONTAINER_NAME}
Image:          ${ISAAC_RUNTIME_IMAGE}
Host ROS 2:     ${ros_status}
Container ROS 2: ${container_ros_status}
ROS_DOMAIN_ID:  ${ROS_DOMAIN_ID}
Workspace:      ${HOST_WORKSPACE_ROOT} -> ${CONTAINER_WORKSPACE}

Useful commands:
  ${SCRIPT_NAME} status
  ${SCRIPT_NAME} logs
  ${SCRIPT_NAME} check
  ${SCRIPT_NAME} shell

Connect from your PC with the Isaac Sim WebRTC client to:
  ${public_ip}
EOF_SUMMARY
}

stop_isaacsim() {
  require_docker_runtime
  if docker_cmd ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    info "Stopping ${CONTAINER_NAME}..."
    docker_cmd stop "$CONTAINER_NAME" >/dev/null
  else
    warn "${CONTAINER_NAME} is not running."
  fi
}

stop_zenoh_bridge() {
  init_paths

  local stopped=0
  local pattern='[z]enoh-bridge-ros2dds.*router'
  local docker_exec_pattern='[d]ocker exec .*start_zenoh_bridge\.sh'

  if have_cmd docker; then
    get_docker_cmd
    if "${DOCKER[@]}" ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME"; then
      if "${DOCKER[@]}" exec "$CONTAINER_NAME" pgrep -f "$pattern" >/dev/null 2>&1; then
        info "Stopping Zenoh bridge inside ${CONTAINER_NAME}..."
        "${DOCKER[@]}" exec "$CONTAINER_NAME" pkill -TERM -f "$pattern" >/dev/null 2>&1 || true
        stopped=1
      fi
    fi
  fi

  if pgrep -f "$pattern" >/dev/null 2>&1; then
    info "Stopping host-visible Zenoh bridge processes..."
    pkill -TERM -f "$pattern" >/dev/null 2>&1 || true
    stopped=1
  fi
  if pgrep -f "$docker_exec_pattern" >/dev/null 2>&1; then
    pkill -TERM -f "$docker_exec_pattern" >/dev/null 2>&1 || true
    stopped=1
  fi

  sleep 1
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    warn "Zenoh bridge did not exit after SIGTERM; forcing remaining bridge processes down."
    pkill -KILL -f "$pattern" >/dev/null 2>&1 || true
    stopped=1
  fi
  if pgrep -f "$docker_exec_pattern" >/dev/null 2>&1; then
    pkill -KILL -f "$docker_exec_pattern" >/dev/null 2>&1 || true
    stopped=1
  fi

  if [[ "$stopped" -eq 1 ]]; then
    success "Stopped Zenoh bridge processes."
  else
    warn "No Zenoh server bridge processes were running."
  fi
}

status_all() {
  require_supported_host
  init_paths
  get_docker_cmd

  local public_ip="unknown"
  public_ip=$(get_public_ip 2>/dev/null || true)

  echo "=== Host ==="
  echo "User:            $(current_user)"
  echo "Ubuntu:          $(. /etc/os-release && echo "${PRETTY_NAME}")"
  echo "Architecture:    $(dpkg --print-architecture)"
  echo "Public IP:       ${public_ip:-unknown}"
  echo

  echo "=== GPU ==="
  if have_cmd nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
  else
    echo "nvidia-smi not found"
  fi
  echo

  echo "=== Docker ==="
  if have_cmd docker; then
    docker --version || true
    docker_cmd info --format 'Server: {{.ServerVersion}}' 2>/dev/null || true
  else
    echo "docker not installed"
  fi
  echo

  echo "=== ROS 2 ==="
  if [[ -f "$ROS_SETUP" ]]; then
    echo "Installed:       yes (${ROS_DISTRO})"
    echo "Setup script:    ${ROS_SETUP}"
  else
    echo "Installed:       no"
  fi
  echo

  echo "=== TigerVNC Desktop ==="
  echo "Enabled:         ${TIGERVNC_ENABLE}"
  echo "Display:         :${TIGERVNC_DISPLAY#:}"
  echo "Port:            ${TIGERVNC_PORT}/tcp"
  if have_cmd vncserver; then
    as_current_user vncserver -list || true
  else
    echo "vncserver not installed"
  fi
  echo

  echo "=== Isaac Sim Container ==="
  if have_cmd docker && docker_cmd ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker_cmd ps -a --filter "name=^/${CONTAINER_NAME}$"
  else
    echo "No container named ${CONTAINER_NAME}"
  fi
  echo

  echo "=== Listening Ports ==="
  ss -lntup | egrep ":(${WEBRTC_SIGNAL_PORT}|${WEBRTC_STREAM_PORT}|${TIGERVNC_PORT})\b" || echo "No listeners yet on ${WEBRTC_SIGNAL_PORT}/${WEBRTC_STREAM_PORT}/${TIGERVNC_PORT}"
}

logs_isaacsim() {
  require_docker_runtime
  docker_cmd logs -f "$CONTAINER_NAME"
}

shell_isaacsim() {
  require_docker_runtime
  init_paths
  docker_cmd exec -it -w "$CONTAINER_WORKSPACE" "$CONTAINER_NAME" bash
}

check_connectivity() {
  require_supported_host
  init_paths

  local public_ip
  public_ip=$(get_public_ip)

  echo "=== Server-side checks ==="
  if have_cmd docker && docker_cmd ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container is running: ${CONTAINER_NAME}"
  else
    echo "Container is not running: ${CONTAINER_NAME}"
  fi
  echo
  echo "Listening sockets on this VM:"
  ss -lntup | egrep ":(${WEBRTC_SIGNAL_PORT}|${WEBRTC_STREAM_PORT}|${TIGERVNC_PORT})\b" || echo "No listeners found yet."
  echo

  echo "=== Run these on your PC ==="
  if webrtc_port_listening; then
    echo "TCP signaling test:"
    echo "  nc -vz ${public_ip} ${WEBRTC_SIGNAL_PORT}"
    echo
    echo "UDP media test (nmap is better for UDP):"
    echo "  nmap -sU -Pn -p ${WEBRTC_STREAM_PORT} ${public_ip}"
    echo
    echo "Isaac Sim WebRTC client target:"
    echo "  ${public_ip}"
  else
    echo "WebRTC is not listening on ${WEBRTC_SIGNAL_PORT}/${WEBRTC_STREAM_PORT}."
    echo "  Start with '${SCRIPT_NAME} start isaacsim' when you want the streaming client."
  fi
  if is_truthy "$TIGERVNC_ENABLE" || tigervnc_port_listening; then
    echo
    echo "TigerVNC TCP test:"
    echo "  nc -vz ${public_ip} ${TIGERVNC_PORT}"
    echo
    echo "TigerVNC client target:"
    echo "  ${public_ip}:${TIGERVNC_PORT}"
  fi
}

run_in_isaac_container() {
  local livestream_mode="0"
  local enable_cameras="0"
  local public_ip=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --livestream)
        [[ $# -ge 2 ]] || error "--livestream requires 'public' or 'private'."
        case "$2" in
          public) livestream_mode="1" ;;
          private) livestream_mode="2" ;;
          *) error "--livestream must be 'public' or 'private'." ;;
        esac
        shift 2
        ;;
      --enable-cameras)
        enable_cameras="1"
        shift
        ;;
      --public-ip)
        [[ $# -ge 2 ]] || error "--public-ip requires an IPv4 or hostname value."
        public_ip="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        error "Usage: ${SCRIPT_NAME} run [--livestream public|private] [--enable-cameras] [--public-ip <ip>] -- <command>"
        ;;
    esac
  done

  [[ $# -gt 0 ]] || error "Usage: ${SCRIPT_NAME} run [--livestream public|private] [--enable-cameras] [--public-ip <ip>] -- <command>"
  local entrypoint_cmd="$1"
  shift

  require_supported_host
  init_paths
  verify_nvidia_container_runtime
  ensure_isaac_dirs
  ensure_isaac_runtime_image
  if [[ "$livestream_mode" != "0" ]]; then
    configure_ufw_if_active
  fi
  build_common_docker_args

  if [[ "$livestream_mode" != "0" ]]; then
    DOCKER_RUN_ARGS+=(-e "LIVESTREAM=${livestream_mode}")
    if [[ "$livestream_mode" == "1" ]]; then
      if [[ -z "$public_ip" ]]; then
        public_ip=$(get_public_ip)
      fi
      DOCKER_RUN_ARGS+=(-e "PUBLIC_IP=${public_ip}")
    elif [[ -n "$public_ip" ]]; then
      DOCKER_RUN_ARGS+=(-e "PUBLIC_IP=${public_ip}")
    fi
  fi

  if [[ "$enable_cameras" == "1" ]]; then
    DOCKER_RUN_ARGS+=(-e "ENABLE_CAMERAS=1")
  fi

  info "Running one-shot command in ${ISAAC_RUNTIME_IMAGE}..."
  info "Workspace: ${HOST_WORKSPACE_ROOT} -> ${CONTAINER_WORKSPACE}"
  if [[ "$livestream_mode" == "1" ]]; then
    info "One-shot WebRTC mode: public"
    info "Connect the Isaac Sim WebRTC client to: ${public_ip}"
    info "Required ports on the host: ${WEBRTC_SIGNAL_PORT}/tcp and ${WEBRTC_STREAM_PORT}/udp"
    info "If this IP is wrong for your cloud provider or VPN setup, rerun with --public-ip <reachable-ip>."
  elif [[ "$livestream_mode" == "2" ]]; then
    info "One-shot WebRTC mode: private/local"
    if [[ -n "$public_ip" ]]; then
      info "Configured reachable client target: ${public_ip}"
    else
      info "Use the host IP reachable from your laptop in the Isaac Sim WebRTC client."
    fi
    info "Required ports on the host: ${WEBRTC_SIGNAL_PORT}/tcp and ${WEBRTC_STREAM_PORT}/udp"
  fi
  if [[ "$enable_cameras" == "1" ]]; then
    info "ENABLE_CAMERAS=1 will be set inside the container."
  fi
  local run_status=0
  if run_docker_attached "isaac-run" \
    "${DOCKER_RUN_ARGS[@]}" \
    --entrypoint "$entrypoint_cmd" \
    "$ISAAC_RUNTIME_IMAGE" \
    "$@"; then
    run_status=0
  else
    run_status=$?
  fi
  return "$run_status"
}

prepare_isaacsim_livestream_wrapper() {
  local wrapper_file
  wrapper_file=$(mktemp "${TMPDIR:-/tmp}/isaac-vmctl-stream-wrapper.XXXXXX.py")
  cat >"$wrapper_file" <<'EOF_STREAM_WRAPPER'
import os
import platform
import runpy
import sys

if platform.machine().lower() in ["aarch64", "arm64"]:
    print("Livestream is not supported on ARM64 architecture. Exiting.", flush=True)
    sys.exit(0)

from isaacsim import SimulationApp as _BaseSimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,
    "renderer": "RaytracedLighting",
    "display_options": 3286,
}

app = _BaseSimulationApp(launch_config=CONFIG)

from isaacsim.core.utils.extensions import enable_extension
import isaacsim

signal_port = int(os.environ.get("ISAAC_VMCTL_STREAM_SIGNAL_PORT", "49100"))
stream_port = int(os.environ.get("ISAAC_VMCTL_STREAM_PORT", "47998"))
public_ip = os.environ.get("ISAAC_VMCTL_PUBLIC_IP", "")

app.set_setting("/app/window/drawMouse", True)
app.set_setting("/app/livestream/port", signal_port)
if public_ip:
    app.set_setting("/app/livestream/publicEndpointAddress", public_ip)
app.set_setting("/exts/omni.kit.livestream.app/primaryStream/signalPort", signal_port)
app.set_setting("/exts/omni.kit.livestream.app/primaryStream/streamPort", stream_port)
if public_ip:
    app.set_setting("/exts/omni.kit.livestream.app/primaryStream/publicIp", public_ip)

enable_extension("omni.services.livestream.nvcf")


def _existing_app(*_args, **_kwargs):
    return app


isaacsim.SimulationApp = _existing_app

target_script = sys.argv[1]
sys.argv = [target_script, *sys.argv[2:]]

try:
    runpy.run_path(target_script, run_name="__main__")
finally:
    try:
        app.close()
    except Exception:
        pass
EOF_STREAM_WRAPPER
  chmod 0644 "$wrapper_file"
  printf '%s\n' "$wrapper_file"
}

run_isaacsim_python() {
  local mode="plain"
  local livestream_mode="0"
  local public_ip=""
  local webrtc_signal_port="$WEBRTC_SIGNAL_PORT"
  local webrtc_stream_port="$WEBRTC_STREAM_PORT"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gui|--vnc)
        [[ "$livestream_mode" == "0" ]] || error "Use either --gui or --livestream with 'run --py', not both."
        mode="gui"
        shift
        ;;
      --headless)
        [[ "$livestream_mode" == "0" ]] || error "Use either --headless or --livestream with 'run --py', not both."
        mode="plain"
        shift
        ;;
      --webrtc|--streaming)
        [[ "$mode" != "gui" ]] || error "Use either --gui or --livestream with 'run --py', not both."
        mode="webrtc"
        livestream_mode="1"
        shift
        ;;
      --livestream)
        [[ $# -ge 2 ]] || error "--livestream requires 'public' or 'private'."
        [[ "$mode" != "gui" ]] || error "Use either --gui or --livestream with 'run --py', not both."
        mode="webrtc"
        case "$2" in
          public) livestream_mode="1" ;;
          private) livestream_mode="2" ;;
          *) error "--livestream must be 'public' or 'private'." ;;
        esac
        shift 2
        ;;
      --public-ip)
        [[ $# -ge 2 ]] || error "--public-ip requires an IPv4 or hostname value."
        public_ip="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        break
        ;;
    esac
  done

  [[ $# -gt 0 ]] || error "Usage: ${SCRIPT_NAME} run --py [--gui|--livestream public|private [--public-ip <ip>]] '<python.sh args>'"

  local -a python_args=()
  if [[ $# -eq 1 ]]; then
    mapfile -t python_args < <(parse_shell_words "$1")
  else
    python_args=("$@")
  fi
  (( ${#python_args[@]} > 0 )) || error "Isaac Sim python.sh arguments resolved to an empty command."

  require_supported_host
  init_paths
  verify_nvidia_container_runtime
  ensure_isaac_dirs
  ensure_isaac_runtime_image
  local isaac_tag
  isaac_tag="${ISAAC_IMAGE##*:}"
  if [[ "$livestream_mode" != "0" ]]; then
    if [[ "$isaac_tag" == 5.* ]]; then
      if [[ "$WEBRTC_SIGNAL_PORT" != "49100" || "$WEBRTC_STREAM_PORT" != "47998" ]]; then
        warn "Isaac Sim ${isaac_tag} standalone python WebRTC uses the fixed default ports 49100/tcp and 47998/udp. Ignoring WEBRTC_SIGNAL_PORT=${WEBRTC_SIGNAL_PORT} and WEBRTC_STREAM_PORT=${WEBRTC_STREAM_PORT} for 'run --py --livestream'."
      fi
      webrtc_signal_port=49100
      webrtc_stream_port=47998
    fi
    configure_ufw_if_active
    if [[ "$livestream_mode" == "1" && -z "$public_ip" ]]; then
      public_ip=$(get_public_ip)
    fi
  elif [[ -n "$public_ip" ]]; then
    error "--public-ip only applies with 'run --py --livestream ...'."
  fi

  build_common_docker_args
  DOCKER_RUN_ARGS+=(-e "TERM=xterm")
  if [[ "$mode" == "gui" ]]; then
    prepare_gui_x11_access
    append_gui_docker_args
  fi

  info "Running Isaac Sim python.sh command from ${ISAAC_RUNTIME_IMAGE} (${mode})..."
  info "Workspace: ${HOST_WORKSPACE_ROOT} -> ${CONTAINER_WORKSPACE}"
  if [[ "$mode" == "gui" ]]; then
    info "Isaac Sim GUI will open on ${GUI_DISPLAY}. Run this from the terminal inside TigerVNC when you want the viewport there."
  elif [[ "$livestream_mode" == "1" ]]; then
    info "Isaac Sim WebRTC client target: ${public_ip}"
    info "Required ports on the host: ${webrtc_signal_port}/tcp and ${webrtc_stream_port}/udp"
    info "This path is meant for standard Isaac Sim standalone examples."
  elif [[ "$livestream_mode" == "2" ]]; then
    if [[ -n "$public_ip" ]]; then
      info "Configured reachable WebRTC client target: ${public_ip}"
    else
      info "Use the host IP reachable from your laptop in the Isaac Sim WebRTC client."
    fi
    info "Required ports on the host: ${webrtc_signal_port}/tcp and ${webrtc_stream_port}/udp"
    info "This path is meant for standard Isaac Sim standalone examples."
  else
    info "This runs the plain python.sh command without attaching to the current X display."
  fi

  local wrapper_file=""
  local wrapper_container_file="/tmp/isaac-vmctl-stream-wrapper.py"
  local -a python_launch_args=("${python_args[@]}")
  if [[ "$livestream_mode" != "0" ]]; then
    [[ "${python_args[0]}" != -* ]] || error "'run --py --livestream' currently supports script-path invocations, not python options like '${python_args[0]}'."
    wrapper_file=$(prepare_isaacsim_livestream_wrapper)
    DOCKER_RUN_ARGS+=(
      -e "ISAAC_VMCTL_STREAM_SIGNAL_PORT=${webrtc_signal_port}"
      -e "ISAAC_VMCTL_STREAM_PORT=${webrtc_stream_port}"
      -v "${wrapper_file}:${wrapper_container_file}:ro"
    )
    if [[ -n "$public_ip" ]]; then
      DOCKER_RUN_ARGS+=(-e "ISAAC_VMCTL_PUBLIC_IP=${public_ip}")
    fi
    python_launch_args=("${wrapper_container_file}" "${python_args[@]}")
  fi
  local python_launch_cmd=""
  printf -v python_launch_cmd '%q ' "${python_launch_args[@]}"
  python_launch_cmd=${python_launch_cmd% }

  local run_status=0
  if run_docker_attached "isaacpy-run" \
    "${DOCKER_RUN_ARGS[@]}" \
    --entrypoint bash \
    "$ISAAC_RUNTIME_IMAGE" \
    -c "cd /isaac-sim && ./python.sh ${python_launch_cmd}"; then
    run_status=0
  else
    run_status=$?
  fi
  if [[ -n "$wrapper_file" ]]; then
    rm -f "$wrapper_file"
  fi
  return "$run_status"
}

run_isaaclab() {
  local custom_path=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --path)
        [[ $# -ge 2 ]] || error "Usage: ${SCRIPT_NAME} run isaaclab [--path <workspace-subdir>] '<isaaclab.sh args>'"
        custom_path="$2"
        shift 2
        ;;
      *)
        break
        ;;
    esac
  done

  [[ $# -gt 0 ]] || error "Usage: ${SCRIPT_NAME} run isaaclab [--path <workspace-subdir>] '<isaaclab.sh args>'"

  local -a isaaclab_args=()
  if [[ $# -eq 1 ]]; then
    mapfile -t isaaclab_args < <(parse_shell_words "$1")
  else
    isaaclab_args=("$@")
  fi
  (( ${#isaaclab_args[@]} > 0 )) || error "Isaac Lab arguments resolved to an empty command."
  if isaaclab_args_request_install "${isaaclab_args[@]}"; then
    error "Isaac Lab installation is bootstrap-managed in this repo. Set ISAACLAB_REF / ISAACLAB_FRAMEWORKS in your env and rerun '${SCRIPT_NAME} bootstrap' instead of running 'isaaclab.sh --install' manually."
  fi

  require_supported_host
  init_paths
  verify_nvidia_container_runtime
  ensure_isaac_dirs

  local isaaclab_path="${custom_path:-$ISAACLAB_PATH}"
  local saved_isaaclab_path="$ISAACLAB_PATH"
  ISAACLAB_PATH="$isaaclab_path"
  ensure_isaaclab_image

  build_common_docker_args
  DOCKER_RUN_ARGS+=(-e "TERM=xterm")
  local mode="headless"
  if isaaclab_args_need_gui "${isaaclab_args[@]}"; then
    mode="gui"
    prepare_gui_x11_access
    append_gui_docker_args
  fi
  DOCKER_RUN_ARGS+=(--entrypoint bash)

  local host_isaaclab_dir="${HOST_WORKSPACE_ROOT}/${ISAACLAB_PATH}"
  local container_isaaclab_dir="${CONTAINER_WORKSPACE}/${ISAACLAB_PATH}"
  local isaaclab_image
  isaaclab_image=$(resolve_isaaclab_image)

  info "Running Isaac Lab from ${host_isaaclab_dir} (${mode})..."
  info "Isaac Lab image: ${isaaclab_image}"
  info "Workspace: ${HOST_WORKSPACE_ROOT} -> ${CONTAINER_WORKSPACE}"
  if [[ "$mode" == "gui" ]]; then
    info "Isaac Lab GUI will open on ${GUI_DISPLAY}. Run this from the terminal inside TigerVNC when you want the viewport there."
  else
    info "Isaac Lab command will run without attaching to the current X display."
  fi

  local run_status=0
  if run_docker_attached "isaaclab-run" \
    "${DOCKER_RUN_ARGS[@]}" \
    "$isaaclab_image" \
    -c 'cd "$1" && shift && exec ./isaaclab.sh "$@"' _ "$container_isaaclab_dir" "${isaaclab_args[@]}"; then
    run_status=0
  else
    run_status=$?
  fi
  ISAACLAB_PATH="$saved_isaaclab_path"
  return "$run_status"
}

vast_driver_compatible() {
  local versions version
  have_cmd nvidia-smi || return 1
  versions=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null) || return 1
  [[ -n "$versions" ]] || return 1
  while IFS= read -r version; do
    version="${version//[[:space:]]/}"
    dpkg --compare-versions "$version" ge "$NVIDIA_MIN_DRIVER_VERSION" || return 1
  done <<< "$versions"
}

vast_kernel_supports_firmware() {
  local config="/boot/config-$(uname -r)"
  [[ -r "$config" ]] || error "Cannot inspect ${config}; have the administrator verify CONFIG_FW_LOADER before driver repair."
  grep -Eq '^CONFIG_FW_LOADER=(y|m)$' "$config"
}

install_vast_environment_repair() {
  local target=/usr/local/lib/isaac-projects/cloud_host.py
  local helper unit
  cloud_python_as_root environment-repair
  helper=$(mktemp)
  cloud_python_source >"$helper"
  as_root install -d -m 0755 /usr/local/lib/isaac-projects
  as_root install -m 0755 "$helper" "$target"
  rm -f "$helper"
  unit=$(mktemp)
  cat >"$unit" <<'EOF_VAST_ENV'
[Unit]
Description=Quote Vast public-key environment entries before package updates
After=local-fs.target
Before=apt-daily.service apt-daily-upgrade.service

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /usr/local/lib/isaac-projects/cloud_host.py environment-repair
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF_VAST_ENV
  as_root install -m 0644 "$unit" /etc/systemd/system/isaac-projects-vast-environment.service
  rm -f "$unit"
  as_root systemctl daemon-reload
  as_root systemctl enable isaac-projects-vast-environment.service
}

ensure_vast_vm_stays_awake() {
  # A GNOME idle timer can suspend the guest while Vast still reports its
  # outer VM container as running. This cloud server must remain reachable.
  local policy
  policy=$(mktemp)
  cat >"$policy" <<'EOF_VAST_SLEEP'
[Sleep]
AllowSuspend=no
AllowHibernation=no
AllowSuspendThenHibernate=no
AllowHybridSleep=no
EOF_VAST_SLEEP
  as_root install -d -m 0755 /etc/systemd/sleep.conf.d
  as_root install -m 0644 "$policy" /etc/systemd/sleep.conf.d/99-isaac-projects-server.conf
  rm -f "$policy"
  as_root systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
  info "Cloud VM automatic sleep and hibernation are disabled."
}

verify_vast_vm_stays_awake() {
  local target
  for target in sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target; do
    [[ "$(systemctl is-enabled "$target" 2>/dev/null || true)" == masked ]] || \
      error "Cloud VM sleep protection is missing (${target}); rerun bootstrap to prevent idle SSH outages."
  done
  grep -Fxq 'AllowSuspend=no' /etc/systemd/sleep.conf.d/99-isaac-projects-server.conf || \
    error "Cloud VM sleep policy is missing; rerun bootstrap."
}

vast_reboot_required() {
  warn "Vast host preparation requires a reboot (exit 42). No reboot is performed automatically."
  warn "Ask the server administrator to reboot this VM, reconnect, rerun bootstrap with the same settings."
  exit 42
}

prepare_vast_vm() {
  [[ "${ID:-}" == ubuntu && "${VERSION_ID:-}" == 22.04 && "${ARCH:-}" == amd64 ]] || \
    error "VAST_VM_PREPARE supports Ubuntu 22.04 amd64 full VMs only."
  if systemd-detect-virt --container --quiet; then
    error "This is a container rental. Driver/kernel installation requires a full VM; ask the provider for a prepared GPU host."
  fi
  systemd-detect-virt --vm --quiet || error "VAST_VM_PREPARE is for a new full VM, not a workstation."
  install_vast_environment_repair
  ensure_vast_vm_stays_awake

  local state_dir="${VAST_HOST_STATE_DIR:-/var/lib/isaac-projects}"
  local marker="${state_dir}/vast-host-reboot-required"
  local boot_id
  boot_id=$(cat /proc/sys/kernel/random/boot_id)
  if [[ -r "$marker" && "$(cat "$marker")" == "$boot_id" ]]; then
    vast_reboot_required
  fi
  if vast_driver_compatible && vast_kernel_supports_firmware; then
    info "NVIDIA driver/kernel already meet the configured minimum; preserving them."
    as_root rm -f "$marker"
    return 0
  fi
  [[ ! -e "$marker" ]] || error "Driver/kernel still unhealthy after reboot. Check the booted kernel and nvidia-smi before retrying; no repeated package repair attempted."

  # Changing a driver on a shared/busy machine can interrupt graphics and
  # compute. Inspect current workloads, not historical allocation records.
  cloud_python_as_root gpu-idle
  if have_cmd docker && as_root systemctl is-active --quiet docker; then
    assert_local_docker_idle
  fi
  local driver_repair=0
  if ! vast_driver_compatible; then driver_repair=1; fi
  local -a old_packages=()
  local pkg
  while IFS= read -r pkg; do
    if [[ "$driver_repair" -eq 1 && "$pkg" =~ ^nvidia-(driver|dkms|kernel-source|kernel-common)-[0-9]+(-open)?$ &&
          ! "$pkg" =~ -580(-open)?$ ]] && package_installed "$pkg"; then
      old_packages+=("$pkg")
    fi
  done < <(dpkg-query -W -f='${binary:Package}\n' 'nvidia-*' 2>/dev/null || true)
  apt_get update -y
  if [[ ${#old_packages[@]} -gt 0 ]]; then
    apt_get remove -y "${old_packages[@]}"
  fi
  if ! vast_kernel_supports_firmware; then
    apt_get install -y --no-install-recommends linux-generic-hwe-22.04
    as_root update-grub
  fi
  if [[ "$driver_repair" -eq 1 ]]; then
    apt_get install -y --no-install-recommends nvidia-driver-580-open vulkan-tools
  fi
  as_root install -d -m 0755 "$state_dir"
  printf '%s\n' "$boot_id" | as_root tee "$marker" >/dev/null
  vast_reboot_required
}

prepare_host_context() {
  require_supported_host
  init_paths
  if is_truthy "$VAST_VM_PREPARE"; then
    prepare_vast_vm
  fi
}

doctor() {
  # Deliberately avoid prepare_host_context: doctor must never repair/reboot.
  require_supported_host
  init_paths
  if is_truthy "$VAST_VM_PREPARE"; then
    verify_vast_vm_stays_awake
  fi
  ensure_nvidia_driver_present
  require_docker_runtime
  local audit runtime_image
  audit=$(dpkg --audit)
  [[ -z "$audit" ]] || error "Unfinished package configuration found; have the administrator complete OS updates before bootstrap."
  [[ -f "$ROS_SETUP" ]] || error "ROS setup is missing: ${ROS_SETUP}"
  docker_cmd image inspect "$ISAAC_IMAGE" >/dev/null
  runtime_image=$(resolve_isaac_runtime_image)
  docker_cmd image inspect "$runtime_image" >/dev/null
  if is_truthy "$ISAACLAB_ENABLE"; then
    [[ -f "${HOST_WORKSPACE_ROOT}/${ISAACLAB_PATH}/isaaclab.sh" ]] || error "Isaac Lab checkout is missing; rerun bootstrap."
    docker_cmd image inspect "$(resolve_isaaclab_image)" >/dev/null || error "Isaac Lab image is missing; rerun bootstrap."
  fi
  docker_cmd run --rm --pull=never --network=none --gpus all --entrypoint nvidia-smi "$runtime_image" >/dev/null
  if is_truthy "$VSCODE_REMOTE_ENABLE"; then
    [[ -x "${USER_HOME}/.local/bin/code" ]] || have_cmd code || error "VS Code CLI is missing."
  fi
  if is_truthy "$JUPYTER_ENABLE"; then
    [[ -x "${USER_HOME}/.local/bin/jupyter-lab" ]] || error "JupyterLab wrapper is missing."
  fi
  if is_truthy "$TIGERVNC_ENABLE"; then
    validate_tigervnc_config
    verify_tigervnc_ready
  fi
  success "Bootstrap readiness checks passed. This does not launch or qualify an Isaac Sim scene."
}

begin_install_session() {
  local label="$1"
  prepare_log_file "$label"
  info "Installer output is summarized in the terminal."
  show_install_log_hint "$label"
}

print_install_success_summary() {
  local label="$1"
  success "${label} completed successfully."
  info "Detailed log: ${ACTIVE_LOG_FILE}"
  if [[ "$DOCKER_GROUP_RELOGIN_REQUIRED" -eq 1 ]]; then
    warn "Docker group membership changed during install. Open a new shell before running plain 'docker' without sudo."
  fi
}

zenoh_setup() {
  local setup_script="${SCRIPT_DIR}/zenoh/setup.sh"
  [[ -f "$setup_script" ]] || error "Zenoh setup script not found at ${setup_script}"
  exec bash "$setup_script" "$@"
}

zenoh_container_bridge_available() {
  have_cmd docker || return 1
  get_docker_cmd
  "${DOCKER[@]}" ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER_NAME" || return 1
  "${DOCKER[@]}" exec "$CONTAINER_NAME" test -f "${CONTAINER_WORKSPACE}/zenoh/start_zenoh_bridge.sh" >/dev/null 2>&1
}

zenoh_bridge_port_from_args() {
  local port="7447"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --domain|--namespace|--config)
        shift 2
        ;;
      -*)
        shift
        ;;
      *)
        port="$1"
        break
        ;;
    esac
  done
  printf '%s\n' "$port"
}

zenoh_report_existing_listener() {
  local port="$1"
  local listener pid command_line

  listener=$(ss -H -lntp "sport = :${port}" 2>/dev/null || true)
  [[ -n "$listener" ]] || return 1

  if [[ "$listener" != *zenoh-bridge* ]]; then
    error "TCP port ${port} is already in use by another process: ${listener}"
  fi

  pid=$(printf '%s\n' "$listener" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)
  command_line=""
  if [[ -n "$pid" ]]; then
    command_line=$(ps -p "$pid" -o args= 2>/dev/null || true)
  fi

  info "Zenoh bridge is already listening on tcp/0.0.0.0:${port}${pid:+ (pid ${pid})}."
  if [[ "$command_line" == *"${CONTAINER_WORKSPACE}/zenoh/zenoh-bridge-ros2dds"* ]]; then
    info "Existing bridge is running from the Isaac container workspace."
  elif [[ -n "$command_line" ]]; then
    warn "Existing bridge command: ${command_line}"
  fi
  info "Local laptops connect to: tcp/<GPU_PUBLIC_IP>:${port}"
  return 0
}

zenoh_start_bridge() {
  local start_script="${SCRIPT_DIR}/zenoh/start_zenoh_bridge.sh"
  local has_domain_arg=0
  local domain_for_bridge="${ROS_DOMAIN_ID}"
  local requested_port
  local idx
  local -a bridge_args=("$@")
  [[ -f "$start_script" ]] || error "Zenoh start script not found at ${start_script}"
  for idx in "${!bridge_args[@]}"; do
    case "${bridge_args[$idx]}" in
      -h|--help) exec bash "$start_script" "${bridge_args[@]}" ;;
    esac
  done
  case "$ZENOH_BRIDGE_CONTEXT" in
    auto|isaac|host) ;;
    *) error "ZENOH_BRIDGE_CONTEXT must be one of: auto, isaac, host. Found: ${ZENOH_BRIDGE_CONTEXT}" ;;
  esac
  init_paths
  requested_port=$(zenoh_bridge_port_from_args "${bridge_args[@]}")

  for idx in "${!bridge_args[@]}"; do
    if [[ "${bridge_args[$idx]}" == "--domain" ]]; then
      has_domain_arg=1
      [[ $((idx + 1)) -lt ${#bridge_args[@]} ]] || error "--domain requires a value"
      domain_for_bridge="${bridge_args[$((idx + 1))]}"
      break
    fi
  done
  if [[ "$has_domain_arg" -eq 0 ]]; then
    validate_ros_domain_id
    domain_for_bridge="$ROS_DOMAIN_ID"
    bridge_args+=(--domain "$domain_for_bridge")
  else
    ROS_DOMAIN_ID="$domain_for_bridge"
    validate_ros_domain_id
    domain_for_bridge="$ROS_DOMAIN_ID"
  fi

  if zenoh_report_existing_listener "$requested_port"; then
    return 0
  fi

  if [[ "$ZENOH_BRIDGE_CONTEXT" != "host" ]]; then
    if zenoh_container_bridge_available; then
      local -a docker_exec_args=(
        exec -i
        -w "$CONTAINER_WORKSPACE"
        -e "ROS_DOMAIN_ID=${domain_for_bridge}"
      )
      if [[ -n "$RMW_IMPLEMENTATION" ]]; then
        docker_exec_args+=(-e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}")
      fi
      docker_exec_args+=(
        "$CONTAINER_NAME"
        bash -lc 'if [[ -r /etc/profile.d/isaac-projects-ros2.sh ]]; then source /etc/profile.d/isaac-projects-ros2.sh; fi; exec bash "$@"'
        _
        ./zenoh/start_zenoh_bridge.sh
        "${bridge_args[@]}"
      )
      info "Starting Zenoh bridge inside ${CONTAINER_NAME} so it uses the Isaac container ROS runtime."
      exec "${DOCKER[@]}" "${docker_exec_args[@]}"
    fi

    if [[ "$ZENOH_BRIDGE_CONTEXT" == "isaac" ]]; then
      error "ZENOH_BRIDGE_CONTEXT=isaac requires a running ${CONTAINER_NAME} container with ${CONTAINER_WORKSPACE} mounted."
    fi

    warn "Isaac container bridge context was not available; starting Zenoh with the host ROS runtime."
    warn "For Isaac Sim topics on mixed Ubuntu 22/Humble host and Jazzy Isaac containers, start Isaac Sim first or set ZENOH_BRIDGE_CONTEXT=isaac."
  fi

  exec bash "$start_script" "${bridge_args[@]}"
}

install_docker_stack() {
  begin_install_session "install-docker"
  run_install_step 2 1 "Validate host and workspace settings" prepare_host_context
  run_install_step 2 2 "Install or verify Docker and NVIDIA Container Toolkit" ensure_nvidia_container_runtime
  print_install_success_summary "Docker and NVIDIA runtime setup"
}

install_ros2_stack() {
  begin_install_session "install-ros2"
  run_install_step 2 1 "Validate host and workspace settings" prepare_host_context
  run_install_step 2 2 "Install or verify ROS 2" ensure_ros2_installed
  print_install_success_summary "ROS 2 setup"
}

install_tigervnc_stack() {
  begin_install_session "install-tigervnc"
  run_install_step 2 1 "Validate host and workspace settings" prepare_host_context
  run_install_step 2 2 "Install and start TigerVNC desktop" ensure_tigervnc_ready
  print_install_success_summary "TigerVNC desktop setup"
  print_tigervnc_connection_summary
}

install_all() {
  local total_steps=6
  if is_truthy "$ISAACLAB_ENABLE"; then
    total_steps=7
  fi
  if is_truthy "$TIGERVNC_ENABLE"; then
    total_steps=$((total_steps + 1))
  fi

  begin_install_session "bootstrap"
  run_install_step "$total_steps" 1 "Validate host and workspace settings" prepare_host_context
  run_install_step "$total_steps" 2 "Install or verify Docker and NVIDIA Container Toolkit" ensure_nvidia_container_runtime
  run_install_step "$total_steps" 3 "Install or verify ROS 2" ensure_ros2_installed
  run_install_step "$total_steps" 4 "Install or verify default student tooling" ensure_student_tooling
  run_install_step "$total_steps" 5 "Prepare Isaac Sim cache and data directories" ensure_isaac_dirs
  run_install_step "$total_steps" 6 "Pull or verify the Isaac Sim runtime image" ensure_isaac_runtime_image
  if is_truthy "$ISAACLAB_ENABLE"; then
    run_install_step "$total_steps" 7 "Clone/update Isaac Lab and build the managed Isaac Lab image" ensure_isaaclab_image
  fi
  if is_truthy "$TIGERVNC_ENABLE"; then
    run_install_step "$total_steps" "$total_steps" "Install and start TigerVNC desktop" ensure_tigervnc_ready
  fi
  print_install_success_summary "Bootstrap"
  if is_truthy "$TIGERVNC_ENABLE"; then
    print_tigervnc_connection_summary
  fi
}

main() {
  setup_colors
  parse_leading_common_flags "$@"
  set -- "${PARSED_ARGS[@]}"
  INVOCATION_STRING="${SCRIPT_NAME}"
  if [[ $# -gt 0 ]]; then
    INVOCATION_STRING+=" $*"
  fi

  local cmd=${1:-help}
  local sub=${2:-}

  case "$cmd" in
    start)
      case "$sub" in
        isaacsim)
          shift 2
          start_isaacsim "$@"
          ;;
        tigervnc|vnc)
          shift 2
          [[ $# -eq 0 ]] || error "Usage: ${SCRIPT_NAME} start {tigervnc|vnc}"
          ensure_tigervnc_ready
          print_tigervnc_connection_summary
          ;;
        zenoh|bridge)
          shift 2
          zenoh_start_bridge "$@"
          ;;
        *)
          error "Usage: ${SCRIPT_NAME} start {isaacsim|tigervnc|vnc|zenoh|bridge}"
          ;;
      esac
      ;;
    stop)
      case "$sub" in
        isaacsim)
          stop_isaacsim
          ;;
        tigervnc|vnc)
          stop_tigervnc_server
          ;;
        zenoh|bridge)
          shift 2
          [[ $# -eq 0 ]] || error "Usage: ${SCRIPT_NAME} stop {zenoh|bridge}"
          stop_zenoh_bridge
          ;;
        *)
          error "Usage: ${SCRIPT_NAME} stop {isaacsim|tigervnc|vnc|zenoh|bridge}"
          ;;
      esac
      ;;
    restart)
      [[ "$sub" == "isaacsim" ]] || error "Usage: ${SCRIPT_NAME} restart isaacsim"
      shift 2
      stop_isaacsim || true
      start_isaacsim "$@"
      ;;
    run)
      shift
      case "${1:-}" in
        --py)
          shift
          run_isaacsim_python "$@"
          ;;
        isaaclab)
          shift
          run_isaaclab "$@"
          ;;
        *)
          run_in_isaac_container "$@"
          ;;
      esac
      ;;
    bootstrap)
      case "$sub" in
        zenoh|bridge)
          shift 2
          zenoh_setup "$@"
          ;;
        *)
          shift
          parse_common_flags "$@"
          [[ ${#PARSED_ARGS[@]} -eq 0 ]] || error "Usage: ${SCRIPT_NAME} bootstrap [--no-vast-vm] [--verbose] [--log-file <path>] | ${SCRIPT_NAME} bootstrap {zenoh|bridge} [--force]"
          install_all
          ;;
      esac
      ;;
    status)
      status_all
      ;;
    doctor)
      shift
      parse_common_flags "$@"
      [[ ${#PARSED_ARGS[@]} -eq 0 ]] || error "Usage: ${SCRIPT_NAME} doctor [--no-vast-vm]"
      doctor
      ;;
    logs)
      logs_isaacsim
      ;;
    shell)
      shell_isaacsim
      ;;
    check)
      check_connectivity
      ;;
    install)
      case "$sub" in
        all)
          shift 2
          parse_common_flags "$@"
          [[ ${#PARSED_ARGS[@]} -eq 0 ]] || error "Usage: ${SCRIPT_NAME} install all [--verbose] [--log-file <path>]"
          install_all
          ;;
        ros2)
          shift 2
          parse_common_flags "$@"
          [[ ${#PARSED_ARGS[@]} -eq 0 ]] || error "Usage: ${SCRIPT_NAME} install ros2 [--verbose] [--log-file <path>]"
          install_ros2_stack
          ;;
        docker)
          shift 2
          parse_common_flags "$@"
          [[ ${#PARSED_ARGS[@]} -eq 0 ]] || error "Usage: ${SCRIPT_NAME} install docker [--verbose] [--log-file <path>]"
          install_docker_stack
          ;;
        zenoh|bridge)
          shift 2
          zenoh_setup "$@"
          ;;
        tigervnc|vnc)
          shift 2
          parse_common_flags "$@"
          [[ ${#PARSED_ARGS[@]} -eq 0 ]] || error "Usage: ${SCRIPT_NAME} install {tigervnc|vnc} [--verbose] [--log-file <path>]"
          install_tigervnc_stack
          ;;
        *) error "Usage: ${SCRIPT_NAME} install {all|ros2|docker|zenoh|bridge|tigervnc|vnc}" ;;
      esac
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      error "Unknown command: ${cmd}. Run '${SCRIPT_NAME} help'."
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  trap 'installer_exit "$?"' EXIT
  main "$@"
fi
