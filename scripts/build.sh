#!/bin/bash
set -e

# ==========================================
# 🔍 Auto-detect ros2_ws*
# ==========================================
find_ros2_ws_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        local base=$(basename "$dir")
        if [[ $base == ros2_ws* ]]; then
            echo "$dir"
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

# Current script directory
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR=$(find_ros2_ws_root "$SCRIPT_PATH")

if [[ -z "$WORKSPACE_DIR" ]]; then
    echo "❌ Error: Could not find any directory named 'ros2_ws*' above $SCRIPT_PATH"
    exit 1
fi

echo "✅ ROS 2 workspace detected: $WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# ==========================================
# ⚙️ Args for colcon build
# ==========================================
BUILD_ARGS="--symlink-install"

# ==========================================
# 🏗️ Start sequential builds
# ==========================================
echo "⚙️ Starting builds under workspace: $WORKSPACE_DIR"
echo "------------------------------------------------------------"

echo "🔹 [1/4] Building edge_bringup..."
colcon build --packages-up-to edge_bringup ${BUILD_ARGS}

echo "🔹 [2/4] Building edge_quadruped_controller..."
colcon build --packages-up-to edge_quadruped_controller ${BUILD_ARGS}

echo "🔹 [3/4] Building edge_interfaces..."
colcon build --base-paths src/edge-brain-go2/edge_common/edge_interfaces ${BUILD_ARGS}

echo "🔹 [4/4] Building edge_navigation..."
colcon build --base-paths src/edge-brain-go2/edge_navigation ${BUILD_ARGS}

echo "------------------------------------------------------------"
echo "✅ All builds completed successfully in: $WORKSPACE_DIR"
echo "------------------------------------------------------------"
