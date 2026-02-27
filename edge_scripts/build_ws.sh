#!/bin/bash
set -e

# ==========================================
# 📋 Usage
# ==========================================
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -w, --workspace DIR         Specify the ROS 2 workspace directory (default: edge_ws*)
    -h, --help                  Show this help message

Example:
    $0                             # Auto-detect workspace
    $0 -w /path/to/edge_ws         # Specify workspace manually
    $0 --workspace ~/edge_ws_dev   # Using long option
EOF
    exit 0
}

# ==========================================
# 🔍 Auto-detect workspace (default: edge_ws*)  
# ==========================================
find_workspace_root() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        local base=$(basename "$dir")
        if [[ $base == edge_ws* ]]; then
            echo "$dir"
            return 0
        fi
        dir=$(dirname "$dir")
    done
    return 1
}

# ==========================================
# 🎯 Parse command line arguments
# ==========================================
WORKSPACE_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "❌ Unknown option: $1"
            usage
            ;;
    esac
done

# ==========================================
# 🔍 Determine workspace directory
# ==========================================
if [[ -n "$WORKSPACE_DIR" ]]; then
    # User specified workspace
    if [[ ! -d "$WORKSPACE_DIR" ]]; then
        echo "❌ Error: Specified workspace directory does not exist: $WORKSPACE_DIR"
        exit 1
    fi
    
    # Verify it's a edge_ws* directory
    base=$(basename "$WORKSPACE_DIR")
    if [[ $base != edge_ws* ]]; then
        echo "⚠️  Warning: Directory name '$base' doesn't match pattern 'edge_ws*'"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    echo "✅ Using specified workspace: $WORKSPACE_DIR"
else
    # Auto-detect workspace
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    WORKSPACE_DIR=$(find_workspace_root "$SCRIPT_PATH")
    
    if [[ -z "$WORKSPACE_DIR" ]]; then
        echo "❌ Error: Could not find any directory named 'edge_ws*' above $SCRIPT_PATH"
        echo "💡 Tip: Use -w option to specify workspace manually"
        exit 1
    fi
    
    echo "✅ ROS 2 workspace auto-detected: $WORKSPACE_DIR"
fi

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

echo "🔹 [1/6] Building edge_bringup..."
colcon build --packages-up-to edge_bringup ${BUILD_ARGS}

echo "🔹 [2/6] Building edge_dummy_env..."
colcon build --packages-up-to edge_dummy_env ${BUILD_ARGS}

echo "🔹 [3/6] Building edge_quadruped_controller..."
colcon build --packages-up-to edge_quadruped_controller ${BUILD_ARGS}

echo "🔹 [4/6] Building edge_interfaces..."
colcon build --base-paths src/edge-brain-go2/edge_interfaces ${BUILD_ARGS}

echo "🔹 [5/6] Building edge_navigation..."
colcon build --base-paths src/edge-brain-go2/edge_navigation ${BUILD_ARGS} 

echo "🔹 [6/6] Building edge_perception..."
colcon build --base-paths src/edge-brain-go2/edge_perception ${BUILD_ARGS}

echo "------------------------------------------------------------"
echo "✅ All builds completed successfully in: $WORKSPACE_DIR"
echo "------------------------------------------------------------"
