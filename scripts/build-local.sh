#!/bin/bash
# Build Docker images locally for development/testing
# Usage: ./scripts/build-local.sh [backend|frontend|worker|all]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Get version from VERSION file
VERSION=$(cat "$PROJECT_DIR/VERSION" 2>/dev/null || echo "dev")

# Image names
BACKEND_IMAGE="infinirc/lmstack-backend:local"
FRONTEND_IMAGE="infinirc/lmstack-frontend:local"
WORKER_IMAGE="infinirc/lmstack-worker:local"

build_backend() {
    echo "Building backend image..."
    docker build -t "$BACKEND_IMAGE" -f "$PROJECT_DIR/backend/Dockerfile" "$PROJECT_DIR/backend"
    echo "✓ Backend image built: $BACKEND_IMAGE"
}

build_frontend() {
    echo "Building frontend image..."
    docker build -t "$FRONTEND_IMAGE" -f "$PROJECT_DIR/frontend/Dockerfile" "$PROJECT_DIR/frontend"
    echo "✓ Frontend image built: $FRONTEND_IMAGE"
}

build_worker() {
    echo "Building worker image..."
    docker build -t "$WORKER_IMAGE" -f "$PROJECT_DIR/worker/Dockerfile" "$PROJECT_DIR/worker"
    echo "✓ Worker image built: $WORKER_IMAGE"
}

build_all() {
    build_backend
    build_frontend
    build_worker
}

case "${1:-all}" in
    backend)
        build_backend
        ;;
    frontend)
        build_frontend
        ;;
    worker)
        build_worker
        ;;
    all)
        build_all
        ;;
    *)
        echo "Usage: $0 [backend|frontend|worker|all]"
        exit 1
        ;;
esac

echo ""
echo "To run locally:"
echo "  docker compose -f docker-compose.local.yml up -d"
