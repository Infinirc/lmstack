#!/bin/bash
# Bump version across all project files
# Usage: ./scripts/bump-version.sh <new_version>
# Example: ./scripts/bump-version.sh 0.2.0

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.2.0"
    exit 1
fi

NEW_VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Validate version format (semver)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 1.0.0)"
    exit 1
fi

echo "Bumping version to $NEW_VERSION..."

# Update VERSION file
echo "$NEW_VERSION" > "$PROJECT_ROOT/VERSION"
echo "  ✓ VERSION"

# Update backend/pyproject.toml
sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PROJECT_ROOT/backend/pyproject.toml"
echo "  ✓ backend/pyproject.toml"

# Update frontend/package.json
sed -i "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" "$PROJECT_ROOT/frontend/package.json"
echo "  ✓ frontend/package.json"

echo ""
echo "Version updated to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  1. git add -A"
echo "  2. git commit -m \"chore: bump version to $NEW_VERSION\""
echo "  3. git tag -a v$NEW_VERSION -m \"Release v$NEW_VERSION\""
echo "  4. git push && git push --tags"
