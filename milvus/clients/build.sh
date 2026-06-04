#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <go_project_directory>"
    exit 1
fi

DIR="$1"

if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' does not exist"
    exit 1
fi

cd "$DIR"

# Ensure there is at least one Go file
if ! ls *.go >/dev/null 2>&1; then
    echo "Error: No .go files found in $DIR"
    exit 1
fi

BIN_NAME=$(basename "$DIR")

echo "Building Go project in: $DIR"
echo "Output binary: $DIR/$BIN_NAME"

# If go.mod does not exist, initialize module
if [ ! -f "go.mod" ]; then
    echo "No go.mod found. Initializing module..."
    go mod init "$BIN_NAME"

    echo "Pinning dependency versions compatible with etcd v3.5.5..."
    go get go.opentelemetry.io/otel@v1.19.0
    go get go.opentelemetry.io/otel/trace@v1.19.0
    go get go.opentelemetry.io/otel/metric@v1.19.0
    go get go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc@v0.45.0

    # Download and clean dependencies
    echo "Tidying modules..."
    go mod tidy
fi


# Build entire module (recommended over building main.go explicitly)
go build -o "$BIN_NAME"

echo "Build complete."
