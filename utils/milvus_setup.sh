if ! command -v go >/dev/null 2>&1; then
  echo "Go is not installed"

    ########################
    # CONFIG
    ########################

    # Base directory where everything will live.
    # Defaults to ~/goInstall, but can be overridden with:
    #   GO_INSTALL_BASE=/some/path ./install_go.sh
    BASE_DIR="${GO_INSTALL_BASE:-${HOME}/goInstall}"

    # Go versions
    BOOTSTRAP_VERSION="1.22.6"   # binary bootstrap
    GO_VERSION="go1.25.4"        # target version to build

    # Paths
    BOOTSTRAP_TARBALL="go${BOOTSTRAP_VERSION}.linux-amd64.tar.gz"
    BOOTSTRAP_DIR="${BASE_DIR}/go"                    # will contain bin/go
    GOROOT_TARGET="${BASE_DIR}/opt/${GO_VERSION}"     # source + final toolchain

    ########################
    # 0. Modules
    ########################

    # module load PrgEnv-gnu

    mkdir -p "${BASE_DIR}/opt"
    cd "${BASE_DIR}"

    echo "=== Using BASE_DIR=${BASE_DIR}"
    echo "=== Bootstrap Go version: ${BOOTSTRAP_VERSION}"
    echo "=== Target Go version:    ${GO_VERSION}"
    echo

    ########################
    # 1. Get bootstrap Go if needed
    ########################

    if [ ! -x "${BOOTSTRAP_DIR}/bin/go" ]; then
    echo "=== Downloading bootstrap Go ${BOOTSTRAP_VERSION}..."
    rm -rf "${BOOTSTRAP_DIR}"
    rm -f "${BOOTSTRAP_TARBALL}"

    wget "https://go.dev/dl/${BOOTSTRAP_TARBALL}"
    tar -xzf "${BOOTSTRAP_TARBALL}"
    else
    echo "=== Bootstrap Go already present at ${BOOTSTRAP_DIR}"
    fi

    echo "=== Bootstrap go: $(${BOOTSTRAP_DIR}/bin/go version)"
    echo

    ########################
    # 2. Clone Go source if needed
    ########################

    if [ ! -d "${GOROOT_TARGET}" ]; then
    echo "=== Cloning Go repository into ${GOROOT_TARGET}..."
    git clone https://go.googlesource.com/go "${GOROOT_TARGET}"
    fi

    cd "${GOROOT_TARGET}"

    echo "=== Checking out ${GO_VERSION}..."
    git fetch --tags
    git checkout "${GO_VERSION}"

    ########################
    # 3. Build Go from source
    ########################

    echo "=== Building Go ${GO_VERSION} from source..."
    cd src

    unset GOROOT || true
    export GOROOT_BOOTSTRAP="${BOOTSTRAP_DIR}"

    echo "=== Using GOROOT_BOOTSTRAP=${GOROOT_BOOTSTRAP}"
    echo "=== Running ./make.bash..."
    ./make.bash

    echo
    echo "=== Build completed successfully. ==="
    echo

    export PATH="${GOROOT_TARGET}/bin:$PATH"

    echo "=== Go is now available in this shell:"
    echo "    $(${GOROOT_TARGET}/bin/go version)"
    echo
    echo "To make this permanent, add this to ~/.bashrc:"
    echo
    echo "    export GOROOT=${GOROOT_TARGET}"
    echo "    export PATH=\"\$GOROOT/bin:\$PATH\""
    echo
    echo "Then run:"
    echo
    echo "    source ~/.bashrc"
else
  echo "Go is installed: $(go version)"
  # do other thing here
fi



git clone https://github.com/OckermanSethGVSU/VECHINI.git
cd VECHINI/

module load apptainer
cd milvus/utils/

MILVUS_SIF_NAME=milvus.sif bash download_sif.sh 2.6.6

cd ../clients/

bash build.sh batch_client/
