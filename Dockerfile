# =============================================================================
# ResonanceNet — Multi-stage Docker Build
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        pkg-config \
        libssl-dev \
        ocl-icd-opencl-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
    && cmake --build build --parallel "$(nproc)"

# Verify the build produced key binaries
RUN test -x build/rnetd && test -x build/rnet-cli

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libssl3t64 \
        ocl-icd-libopencl1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -r -m -d /var/lib/resonancenet -s /usr/sbin/nologin rnet

# Copy binaries from builder
COPY --from=builder /src/build/rnetd       /usr/local/bin/
COPY --from=builder /src/build/rnet-cli    /usr/local/bin/
COPY --from=builder /src/build/rnet-tx     /usr/local/bin/
COPY --from=builder /src/build/rnet-miner  /usr/local/bin/
COPY --from=builder /src/build/rnet-util   /usr/local/bin/

# Data and config directories
RUN mkdir -p /etc/resonancenet /var/lib/resonancenet \
    && chown -R rnet:rnet /var/lib/resonancenet

VOLUME ["/var/lib/resonancenet"]

# P2P and RPC ports
EXPOSE 9555 9556

USER rnet

ENTRYPOINT ["rnetd"]
CMD ["-conf=/etc/resonancenet/resonancenet.conf", "-datadir=/var/lib/resonancenet", "-printtoconsole"]
