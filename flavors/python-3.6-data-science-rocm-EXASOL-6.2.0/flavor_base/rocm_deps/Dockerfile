FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

COPY 01_nodoc /etc/dpkg/dpkg.cfg.d/01_nodoc


# Register the ROCM package repository, and install rocm-dev package
ARG ROCM_VERSION=4.2
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl libnuma-dev gnupg \
  && curl -sL https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - \
  && printf "deb [arch=amd64] https://repo.radeon.com/rocm/apt/$ROCM_VERSION/ ubuntu main" | tee /etc/apt/sources.list.d/rocm.list \
  && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  sudo \
  libelf1 \
  kmod \
  file \
  rocm-dev \
  rocm-libs \
  miopen-hip \
  rccl \
  build-essential && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

