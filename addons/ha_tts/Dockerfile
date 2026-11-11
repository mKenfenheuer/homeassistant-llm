# Text-to-speech with llama-server (POST /tts) on the GPU or the CPU, the same
# image as the HA LLM Inference app. llama.cpp loads its backends at runtime
# and skips those without hardware: CPU, AMD and Intel GPUs (Vulkan, Mesa) and
# NVIDIA GPUs (CUDA, driver libraries from the host).
#
# The Vulkan image (Ubuntu 26.04) is the base. On amd64 the CUDA backend and
# the CUDA 12.8 runtime come from the CUDA image (Ubuntu 24.04), they run on
# the newer system. Without an NVIDIA driver the CUDA backend isn't loaded.
ARG BUILD_ARCH=amd64

FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b12063 AS cuda

FROM ghcr.io/ggml-org/llama.cpp:server-vulkan-b12063 AS base-aarch64

FROM base-aarch64 AS base-amd64
COPY --from=cuda /app/libggml-cuda.so /app/
COPY --from=cuda \
    /usr/local/cuda-12.8/lib64/libcudart.so.12* \
    /usr/local/cuda-12.8/lib64/libcublas.so.12* \
    /usr/local/cuda-12.8/lib64/libcublasLt.so.12* \
    /lib/x86_64-linux-gnu/libnccl.so.2* \
    /usr/local/lib/cuda/
RUN echo /usr/local/lib/cuda > /etc/ld.so.conf.d/cuda.conf && ldconfig

FROM base-${BUILD_ARCH}
COPY run.sh /run.sh
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
