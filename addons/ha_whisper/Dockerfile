# Speech-to-text with whisper.cpp on the GPU or the CPU, built on the same
# official llama.cpp images as the HA LLM Inference app. whisper.cpp is
# compiled against the ggml of the llama.cpp image, so it uses the same
# runtime backends: CUDA (NVIDIA, amd64), Vulkan (AMD, Intel, ARM) and the
# CPU. Docker shares the base layers with the LLM app.
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

# whisper.cpp against the ggml (headers of the same llama.cpp build) in /app
FROM base-aarch64 AS build
ARG WHISPER_VERSION=v1.9.5
ARG LLAMA_VERSION=b12063
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b "${WHISPER_VERSION}" https://github.com/ggml-org/whisper.cpp.git /src/whisper.cpp \
    && git clone --depth 1 --filter=blob:none --sparse -b "${LLAMA_VERSION}" https://github.com/ggml-org/llama.cpp.git /src/llama.cpp \
    && git -C /src/llama.cpp sparse-checkout set ggml/include \
    && mkdir -p /usr/local/include \
    && cp -r /src/llama.cpp/ggml/include /usr/local/include/ggml
COPY ggml-config.cmake /usr/local/lib/cmake/ggml/ggml-config.cmake
RUN cmake -S /src/whisper.cpp -B /build -DCMAKE_BUILD_TYPE=Release \
        -DWHISPER_USE_SYSTEM_GGML=ON -DBUILD_SHARED_LIBS=ON -DWHISPER_SDL2=OFF \
        -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_EXAMPLES=ON \
    && cmake --build /build -j"$(nproc)" --target whisper-server

FROM base-${BUILD_ARCH}
COPY --from=build /build/bin/whisper-server /app/whisper-server
COPY --from=build /build/bin/libwhisper.so* /app/
RUN ldconfig /app
COPY run.sh /run.sh
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
