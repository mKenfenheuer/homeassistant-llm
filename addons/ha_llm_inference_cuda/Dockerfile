# Official llama.cpp server image with CUDA 12.8 (Ubuntu 24.04). CUDA 12
# still supports Maxwell, Pascal and Volta. The NVIDIA driver comes from the
# host through CDI. Pinned to the tested build.
FROM ghcr.io/ggml-org/llama.cpp:server-cuda-b12063

ENV HA_LLM_BACKEND=cuda
COPY run.sh /run.sh
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
