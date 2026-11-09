# Official llama.cpp server image with Vulkan (Ubuntu 26.04, Mesa drivers for
# AMD and Intel). Pinned to the tested build.
FROM ghcr.io/ggml-org/llama.cpp:server-vulkan-b12063

ENV HA_LLM_BACKEND=vulkan
COPY run.sh /run.sh
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
