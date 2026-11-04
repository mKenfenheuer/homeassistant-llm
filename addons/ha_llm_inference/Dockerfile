# Official llama.cpp server image (Ubuntu 24.04, CPU build with runtime
# selection of the best x86/ARM kernel). Pinned to the tested build.
FROM ghcr.io/ggml-org/llama.cpp:server-b12063

COPY run.sh /run.sh
RUN chmod a+x /run.sh

ENTRYPOINT ["/run.sh"]
