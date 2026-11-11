# ggml of the llama.cpp image in /app for WHISPER_USE_SYSTEM_GGML
set(ggml_FOUND TRUE)
foreach(lib ggml ggml-base)
  if(NOT TARGET ggml::${lib})
    add_library(ggml::${lib} SHARED IMPORTED)
    set_target_properties(ggml::${lib} PROPERTIES
      IMPORTED_LOCATION /app/lib${lib}.so
      INTERFACE_INCLUDE_DIRECTORIES /usr/local/include/ggml)
  endif()
endforeach()
set_target_properties(ggml::ggml PROPERTIES INTERFACE_LINK_LIBRARIES ggml::ggml-base)
