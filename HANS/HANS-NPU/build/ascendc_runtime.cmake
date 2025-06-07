add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/root/yjw/HANS/HANS-NPU/build/elf_tool.c.o;/root/yjw/HANS/HANS-NPU/build/ascendc_runtime.cpp.o"
)
