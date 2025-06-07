set(MIX_SOURCES
)
set(AIV_SOURCES
    /root/yjw/HANS/HANS-NPU/build/auto_gen/hans_compress/auto_gen_hans_table.cpp
    /root/yjw/HANS/HANS-NPU/build/auto_gen/hans_compress/auto_gen_hans_compress.cpp
    /root/yjw/HANS/HANS-NPU/build/auto_gen/hans_compress/auto_gen_hans_merge.cpp
)
set_source_files_properties(/root/yjw/HANS/HANS-NPU/build/auto_gen/hans_compress/auto_gen_hans_compress.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_comp_kernel=comp_2;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/root/yjw/HANS/HANS-NPU/build/auto_gen/hans_compress/auto_gen_hans_merge.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_calcprefix_kernel=calcprefix_3;ONE_CORE_DUMP_SIZE=1048576;;auto_gen_coalesce_kernel=coalesce_4;ONE_CORE_DUMP_SIZE=1048576"
)
set_source_files_properties(/root/yjw/HANS/HANS-NPU/build/auto_gen/hans_compress/auto_gen_hans_table.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_MergeHistogram_kernel=MergeHistogram_0;ONE_CORE_DUMP_SIZE=1024;;auto_gen_extractbits_and_histogram_kernel=extractbits_and_histogram_1;ONE_CORE_DUMP_SIZE=1024"
)
