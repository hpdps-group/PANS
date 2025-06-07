file(REMOVE_RECURSE
  "lib/libhans_compress.a"
  "lib/libhans_compress.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/hans_compress.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
