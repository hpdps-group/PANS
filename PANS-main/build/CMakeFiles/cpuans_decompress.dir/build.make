# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/conda/bin/cmake

# The command to remove a file.
RM = /opt/conda/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/yangjinwu/PANS-main

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yangjinwu/PANS-main/build

# Include any dependencies generated for this target.
include CMakeFiles/cpuans_decompress.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cpuans_decompress.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cpuans_decompress.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpuans_decompress.dir/flags.make

CMakeFiles/cpuans_decompress.dir/decompress.cpp.o: CMakeFiles/cpuans_decompress.dir/flags.make
CMakeFiles/cpuans_decompress.dir/decompress.cpp.o: ../decompress.cpp
CMakeFiles/cpuans_decompress.dir/decompress.cpp.o: CMakeFiles/cpuans_decompress.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yangjinwu/PANS-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cpuans_decompress.dir/decompress.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cpuans_decompress.dir/decompress.cpp.o -MF CMakeFiles/cpuans_decompress.dir/decompress.cpp.o.d -o CMakeFiles/cpuans_decompress.dir/decompress.cpp.o -c /home/yangjinwu/PANS-main/decompress.cpp

CMakeFiles/cpuans_decompress.dir/decompress.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpuans_decompress.dir/decompress.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yangjinwu/PANS-main/decompress.cpp > CMakeFiles/cpuans_decompress.dir/decompress.cpp.i

CMakeFiles/cpuans_decompress.dir/decompress.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpuans_decompress.dir/decompress.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yangjinwu/PANS-main/decompress.cpp -o CMakeFiles/cpuans_decompress.dir/decompress.cpp.s

# Object files for target cpuans_decompress
cpuans_decompress_OBJECTS = \
"CMakeFiles/cpuans_decompress.dir/decompress.cpp.o"

# External object files for target cpuans_decompress
cpuans_decompress_EXTERNAL_OBJECTS =

cpuans_decompress: CMakeFiles/cpuans_decompress.dir/decompress.cpp.o
cpuans_decompress: CMakeFiles/cpuans_decompress.dir/build.make
cpuans_decompress: /usr/lib/gcc/x86_64-linux-gnu/9/libgomp.so
cpuans_decompress: /usr/lib/x86_64-linux-gnu/libpthread.so
cpuans_decompress: CMakeFiles/cpuans_decompress.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yangjinwu/PANS-main/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cpuans_decompress"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpuans_decompress.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpuans_decompress.dir/build: cpuans_decompress
.PHONY : CMakeFiles/cpuans_decompress.dir/build

CMakeFiles/cpuans_decompress.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpuans_decompress.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpuans_decompress.dir/clean

CMakeFiles/cpuans_decompress.dir/depend:
	cd /home/yangjinwu/PANS-main/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yangjinwu/PANS-main /home/yangjinwu/PANS-main /home/yangjinwu/PANS-main/build /home/yangjinwu/PANS-main/build /home/yangjinwu/PANS-main/build/CMakeFiles/cpuans_decompress.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cpuans_decompress.dir/depend

