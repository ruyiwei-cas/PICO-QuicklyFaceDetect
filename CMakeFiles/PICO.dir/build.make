# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ruyiwei/PICO

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ruyiwei/PICO

# Include any dependencies generated for this target.
include CMakeFiles/PICO.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PICO.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PICO.dir/flags.make

CMakeFiles/PICO.dir/picornt.cpp.o: CMakeFiles/PICO.dir/flags.make
CMakeFiles/PICO.dir/picornt.cpp.o: picornt.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ruyiwei/PICO/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PICO.dir/picornt.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PICO.dir/picornt.cpp.o -c /home/ruyiwei/PICO/picornt.cpp

CMakeFiles/PICO.dir/picornt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PICO.dir/picornt.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ruyiwei/PICO/picornt.cpp > CMakeFiles/PICO.dir/picornt.cpp.i

CMakeFiles/PICO.dir/picornt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PICO.dir/picornt.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ruyiwei/PICO/picornt.cpp -o CMakeFiles/PICO.dir/picornt.cpp.s

CMakeFiles/PICO.dir/picornt.cpp.o.requires:

.PHONY : CMakeFiles/PICO.dir/picornt.cpp.o.requires

CMakeFiles/PICO.dir/picornt.cpp.o.provides: CMakeFiles/PICO.dir/picornt.cpp.o.requires
	$(MAKE) -f CMakeFiles/PICO.dir/build.make CMakeFiles/PICO.dir/picornt.cpp.o.provides.build
.PHONY : CMakeFiles/PICO.dir/picornt.cpp.o.provides

CMakeFiles/PICO.dir/picornt.cpp.o.provides.build: CMakeFiles/PICO.dir/picornt.cpp.o


CMakeFiles/PICO.dir/sample.cpp.o: CMakeFiles/PICO.dir/flags.make
CMakeFiles/PICO.dir/sample.cpp.o: sample.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ruyiwei/PICO/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/PICO.dir/sample.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PICO.dir/sample.cpp.o -c /home/ruyiwei/PICO/sample.cpp

CMakeFiles/PICO.dir/sample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PICO.dir/sample.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ruyiwei/PICO/sample.cpp > CMakeFiles/PICO.dir/sample.cpp.i

CMakeFiles/PICO.dir/sample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PICO.dir/sample.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ruyiwei/PICO/sample.cpp -o CMakeFiles/PICO.dir/sample.cpp.s

CMakeFiles/PICO.dir/sample.cpp.o.requires:

.PHONY : CMakeFiles/PICO.dir/sample.cpp.o.requires

CMakeFiles/PICO.dir/sample.cpp.o.provides: CMakeFiles/PICO.dir/sample.cpp.o.requires
	$(MAKE) -f CMakeFiles/PICO.dir/build.make CMakeFiles/PICO.dir/sample.cpp.o.provides.build
.PHONY : CMakeFiles/PICO.dir/sample.cpp.o.provides

CMakeFiles/PICO.dir/sample.cpp.o.provides.build: CMakeFiles/PICO.dir/sample.cpp.o


# Object files for target PICO
PICO_OBJECTS = \
"CMakeFiles/PICO.dir/picornt.cpp.o" \
"CMakeFiles/PICO.dir/sample.cpp.o"

# External object files for target PICO
PICO_EXTERNAL_OBJECTS =

PICO: CMakeFiles/PICO.dir/picornt.cpp.o
PICO: CMakeFiles/PICO.dir/sample.cpp.o
PICO: CMakeFiles/PICO.dir/build.make
PICO: /usr/local/lib/libopencv_cudabgsegm.so.3.2.0
PICO: /usr/local/lib/libopencv_cudaobjdetect.so.3.2.0
PICO: /usr/local/lib/libopencv_cudastereo.so.3.2.0
PICO: /usr/local/lib/libopencv_stitching.so.3.2.0
PICO: /usr/local/lib/libopencv_superres.so.3.2.0
PICO: /usr/local/lib/libopencv_videostab.so.3.2.0
PICO: /usr/local/lib/libopencv_aruco.so.3.2.0
PICO: /usr/local/lib/libopencv_bgsegm.so.3.2.0
PICO: /usr/local/lib/libopencv_bioinspired.so.3.2.0
PICO: /usr/local/lib/libopencv_ccalib.so.3.2.0
PICO: /usr/local/lib/libopencv_dpm.so.3.2.0
PICO: /usr/local/lib/libopencv_freetype.so.3.2.0
PICO: /usr/local/lib/libopencv_fuzzy.so.3.2.0
PICO: /usr/local/lib/libopencv_line_descriptor.so.3.2.0
PICO: /usr/local/lib/libopencv_optflow.so.3.2.0
PICO: /usr/local/lib/libopencv_reg.so.3.2.0
PICO: /usr/local/lib/libopencv_saliency.so.3.2.0
PICO: /usr/local/lib/libopencv_stereo.so.3.2.0
PICO: /usr/local/lib/libopencv_structured_light.so.3.2.0
PICO: /usr/local/lib/libopencv_surface_matching.so.3.2.0
PICO: /usr/local/lib/libopencv_tracking.so.3.2.0
PICO: /usr/local/lib/libopencv_xfeatures2d.so.3.2.0
PICO: /usr/local/lib/libopencv_ximgproc.so.3.2.0
PICO: /usr/local/lib/libopencv_xobjdetect.so.3.2.0
PICO: /usr/local/lib/libopencv_xphoto.so.3.2.0
PICO: /usr/local/lib/libopencv_cudafeatures2d.so.3.2.0
PICO: /usr/local/lib/libopencv_shape.so.3.2.0
PICO: /usr/local/lib/libopencv_cudacodec.so.3.2.0
PICO: /usr/local/lib/libopencv_cudaoptflow.so.3.2.0
PICO: /usr/local/lib/libopencv_cudalegacy.so.3.2.0
PICO: /usr/local/lib/libopencv_cudawarping.so.3.2.0
PICO: /usr/local/lib/libopencv_phase_unwrapping.so.3.2.0
PICO: /usr/local/lib/libopencv_rgbd.so.3.2.0
PICO: /usr/local/lib/libopencv_calib3d.so.3.2.0
PICO: /usr/local/lib/libopencv_video.so.3.2.0
PICO: /usr/local/lib/libopencv_datasets.so.3.2.0
PICO: /usr/local/lib/libopencv_dnn.so.3.2.0
PICO: /usr/local/lib/libopencv_face.so.3.2.0
PICO: /usr/local/lib/libopencv_plot.so.3.2.0
PICO: /usr/local/lib/libopencv_text.so.3.2.0
PICO: /usr/local/lib/libopencv_features2d.so.3.2.0
PICO: /usr/local/lib/libopencv_flann.so.3.2.0
PICO: /usr/local/lib/libopencv_objdetect.so.3.2.0
PICO: /usr/local/lib/libopencv_ml.so.3.2.0
PICO: /usr/local/lib/libopencv_highgui.so.3.2.0
PICO: /usr/local/lib/libopencv_photo.so.3.2.0
PICO: /usr/local/lib/libopencv_cudaimgproc.so.3.2.0
PICO: /usr/local/lib/libopencv_cudafilters.so.3.2.0
PICO: /usr/local/lib/libopencv_cudaarithm.so.3.2.0
PICO: /usr/local/lib/libopencv_videoio.so.3.2.0
PICO: /usr/local/lib/libopencv_imgcodecs.so.3.2.0
PICO: /usr/local/lib/libopencv_imgproc.so.3.2.0
PICO: /usr/local/lib/libopencv_core.so.3.2.0
PICO: /usr/local/lib/libopencv_cudev.so.3.2.0
PICO: CMakeFiles/PICO.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ruyiwei/PICO/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable PICO"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PICO.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PICO.dir/build: PICO

.PHONY : CMakeFiles/PICO.dir/build

CMakeFiles/PICO.dir/requires: CMakeFiles/PICO.dir/picornt.cpp.o.requires
CMakeFiles/PICO.dir/requires: CMakeFiles/PICO.dir/sample.cpp.o.requires

.PHONY : CMakeFiles/PICO.dir/requires

CMakeFiles/PICO.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PICO.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PICO.dir/clean

CMakeFiles/PICO.dir/depend:
	cd /home/ruyiwei/PICO && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ruyiwei/PICO /home/ruyiwei/PICO /home/ruyiwei/PICO /home/ruyiwei/PICO /home/ruyiwei/PICO/CMakeFiles/PICO.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PICO.dir/depend

