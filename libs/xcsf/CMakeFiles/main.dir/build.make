# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.27

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "D:\Coding Stuff\Bachelorarbeit\repos\xcsf"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build"

# Include any dependencies generated for this target.
include xcsf/CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include xcsf/CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include xcsf/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include xcsf/CMakeFiles/main.dir/flags.make

xcsf/CMakeFiles/main.dir/main.c.obj: xcsf/CMakeFiles/main.dir/flags.make
xcsf/CMakeFiles/main.dir/main.c.obj: D:/Coding\ Stuff/Bachelorarbeit/repos/xcsf/xcsf/main.c
xcsf/CMakeFiles/main.dir/main.c.obj: xcsf/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir="D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building C object xcsf/CMakeFiles/main.dir/main.c.obj"
	cd /d "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" && C:\msys64\ucrt64\bin\cc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT xcsf/CMakeFiles/main.dir/main.c.obj -MF CMakeFiles\main.dir\main.c.obj.d -o CMakeFiles\main.dir\main.c.obj -c "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\xcsf\main.c"

xcsf/CMakeFiles/main.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/main.dir/main.c.i"
	cd /d "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" && C:\msys64\ucrt64\bin\cc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\xcsf\main.c" > CMakeFiles\main.dir\main.c.i

xcsf/CMakeFiles/main.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/main.dir/main.c.s"
	cd /d "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" && C:\msys64\ucrt64\bin\cc.exe $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\xcsf\main.c" -o CMakeFiles\main.dir\main.c.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.c.obj"

# External object files for target main
main_EXTERNAL_OBJECTS =

xcsf/main.exe: xcsf/CMakeFiles/main.dir/main.c.obj
xcsf/main.exe: xcsf/CMakeFiles/main.dir/build.make
xcsf/main.exe: xcsf/libxcs.a
xcsf/main.exe: C:/msys64/ucrt64/lib/libgomp.dll.a
xcsf/main.exe: C:/msys64/ucrt64/lib/libmingwthrd.a
xcsf/main.exe: xcsf/CMakeFiles/main.dir/linkLibs.rsp
xcsf/main.exe: xcsf/CMakeFiles/main.dir/objects1.rsp
xcsf/main.exe: xcsf/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable main.exe"
	cd /d "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\main.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
xcsf/CMakeFiles/main.dir/build: xcsf/main.exe
.PHONY : xcsf/CMakeFiles/main.dir/build

xcsf/CMakeFiles/main.dir/clean:
	cd /d "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" && $(CMAKE_COMMAND) -P CMakeFiles\main.dir\cmake_clean.cmake
.PHONY : xcsf/CMakeFiles/main.dir/clean

xcsf/CMakeFiles/main.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "D:\Coding Stuff\Bachelorarbeit\repos\xcsf" "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\xcsf" "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build" "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf" "D:\Coding Stuff\Bachelorarbeit\repos\xcsf\build\xcsf\CMakeFiles\main.dir\DependInfo.cmake" "--color=$(COLOR)"
.PHONY : xcsf/CMakeFiles/main.dir/depend

