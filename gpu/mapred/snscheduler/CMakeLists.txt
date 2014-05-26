cmake_minimum_required(VERSION 2.6)
project(TaskDependency)

list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
set(RM_SRC_DIR /home/aaji/proj/hadoopgis/gpu/scheduler/runtimesystem/ResourceManager )
# Make sure the compiler can find the includes in the Execution Engine directory
include_directories(${RM_SRC_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${RM_SRC_DIR})

# Add an executable to be built from the files listed
add_executable(task_dependency TaskId.cpp TaskDependency.cpp)

# Link the executable to the execution engine library
target_link_libraries(task_dependency execengine)
