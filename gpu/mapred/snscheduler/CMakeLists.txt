cmake_minimum_required(VERSION 2.6)
project(crossmatch)

# list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
# add_definitions(-std=c++0x)
if (${APPLE})
  MESSAGE("Apple system makefile")
  set(ROOT_SRC_DIR /Users/ablimit/Documents)
else()
  MESSAGE("Unix system makefile")
  set(ROOT_SRC_DIR /home/aaji)
endif()

set(RM_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/scheduler/runtimesystem/ResourceManager )
set(SP_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/mapred )

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++0x")        ## Optimize
endif()

# Make sure the compiler can find the includes in the Execution Engine directory
include_directories(${RM_SRC_DIR})
include_directories(${SP_SRC_DIR})

# Make sure the compiler can find the library in Execution Engine directory
link_directories(${RM_SRC_DIR})

# Add an executable to be built from the files listed
add_executable(crossmatch ${SP_SRC_DIR}/hilbert.c ${SP_SRC_DIR}/rstar.c ${SP_SRC_DIR}/rtree.c ${SP_SRC_DIR}/spatial.c ${SP_SRC_DIR}/spatialindex.c JoinTask.cpp JoinEngine.cpp )

# Link the executable to the execution engine library
target_link_libraries(crossmatch execengine)
