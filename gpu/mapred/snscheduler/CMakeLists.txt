project(crossmatch)
cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)
# list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
# add_definitions(-std=c++0x)
if (${APPLE})
  MESSAGE("Apple system makefile")
  set(ROOT_SRC_DIR /Users/ablimit/Documents)
else()
  MESSAGE("Unix system makefile")
  set(ROOT_SRC_DIR /home/aaji)
  link_directories(/home/aaji/softs/lib)
endif()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -)


set(RM_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/scheduler/runtimesystem/ResourceManager )
set(SP_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/mapred )
set(CUDA_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/mapred/gpu )

## Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++0x")        ## Optimize
endif()

# Make sure the compiler can find the includes in the Execution Engine directory
include_directories( ${RM_SRC_DIR} )
include_directories( ${SP_SRC_DIR} )
include_directories( ${SP_SRC_DIR}/cpu )
include_directories( ${CUDA_SRC_DIR} )

# Make sure the compiler can find the library in Execution Engine directory

#SET(LIB_TYPE STATIC) #set the lib type

#cuda_add_library(gpurefine ${CUDA_SRC_DIR}/util.cu ${CUDA_SRC_DIR}/parser.cu ${CUDA_SRC_DIR}/boxpixel.cu)

# Add an executable to be built from the files listed
add_executable(crossmatch ${SP_SRC_DIR}/hilbert.c ${SP_SRC_DIR}/rstar.c ${SP_SRC_DIR}/rtree.c ${SP_SRC_DIR}/spatial.c ${SP_SRC_DIR}/spatialindex.c JoinTask.cpp JoinEngine.cpp )


add_executable(jet ${SP_SRC_DIR}/hilbert.c ${SP_SRC_DIR}/rstar.c ${SP_SRC_DIR}/rtree.c ${SP_SRC_DIR}/spatial.c ${SP_SRC_DIR}/spatialindex.c JoinTask.cpp TestJoinEngine.cpp )
# Link the executable to the execution engine library
target_link_libraries(crossmatch execengine)
target_link_libraries(crossmatch cpurefine)
target_link_libraries(crossmatch gpurefine)


target_link_libraries(jet execengine)
target_link_libraries(jet cpurefine)
target_link_libraries(jet gpurefine)
