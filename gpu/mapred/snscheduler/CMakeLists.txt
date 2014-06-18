CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(crossmatch)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE( Threads REQUIRED )

# list( APPEND CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
# add_definitions(-std=c++0x)

IF (${APPLE})
  MESSAGE("Apple system makefile")
  SET(ROOT_SRC_DIR /Users/ablimit/Documents)
ELSE()
  MESSAGE("Unix system makefile")
  SET(ROOT_SRC_DIR /home/aaji)
  LINK_DIRECTORIES(/home/aaji/softs/lib)
ENDIF()

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_20 -)
SET(RM_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/scheduler/runtimesystem/ResourceManager )
SET(SP_SRC_DIR ${ROOT_SRC_DIR}/proj/hadoopgis/gpu/mapred )

## Compiler flags
IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "-O2 -Wall -std=c++0x")        ## Optimize
ENDIF()

# Make sure the compiler can find the includes in the Execution Engine directory
INCLUDE_DIRECTORIES( ${RM_SRC_DIR} )
INCLUDE_DIRECTORIES( ${SP_SRC_DIR} )
INCLUDE_DIRECTORIES( ${SP_SRC_DIR}/cpu )
INCLUDE_DIRECTORIES( ${SP_SRC_DIR}/gpu )

LINK_DIRECTORIES( /usr/local/cuda/lib64 )

# Add an executable to be built from the files listed
ADD_EXECUTABLE(crossmatch ${SP_SRC_DIR}/hilbert.c ${SP_SRC_DIR}/rstar.c ${SP_SRC_DIR}/rtree.c ${SP_SRC_DIR}/spatial.c ${SP_SRC_DIR}/spatialindex.c JoinTask.cpp JoinEngine.cpp )
#add_executable(jet ${SP_SRC_DIR}/hilbert.c ${SP_SRC_DIR}/rstar.c ${SP_SRC_DIR}/rtree.c ${SP_SRC_DIR}/spatial.c ${SP_SRC_DIR}/spatialindex.c JoinTask.cpp TestJoinEngine.cpp )

# Link the executable to the execution engine library
TARGET_LINK_LIBRARIES(crossmatch ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES(crossmatch dl)
TARGET_LINK_LIBRARIES(crossmatch rt)
TARGET_LINK_LIBRARIES(crossmatch libexecengine.a)
TARGET_LINK_LIBRARIES(crossmatch libcpurefine.a)
TARGET_LINK_LIBRARIES(crossmatch libgpurefine.a)
TARGET_LINK_LIBRARIES(crossmatch libcudart_static.a)

#target_link_libraries(jet execengine)
#target_link_libraries(jet cpurefine)
#target_link_libraries(jet gpurefine)

INSTALL(TARGETS crossmatch DESTINATION bin)

