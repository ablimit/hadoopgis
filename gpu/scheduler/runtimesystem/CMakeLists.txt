cmake_minimum_required(VERSION 2.6)
project(Runtime)

# The version number
set(ExecEngine_VERSION_MAJOR 1)
set(ExecEngine_VERSION_MINOR 0)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})


# Source files used by the Runtime System
set( libsyssrcs SysEnv.cpp Worker.cpp Manager.cpp Argument.cpp PipelineComponentBase.cpp CallBackComponentExecution.cpp ./util/Util.cpp )

set( libname runtimesystem )

# if build same applications is set, than build them: currently just one
add_subdirectory("ResourceManager")
add_subdirectory("examples")
add_subdirectory("tests")

include_directories(./util)
include_directories(./ResourceManager)

# Add rule to generate the Runtime System, as a shared library (.so)
add_library("${libname}" SHARED ${libsyssrcs})

#set_target_properties("${libname}" PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}") 
target_link_libraries("${libname}" ${MPI_LIBRARIES} execengine)

