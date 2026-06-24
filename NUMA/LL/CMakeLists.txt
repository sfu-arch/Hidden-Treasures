# CMakeLists.txt
cmake_minimum_required(VERSION 3.10) 

project(NumaCachedListBench CXX) # Changed project name

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

if(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -O2")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /O2")
endif()

find_package(Threads REQUIRED)
find_package(PkgConfig)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(NUMA libnuma)
endif()

if(NUMA_FOUND)
    include_directories(${NUMA_INCLUDE_DIRS})
    link_libraries(${NUMA_LIBRARIES})
    message(STATUS "Found NUMA library using pkg-config: Headers=${NUMA_INCLUDE_DIRS}, Libs=${NUMA_LIBRARIES}")
else()
    find_path(NUMA_INCLUDE_DIR numa.h HINTS /usr/include /usr/local/include ENV CPATH)
    find_library(NUMA_LIBRARY NAMES numa HINTS /usr/lib /usr/local/lib /usr/lib64 /usr/local/lib64 ENV LIBRARY_PATH LD_LIBRARY_PATH)
    if(NUMA_INCLUDE_DIR AND NUMA_LIBRARY)
        include_directories(${NUMA_INCLUDE_DIR})
        message(STATUS "Found NUMA library manually: Headers=${NUMA_INCLUDE_DIR}, Libs=${NUMA_LIBRARY}")
    else()
        message(FATAL_ERROR "NUMA library (libnuma) not found. Please install libnuma-dev (or equivalent) or set NUMA_INCLUDE_DIR and NUMA_LIBRARY.")
    endif()
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(numa_list # Changed executable name
    src/Main.cpp
)

target_include_directories(numa_list PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(numa_list PRIVATE
    Threads::Threads 
    ${NUMA_LIBRARY} 
)

message(STATUS "CMake Project: NumaList")
message(STATUS "CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

