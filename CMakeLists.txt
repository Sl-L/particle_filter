cmake_minimum_required(VERSION 3.8)
project(particle_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclc REQUIRED)
find_package(std_msgs REQUIRED)

include(FetchContent)

# --------------------------------------------
# 1. Fetch CMSIS Core (CMSIS_5)
# --------------------------------------------
FetchContent_Declare(
  CMSIS_Core
  GIT_REPOSITORY https://github.com/ARM-software/CMSIS_5.git
  GIT_TAG develop
)

FetchContent_MakeAvailable(CMSIS_Core)

# Path containing cmsis_compiler.h, core_cm*.h, etc.
set(CMSIS_CORE_INCLUDE_DIR
  ${cmsis_core_SOURCE_DIR}/CMSIS/Core/Include
)

# --------------------------------------------
# 2. Fetch CMSIS-DSP and tell it where CMSIS-Core is
# --------------------------------------------
FetchContent_Declare(
  CMSISDSP
  GIT_REPOSITORY https://github.com/ARM-software/CMSIS-DSP.git
  GIT_TAG main
)
# CMSIS-DSP must know the core path before building
set(CMSIS_CORE_PATH ${CMSIS_CORE_INCLUDE_DIR})

FetchContent_MakeAvailable(CMSISDSP)

# --------------------------------------------
# 3. Build your executable
# --------------------------------------------
add_executable(main src/main.c)

ament_target_dependencies(main rclc std_msgs)

target_include_directories(main PUBLIC
  ${CMSIS_CORE_INCLUDE_DIR}
  ${CMSISDSP_SOURCE_DIR}/Include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(main CMSISDSP)

target_compile_features(main PUBLIC c_std_17 cxx_std_17)

install(TARGETS main DESTINATION lib/${PROJECT_NAME})

ament_package()
