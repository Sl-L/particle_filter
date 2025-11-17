cmake_minimum_required(VERSION 3.8)
project(particle_filter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclc REQUIRED)
find_package(std_msgs REQUIRED)

include(FetchContent)

# --- CMSIS 6 (CORE) ----------------------------------------------------------

FetchContent_Declare(
  CMSIS
  GIT_REPOSITORY https://github.com/ARM-software/CMSIS_6.git
  GIT_TAG main
)

FetchContent_MakeAvailable(CMSIS)

# CMSIS 6 Core include path (required by CMSIS-DSP)
set(CMSIS_CORE_INCLUDE_DIR "${CMSIS_SOURCE_DIR}/CMSIS/Core/Include")

# Make it globally visible so CMSIS-DSP can find cmsis_compiler.h
include_directories(${CMSIS_CORE_INCLUDE_DIR})

# --- CMSIS-DSP ----------------------------------------------------------------

FetchContent_Declare(
  CMSISDSP
  GIT_REPOSITORY https://github.com/ARM-software/CMSIS-DSP
  GIT_TAG main
)

FetchContent_MakeAvailable(CMSISDSP)

# --- Executable ----------------------------------------------------------

add_executable(main src/main.c)
ament_target_dependencies(main rclc std_msgs)

# Add local include folder
target_include_directories(main PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link CMSIS-DSP library
target_link_libraries(main CMSISDSP)

# Require C17 and C++17
target_compile_features(main PUBLIC c_std_17 cxx_std_17)

install(TARGETS main
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()