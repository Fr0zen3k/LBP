cmake_minimum_required(VERSION 3.20.0)

project(LBP VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/LBP/src")

set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/LBP/vendor/opencv/build/x64/vc15/lib")
find_package(OpenCV REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/LBP/vendor/opencv/build")

set(
	LBP_FILES
	"${LBP_SOURCE_FILES}"
	"${LBP_HEADER_FILES}"
	CACHE INTERNAL "" FORCE
	)

add_executable(LBP "${LBP_FILES}")

target_include_directories(LBP PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/LBP/src" "${OpenCV_INCLUDE_DIRS}")

target_compile_definitions(LBP PUBLIC _USE_MATH_DEFINES)

target_link_libraries(LBP ${OpenCV_LIBS})