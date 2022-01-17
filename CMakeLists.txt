cmake_minimum_required(VERSION 3.17.0)

project(Crowd_Detection VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/LBP/src")

set(HAVE_FFMPEG 1)
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/LBP/vendor/opencv/build/")
find_package(OpenCV REQUIRED PATHS "${CMAKE_CURRENT_SOURCE_DIR}/LBP/vendor/opencv/build")

set(
	LBP_FILES
	"${LBP_SOURCE_FILES}"
	"${LBP_HEADER_FILES}"
	CACHE INTERNAL "" FORCE
	)

add_executable(Crowd_Detection "${LBP_FILES}")

target_include_directories(Crowd_Detection PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/LBP/src" "${OpenCV_INCLUDE_DIRS}" "${CMAKE_CURRENT_SOURCE_DIR}/LBP/vendor/STBI")

target_compile_definitions(Crowd_Detection PUBLIC _USE_MATH_DEFINES STB_IMAGE_IMPLEMENTATION)

target_link_libraries(Crowd_Detection ${OpenCV_LIBS})