cmake_minimum_required(VERSION 3.5)
project(labelme)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(SOURCE_CXX main.cpp mainwindow.cpp labeldialogbox.cpp canvas.cpp)
set(SOURCE_H mainwindow.h labeldialogbox.h canvas.h)
set(SOURCE_FILES ${SOURCE_CXX} ${SOURCE_H})

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)

add_executable(labelme ${SOURCE_FILES})
target_link_libraries(labelme Qt5::Core Qt5::Widgets)
