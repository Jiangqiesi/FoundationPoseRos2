cmake_minimum_required(VERSION 3.8)
project(foundationpose_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PickPlace.action"
  "srv/Resegment.srv"
  "srv/AssignModels.srv"
  "srv/GripperControl.srv"
  "srv/UpdateParams.srv"
  "srv/PerceptionStatus.srv"
  "srv/ControllerStatus.srv"
  DEPENDENCIES geometry_msgs std_msgs action_msgs
)

ament_package()
