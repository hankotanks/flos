add_custom_target(python_node_requirements ALL
  COMMAND python3 -m pip install --root-user-action=ignore -r ${CMAKE_CURRENT_SOURCE_DIR}/requirements.txt
  COMMENT "${PROJECT_NAME}: Installing Python dependencies..."
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/rerun_node.py DESTINATION lib/${PROJECT_NAME})