include(FetchContent)

function(find_dep PACKAGE_NAME TARGET_NAME FETCH_CMAKE_FILE)
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UP)
  set(USE_FROM_SYSTEM_OPTION "USE_SYSTEM_${PACKAGE_NAME_UP}")
  option(${USE_FROM_SYSTEM_OPTION} "Use system ${PACKAGE_NAME}" OFF)
  if(${${USE_FROM_SYSTEM_OPTION}})
    message(STATUS "Searching for ${PACKAGE_NAME} in system")
    find_package(${PACKAGE_NAME} QUIET)
    if(TARGET ${TARGET_NAME})
      message(STATUS "${PROJECT_NAME}: Found system ${PACKAGE_NAME}: ${TARGET_NAME}")
      return()
    else()
      message(WARNING "${PROJECT_NAME}: ${PACKAGE_NAME} not found in system, falling back to FetchContent")
    endif()
  endif()
  if(NOT TARGET ${TARGET_NAME})
    message(STATUS "${PROJECT_NAME}: Fetching ${PACKAGE_NAME} via FetchContent (${FETCH_CMAKE_FILE})")
    include(${FETCH_CMAKE_FILE})
  endif()
endfunction()

option(USE_SYSTEM_EIGEN3 "${PROJECT_NAME}: Use system Eigen3" ON)
find_dep("Eigen3" "Eigen3::Eigen"
  "${CMAKE_CURRENT_LIST_DIR}/dependencies/eigen/eigen.cmake")

option(USE_SYSTEM_PYBIND11 "${PROJECT_NAME}: Use system pybind11" ON)
find_dep("pybind11" "pybind11::embed"
  "${CMAKE_CURRENT_LIST_DIR}/dependencies/pybind11/pybind11.cmake")

option(USE_SYSTEM_TSL_ROBIN_MAP "${PROJECT_NAME}: Use sytem tsl-robin-map" ON)
find_dep("tsl-robin-map" "tsl::robin_map" 
  "${CMAKE_CURRENT_LIST_DIR}/dependencies/tsl_robin/tsl_robin.cmake")
