# should handle find_package or fetch content make availabe all necessary
# dependecies
function(
  find_dep
  PACKAGE_NAME
  TARGET_NAME
  FETCH_CMAKE_FILE)
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UP)
  set(USE_FROM_SYSTEM_OPTION "CT_LIO_USE_SYSTEM_${PACKAGE_NAME_UP}")
  if(${${USE_FROM_SYSTEM_OPTION}})
    message(STATUS "Searching for ${PACKAGE_NAME} in system. ")
    find_package(${PACKAGE_NAME} QUIET)
    if(NOT TARGET ${TARGET_NAME})
      message(STATUS "${PACKAGE_NAME}'s target: ${TARGET_NAME} not found.")
    endif()
  endif()
  if(NOT TARGET ${TARGET_NAME})
    message(STATUS "Fetching ${PACKAGE_NAME}: ${TARGET_NAME} using FetchContent (${FETCH_CMAKE_FILE})")
    include(${FETCH_CMAKE_FILE})
  endif()
endfunction()

include(FetchContent)

option(CT_LIO_USE_SYSTEM_EIGEN3 "CT LIO: Use system Eigen" OFF)
find_dep("Eigen3" "Eigen3::Eigen" "${CMAKE_CURRENT_LIST_DIR}/dependencies/eigen/eigen.cmake")

option(CT_LIO_USE_SYSTEM_PYBIND11 "CT LIO: Use system pybind11" OFF)
find_dep("pybind11" "pybind11::embed" "${CMAKE_CURRENT_LIST_DIR}/dependencies/pybind11/pybind11.cmake")
