set(EIGEN_BUILD_DOC OFF CACHE BOOL "Don't build Eigen docs")
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Don't build Eigen tests")
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Don't build Eigen pkg-config")
set(EIGEN_BUILD_BLAS OFF CACHE BOOL "Don't build blas module")
set(EIGEN_BUILD_LAPACK OFF CACHE BOOL "Don't build lapack module")

# OVERRIDE_FIND_PACKAGE to fetch kiss ICP but set use system ON for kiss to
# override its deps
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz)
FetchContent_MakeAvailable(Eigen3)
