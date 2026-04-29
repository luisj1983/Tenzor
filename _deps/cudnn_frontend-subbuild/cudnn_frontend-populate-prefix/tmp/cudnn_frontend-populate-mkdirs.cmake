# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-src")
  file(MAKE_DIRECTORY "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-src")
endif()
file(MAKE_DIRECTORY
  "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-build"
  "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix"
  "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix/tmp"
  "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix/src/cudnn_frontend-populate-stamp"
  "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix/src"
  "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix/src/cudnn_frontend-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix/src/cudnn_frontend-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/lee/Projects/Tenzor/_deps/cudnn_frontend-subbuild/cudnn_frontend-populate-prefix/src/cudnn_frontend-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
