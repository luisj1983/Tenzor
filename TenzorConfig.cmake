
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was TenzorConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(TENZOR_VERSION "1.0.0")

# Backend availability flags (set at configure time)
set(Tenzor_CUDA_FOUND ON)
set(Tenzor_ROCM_FOUND ON)
set(Tenzor_ONEAPI_FOUND ON)
set(Tenzor_VULKAN_FOUND ON)

# Find required dependencies
include(CMakeFindDependencyMacro)
find_dependency(Threads)
find_dependency(OpenMP)

# Model Hub dependencies (CURL + OpenSSL are linked PUBLIC when enabled)
if(ON)
    find_dependency(CURL)
    find_dependency(OpenSSL)
endif()

# Component checking: allow find_package(Tenzor COMPONENTS CUDA Vulkan ...)
foreach(_comp ${Tenzor_FIND_COMPONENTS})
    if(NOT Tenzor_${_comp}_FOUND)
        set(Tenzor_FOUND FALSE)
        set(Tenzor_NOT_FOUND_MESSAGE "Required component '${_comp}' not available")
    endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/TenzorTargets.cmake")

check_required_components(Tenzor)
