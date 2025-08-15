# Copyright (C) 2019-2020 IBM Corp.
# This program is Licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. See accompanying LICENSE file.

include_directories(../../libnpy/include/)
include_directories(../../json/include/)
include_directories(../../CryptoCortex/include/)

find_package(OpenMP)

# Include all the henn files
file(GLOB_RECURSE SOURCE_FILES "../../CryptoCortex/src/*.cpp")

# Get all .cpp files in model folder 
file(GLOB MODEL_SOURCES "*.cpp")

foreach(model_src ${MODEL_SOURCES})
    # Get the filename without extension
    get_filename_component(model_name ${model_src} NAME_WE)
    # Create an executable for each model source
    add_executable(${model_name} ${model_src} ${SOURCE_FILES})
    target_link_libraries(${model_name} helib)
    if(OpenMP_CXX_FOUND)
        target_compile_options(${model_name} PRIVATE -fopenmp)
        target_link_libraries(${model_name} OpenMP::OpenMP_CXX)
    endif()
endforeach()
