# Copyright (c) OpenMMLab. All rights reserved.

cmake_minimum_required(VERSION 3.8)

find_package(CUDAToolkit REQUIRED)

add_library(medusa STATIC
        medusa_weight.cc
        res_block.cc
        medusa_head.cc
        medusa_util.cc)

set_property(TARGET medusa PROPERTY POSITION_INDEPENDENT_CODE ON)
