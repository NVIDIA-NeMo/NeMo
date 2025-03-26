#pragma once

#include <cstdint>


void geglu_cuda(intptr_t out, intptr_t x_and_gate, int64_t n, int dim_last, intptr_t stream);

void geglu_bwd_cuda(intptr_t grad_x_and_gate, intptr_t grad_out, intptr_t x_and_gate, int64_t n, int dim_last, intptr_t stream);
