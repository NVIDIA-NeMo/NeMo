#include <pybind11/pybind11.h>

#include "geglu.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("geglu", &geglu_cuda, "");
    m.def("geglu_bwd", &geglu_bwd_cuda, "");
}
