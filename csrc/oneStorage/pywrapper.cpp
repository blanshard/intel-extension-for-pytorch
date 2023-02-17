#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <string>
#include <vector>
#include "io.h"

namespace {
namespace py = pybind11;
using oneStorage::oneFile;

PYBIND11_MODULE(_pywrap_oneFile, m) {
    py::class_<oneFile>(m, "oneFile")
        .def(py::init<>())
        .def("read",
             [](oneFile* self, const std::string& filename) {
                 std::string result;
                 self->read(filename, &result);
                 return py::bytes(result);
             })
        .def("list_files",
             [](oneFile* self, const std::string& path) {
                 std::vector<std::string> filenames;
                 self->list_files(path, &filenames);
                 return filenames;
             })
        .def("get_file_size",
             [](oneFile* self, const std::string& filename) {
                 return self->get_file_size(filename);
        });
}
}  // namespace