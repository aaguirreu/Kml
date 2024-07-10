#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <ranges>

namespace py = pybind11;
namespace ra = std::ranges;
namespace rv = ra::views;

auto sec2vec(const std::string& sequence, int k) -> std::unordered_map<std::string, int> {
    auto subseqs = sequence | rv::slide(k);
    std::unordered_map<std::string, int> kmer_counts;
    for (const auto& subseq : subseqs) {
        std::string kmer(subseq.begin(), subseq.end());
        kmer_counts[kmer]++;
    }
    return kmer_counts;
}

PYBIND11_MODULE(sec2vec_module, m) {
    m.def("sec2vec", &sec2vec, "K-mer from DNA sequence");
}
