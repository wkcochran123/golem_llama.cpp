
#include "concept_trace.h"

ggml_tensor * ConceptTrace::concept = nullptr;
bool ConceptTrace::wrote_header = false;

void ConceptTrace::capture(ggml_tensor * t) {
    concept = t;
}

void ConceptTrace::dump(const std::string & token) {
    if (!concept || !concept->data) return;

    const float * data = (const float *) concept->data;
    int dim = concept->ne[0];

    FILE *f = fopen("/Users/williamcochran/Code/golem/concept_nodes.log", "a");

    if (!wrote_header) {
        fprintf(f, "# token");
        for (int i = 0; i < dim; ++i) {
            fprintf(f, " c%d", i);
        }
        fprintf(f, "\n");
        wrote_header = true;
    }

    fprintf(f, "%s", token.c_str());
    for (int i = 0; i < dim; ++i) {
        fprintf(f, " %.5f", data[i]);
    }
    fprintf(f, "\n");

    fclose(f);
}

#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>  // for unlink


std::vector<float> ConceptTrace::wait_for_logits_file(const std::string& path) {
    // Wait for file to appear
    while (access(path.c_str(), F_OK) != 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Open and read as float32 binary
    std::ifstream infile(path, std::ios::binary | std::ios::ate);
    std::streamsize size = infile.tellg();
    infile.seekg(0, std::ios::beg);

    std::vector<float> logits(size / sizeof(float));
    if (!infile.read(reinterpret_cast<char*>(logits.data()), size)) {
        throw std::runtime_error("Failed to read logits file");
    }

    infile.close();
    unlink(path.c_str());

    return logits;
}
