#pragma once
#include <cstdio>
#include <string>

#include "llama.h"
#include "llama-context.h"

struct ConceptTrace {
    static ggml_tensor * concept;
    static bool wrote_header;

    static void capture(ggml_tensor * t);
    static void dump(const std::string & token);
	static void wait_for_logits_file(std::vector<float> &logits, const std::string& path = "/Users/williamcochran/Code/golem/python/logits.data");
};


