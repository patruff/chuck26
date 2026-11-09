// Run the frozen GGUF on teacher-forced completions or greedy generation probes.
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void write_int(std::ofstream &out, int32_t value) {
    out.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

int main(int argc, char **argv) try {
    check(argc == 6 || argc == 7, "usage: forward MODEL ADAPTER-or-dash MODE PLANS OUTPUT [CONTEXT]");
    const bool features = std::string(argv[3]) == "features";
    check(features || std::string(argv[3]) == "greedy", "unknown mode");

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(argv[1], mp);
    check(model != nullptr, "GGUF load failed");
    auto cp = llama_context_default_params();
    const int context = argc == 7 ? std::stoi(argv[6]) : 768;
    check(context > 0 && context <= llama_model_n_ctx_train(model), "prompt exceeds model context");
    cp.n_ctx = context;
    cp.n_batch = cp.n_ubatch = 256;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.embeddings = features;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    auto *ctx = llama_init_from_model(model, cp);
    check(ctx != nullptr, "context creation failed");
    llama_adapter_lora *adapter = nullptr;
    if (std::string(argv[2]) != "-") {
        adapter = llama_adapter_lora_init(model, argv[2]);
        check(adapter != nullptr, "adapter load failed");
        float scale = 1.f;
        check(llama_set_adapters_lora(ctx, &adapter, 1, &scale) == 0, "adapter attach failed");
    }
    const auto *vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int hidden_size = llama_model_n_embd_out(model);
    check(hidden_size == llama_model_n_embd(model), "unexpected projected embedding width");
    std::ifstream plans(argv[4]);
    std::ofstream output(argv[5], std::ios::binary);
    check(plans.good() && output.good(), "cannot open input or output file");
    write_int(output, n_vocab);
    write_int(output, hidden_size);
    int size, prefix, limit, row = 0;
    auto batch = llama_batch_init(cp.n_batch, 0, 1);
    while (plans >> size >> prefix >> limit) {
        check(size > 0 && prefix > 0 && prefix <= size && limit >= 0 && int64_t(size) + limit <= context, "invalid sequence length");
        std::vector<llama_token> ids(size);
        for (auto &id : ids) {
            check(bool(plans >> id) && id >= 0 && id < n_vocab, "bad token ID");
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        for (int start = 0; start < size; start += cp.n_batch) {
            batch.n_tokens = std::min(int(cp.n_batch), size - start);
            for (int i = 0; i < batch.n_tokens; ++i) {
                const int pos = start + i;
                batch.token[i] = ids[pos];
                batch.pos[i] = pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = features ? pos >= prefix - 1 && pos < size - 1 : pos == size - 1;
            }
            check(llama_decode(ctx, batch) == 0, "prefill failed");
            if (features) for (int i = 0; i < batch.n_tokens; ++i) {
                if (!batch.logits[i]) continue;
                auto *logits = llama_get_logits_ith(ctx, i);
                auto *hidden = llama_get_embeddings_ith(ctx, i);
                check(logits && hidden, "missing logits or final normalized hidden states");
                write_int(output, ids[start + i + 1]);
                output.write(reinterpret_cast<const char *>(hidden), hidden_size * sizeof(float));
                output.write(reinterpret_cast<const char *>(logits), n_vocab * sizeof(float));
            }
        }
        if (!features) {
            std::vector<int32_t> emitted;
            for (int step = 0; step < limit; ++step) {
                auto *logits = llama_get_logits_ith(ctx, -1);
                check(logits != nullptr, "missing generation logits");
                llama_token next = std::max_element(logits, logits + n_vocab) - logits;
                if (llama_vocab_is_eog(vocab, next)) break;
                emitted.push_back(next);
                if (step + 1 == limit) break;
                batch.n_tokens = 1;
                batch.token[0] = next;
                batch.pos[0] = size + step;
                batch.logits[0] = true;
                check(llama_decode(ctx, batch) == 0, "decode failed");
            }
            write_int(output, emitted.size());
            for (int id : emitted) write_int(output, id);
        }
        check(output.good(), "could not write inference results");
        std::cout << "Completed " << argv[3] << " sequence " << ++row << std::endl;
    }
    check(plans.eof() && row > 0, "malformed or empty plan");
    llama_batch_free(batch);
    llama_free(ctx);
    if (adapter) llama_adapter_lora_free(adapter);
    llama_model_free(model);
    llama_backend_free();
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
}
