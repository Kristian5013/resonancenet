// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Standard library.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// ===========================================================================
//  prepare_dataset — converts text to binary tokenized format
// ===========================================================================

// ---------------------------------------------------------------------------
// print_usage
// ---------------------------------------------------------------------------
static void print_usage(const char* prog)
{
    fprintf(stderr,
        "Usage: %s <input.txt> <train.bin> <val.bin> [val_ratio]\n"
        "\n"
        "Reads a text file, tokenizes with byte-level encoding,\n"
        "splits into train/val, and writes binary uint16 token files.\n"
        "\n"
        "Arguments:\n"
        "  input.txt     Input text corpus\n"
        "  train.bin     Output training tokens (binary uint16)\n"
        "  val.bin       Output validation tokens (binary uint16)\n"
        "  val_ratio     Fraction for validation (default: 0.05 = 5%%)\n"
        "\n"
        "Single output mode:\n"
        "  %s <input.txt> <output.bin>\n"
        "  Converts entire file without splitting.\n",
        prog, prog);
}

// ---------------------------------------------------------------------------
// read_text_file
// ---------------------------------------------------------------------------
static std::string read_text_file(const char* path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) return "";
    in.seekg(0, std::ios::end);
    auto size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::string text(static_cast<size_t>(size), '\0');
    in.read(text.data(), size);
    return text;
}

// ---------------------------------------------------------------------------
// write_tokens
// ---------------------------------------------------------------------------
static bool write_tokens(const char* path, const uint16_t* data, size_t count)
{
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: cannot open output file: %s\n", path);
        return false;
    }
    fwrite(data, sizeof(uint16_t), count, f);
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    // 1. Parse arguments.
    if (argc < 3 || argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "--help") == 0) {
        print_usage(argv[0]);
        return 0;
    }

    const char* input_path = argv[1];
    bool split_mode = (argc >= 4);
    const char* train_path = split_mode ? argv[2] : argv[2];
    const char* val_path   = split_mode ? argv[3] : nullptr;
    float val_ratio = 0.05f;
    if (argc == 5) {
        val_ratio = static_cast<float>(std::atof(argv[4]));
    }

    // 2. Read input text.
    printf("Reading %s...\n", input_path);
    std::string text = read_text_file(input_path);
    if (text.empty()) {
        fprintf(stderr, "Error: cannot read or empty file: %s\n", input_path);
        return 1;
    }
    printf("Read %zu bytes (%.1f MB)\n", text.size(),
           static_cast<double>(text.size()) / (1024.0 * 1024.0));

    // 3. Tokenize: byte-level encoding (each byte = one token).
    //    Vocab size = 256, fits in uint16.  This is simple but effective
    //    for initial training.  GPT-2 BPE tokenizer can be added later.
    printf("Tokenizing (byte-level, vocab=256)...\n");
    std::vector<uint16_t> tokens(text.size());
    for (size_t i = 0; i < text.size(); ++i) {
        tokens[i] = static_cast<uint16_t>(static_cast<uint8_t>(text[i]));
    }
    printf("Total tokens: %zu\n", tokens.size());

    // 4. Create output directories if needed.
    if (split_mode) {
        auto train_dir = std::filesystem::path(train_path).parent_path();
        auto val_dir   = std::filesystem::path(val_path).parent_path();
        if (!train_dir.empty()) std::filesystem::create_directories(train_dir);
        if (!val_dir.empty())   std::filesystem::create_directories(val_dir);
    } else {
        auto out_dir = std::filesystem::path(train_path).parent_path();
        if (!out_dir.empty()) std::filesystem::create_directories(out_dir);
    }

    // 5. Write output.
    if (split_mode) {
        size_t val_count = static_cast<size_t>(
            static_cast<float>(tokens.size()) * val_ratio);
        size_t train_count = tokens.size() - val_count;

        printf("Splitting: %zu train, %zu val (%.1f%%)\n",
               train_count, val_count,
               static_cast<double>(val_ratio) * 100.0);

        if (!write_tokens(train_path, tokens.data(), train_count)) return 1;
        printf("Train: %s (%zu tokens, %.1f MB)\n", train_path, train_count,
               static_cast<double>(train_count * 2) / (1024.0 * 1024.0));

        if (!write_tokens(val_path, tokens.data() + train_count, val_count)) return 1;
        printf("Val:   %s (%zu tokens, %.1f MB)\n", val_path, val_count,
               static_cast<double>(val_count * 2) / (1024.0 * 1024.0));
    } else {
        if (!write_tokens(train_path, tokens.data(), tokens.size())) return 1;
        printf("Output: %s (%zu tokens, %.1f MB)\n", train_path, tokens.size(),
               static_cast<double>(tokens.size() * 2) / (1024.0 * 1024.0));
    }

    printf("Done.\n");
    return 0;
}
