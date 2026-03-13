// Usage: prepare_dataset <input.txt> <output.bin>
// Reads a text file, converts each byte to an int32 token, writes binary output.
// Character-level tokenization: vocab_size = 256 (each byte = one token).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.txt> <output.bin>\n", argv[0]);
        return 1;
    }

    // Read input text file
    FILE* fin = fopen(argv[1], "rb");
    if (!fin) {
        fprintf(stderr, "Error: cannot open input file: %s\n", argv[1]);
        return 1;
    }

    fseek(fin, 0, SEEK_END);
    long file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);

    std::vector<uint8_t> text(file_size);
    fread(text.data(), 1, file_size, fin);
    fclose(fin);

    // Convert to int32 tokens (character-level)
    std::vector<int32_t> tokens(file_size);
    for (long i = 0; i < file_size; ++i) {
        tokens[i] = static_cast<int32_t>(text[i]);
    }

    // Write binary output
    FILE* fout = fopen(argv[2], "wb");
    if (!fout) {
        fprintf(stderr, "Error: cannot open output file: %s\n", argv[2]);
        return 1;
    }
    fwrite(tokens.data(), sizeof(int32_t), tokens.size(), fout);
    fclose(fout);

    printf("Converted %ld characters to %zu tokens\n", file_size, tokens.size());
    printf("Output: %s (%zu bytes)\n", argv[2], tokens.size() * sizeof(int32_t));
    return 0;
}
