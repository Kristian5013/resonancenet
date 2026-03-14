// Copyright (c) 2024-present ResonanceNet developers
// Distributed under the MIT software license, see the accompanying
// file COPYING or https://opensource.org/licenses/MIT.

// Own header.
#include "training/tokenizer.h"

// Standard library.
#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>

namespace rnet::training {

namespace {

// ---------------------------------------------------------------------------
// parse_vocab_json
// ---------------------------------------------------------------------------
// Simple JSON string parser -- extracts key-value pairs from a flat
// {"key": int, ...} object.  Sufficient for GPT-2 vocab.json.
// ---------------------------------------------------------------------------
bool parse_vocab_json(const std::string& json,
                      std::unordered_map<std::string, int>& out)
{
    size_t pos = json.find('{');
    if (pos == std::string::npos) return false;
    ++pos;

    while (pos < json.size()) {
        // 1. Skip whitespace and commas.
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' ||
               json[pos] == '\r' || json[pos] == '\t' || json[pos] == ',')) {
            ++pos;
        }
        if (pos >= json.size() || json[pos] == '}') break;

        // 2. Expect quoted key.
        if (json[pos] != '"') return false;
        ++pos;
        std::string key;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                switch (json[pos]) {
                    case '"':  key += '"'; break;
                    case '\\': key += '\\'; break;
                    case '/':  key += '/'; break;
                    case 'n':  key += '\n'; break;
                    case 'r':  key += '\r'; break;
                    case 't':  key += '\t'; break;
                    case 'u': {
                        // Parse 4-hex-digit unicode escape.
                        if (pos + 4 < json.size()) {
                            std::string hex = json.substr(pos + 1, 4);
                            unsigned long cp = std::stoul(hex, nullptr, 16);
                            // Simple UTF-8 encoding for BMP characters.
                            if (cp < 0x80) {
                                key += static_cast<char>(cp);
                            } else if (cp < 0x800) {
                                key += static_cast<char>(0xC0 | (cp >> 6));
                                key += static_cast<char>(0x80 | (cp & 0x3F));
                            } else {
                                key += static_cast<char>(0xE0 | (cp >> 12));
                                key += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                                key += static_cast<char>(0x80 | (cp & 0x3F));
                            }
                            pos += 4;
                        }
                        break;
                    }
                    default: key += json[pos]; break;
                }
            } else {
                key += json[pos];
            }
            ++pos;
        }
        if (pos >= json.size()) return false;
        ++pos; // skip closing quote

        // 3. Skip colon and whitespace.
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) ++pos;

        // 4. Parse integer value.
        std::string num_str;
        bool negative = false;
        if (pos < json.size() && json[pos] == '-') {
            negative = true;
            ++pos;
        }
        while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
            num_str += json[pos];
            ++pos;
        }
        if (num_str.empty()) return false;
        int val = std::stoi(num_str);
        if (negative) val = -val;

        out[key] = val;
    }
    return true;
}

} // namespace

// ===========================================================================
//  Byte Mappings
// ===========================================================================

// ---------------------------------------------------------------------------
// init_byte_mappings
// ---------------------------------------------------------------------------
// Builds the GPT-2 byte-to-unicode mapping.  Printable ASCII and Latin-1
// supplement get identity mapping; remaining bytes are offset to Unicode
// 256+.
// ---------------------------------------------------------------------------
void Tokenizer::init_byte_mappings()
{
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool is_printable = (b >= 33 && b <= 126) ||
                            (b >= 161 && b <= 172) ||
                            (b >= 174 && b <= 255);
        if (is_printable) {
            // Identity mapping -- represent as UTF-8.
            if (b < 128) {
                byte_encoder_[b] = std::string(1, static_cast<char>(b));
            } else if (b < 0x800) {
                byte_encoder_[b] = std::string{
                    static_cast<char>(0xC0 | (b >> 6)),
                    static_cast<char>(0x80 | (b & 0x3F))};
            }
        } else {
            // Map to Unicode 256 + n.
            int cp = 256 + n;
            ++n;
            if (cp < 0x800) {
                byte_encoder_[b] = std::string{
                    static_cast<char>(0xC0 | (cp >> 6)),
                    static_cast<char>(0x80 | (cp & 0x3F))};
            } else {
                byte_encoder_[b] = std::string{
                    static_cast<char>(0xE0 | (cp >> 12)),
                    static_cast<char>(0x80 | ((cp >> 6) & 0x3F)),
                    static_cast<char>(0x80 | (cp & 0x3F))};
            }
        }
    }

    // Build reverse mapping.
    for (int b = 0; b < 256; ++b) {
        byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
    }
}

// ===========================================================================
//  Loading
// ===========================================================================

// ---------------------------------------------------------------------------
// load
// ---------------------------------------------------------------------------
// Reads vocab.json and merges.txt from the given directory to populate the
// encoder, decoder, and BPE rank tables.
// ---------------------------------------------------------------------------
Result<void> Tokenizer::load(const std::filesystem::path& vocab_path)
{
    init_byte_mappings();

    // 1. Load vocab.json.
    auto vocab_file = vocab_path / "vocab.json";
    std::ifstream vf(vocab_file);
    if (!vf.is_open()) {
        // Try treating vocab_path as the file itself.
        vf.open(vocab_path);
        if (!vf.is_open()) {
            return Result<void>::err("Failed to open vocab file: " + vocab_file.string());
        }
    }

    std::string vocab_json((std::istreambuf_iterator<char>(vf)),
                            std::istreambuf_iterator<char>());
    vf.close();

    if (!parse_vocab_json(vocab_json, encoder_)) {
        return Result<void>::err("Failed to parse vocab.json");
    }

    // 2. Build decoder from encoder.
    decoder_.resize(VOCAB_SIZE);
    for (const auto& [token, id] : encoder_) {
        if (id >= 0 && id < VOCAB_SIZE) {
            decoder_[id] = token;
        }
    }

    // 3. Load merges.txt.
    auto merges_file = vocab_path / "merges.txt";
    std::ifstream mf(merges_file);
    if (!mf.is_open()) {
        return Result<void>::err("Failed to open merges file: " + merges_file.string());
    }

    std::string line;
    int rank = 0;
    // Skip the first line (header: "#version: 0.2").
    if (std::getline(mf, line)) {
        // Check if it is actually a merge line (no # prefix).
        if (!line.empty() && line[0] != '#') {
            bpe_ranks_[line] = rank++;
        }
    }

    while (std::getline(mf, line)) {
        if (line.empty()) continue;
        bpe_ranks_[line] = rank++;
    }
    mf.close();

    loaded_ = true;
    return Result<void>::ok();
}

// ===========================================================================
//  BPE
// ===========================================================================

// ---------------------------------------------------------------------------
// bpe
// ---------------------------------------------------------------------------
// Applies byte-pair encoding to a single word by iteratively merging the
// lowest-ranked adjacent pair until no more merges are available.
// ---------------------------------------------------------------------------
std::vector<std::string> Tokenizer::bpe(const std::string& word) const
{
    // 1. Split word into individual UTF-8 characters.
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < word.size()) {
        uint8_t c = static_cast<uint8_t>(word[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        if (i + char_len > word.size()) char_len = 1;
        tokens.push_back(word.substr(i, char_len));
        i += char_len;
    }

    if (tokens.size() <= 1) return tokens;

    // 2. Iteratively apply merges.
    while (true) {
        // Find the pair with lowest rank.
        int best_rank = std::numeric_limits<int>::max();
        size_t best_idx = std::string::npos;

        for (size_t j = 0; j + 1 < tokens.size(); ++j) {
            std::string pair = tokens[j] + " " + tokens[j + 1];
            auto it = bpe_ranks_.find(pair);
            if (it != bpe_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = j;
            }
        }

        if (best_idx == std::string::npos) break;

        // Merge the pair.
        std::string merged = tokens[best_idx] + tokens[best_idx + 1];
        std::vector<std::string> new_tokens;
        new_tokens.reserve(tokens.size() - 1);
        for (size_t j = 0; j < tokens.size(); ++j) {
            if (j == best_idx) {
                new_tokens.push_back(merged);
                ++j; // skip next
            } else {
                new_tokens.push_back(tokens[j]);
            }
        }
        tokens = std::move(new_tokens);

        if (tokens.size() <= 1) break;
    }

    return tokens;
}

// ===========================================================================
//  Encode / Decode
// ===========================================================================

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------
// Tokenises text using GPT-2-style pre-tokenisation (word-level split on
// spaces/punctuation) followed by BPE.
// ---------------------------------------------------------------------------
std::vector<int> Tokenizer::encode(std::string_view text) const
{
    if (!loaded_) return {};

    std::vector<int> result;

    // Simple word-level pre-tokenization: split on spaces and punctuation,
    // similar to GPT-2's regex pattern.
    size_t i = 0;
    while (i < text.size()) {
        // 1. Collect a word.
        std::string word;
        if (text[i] == ' ' && i + 1 < text.size() &&
            (std::isalpha(static_cast<unsigned char>(text[i + 1])) ||
             std::isdigit(static_cast<unsigned char>(text[i + 1])))) {
            // Space followed by alphanumeric -- include the space as a prefix.
            word += byte_encoder_[static_cast<uint8_t>(text[i])];
            ++i;
            while (i < text.size() &&
                   (std::isalpha(static_cast<unsigned char>(text[i])) ||
                    std::isdigit(static_cast<unsigned char>(text[i])))) {
                word += byte_encoder_[static_cast<uint8_t>(text[i])];
                ++i;
            }
        } else if (std::isalpha(static_cast<unsigned char>(text[i])) ||
                   std::isdigit(static_cast<unsigned char>(text[i]))) {
            while (i < text.size() &&
                   (std::isalpha(static_cast<unsigned char>(text[i])) ||
                    std::isdigit(static_cast<unsigned char>(text[i])))) {
                word += byte_encoder_[static_cast<uint8_t>(text[i])];
                ++i;
            }
        } else {
            // Single character token (punctuation, space, etc.).
            word += byte_encoder_[static_cast<uint8_t>(text[i])];
            ++i;
        }

        // 2. Apply BPE.
        auto bpe_tokens = bpe(word);

        // 3. Look up token IDs.
        for (const auto& tok : bpe_tokens) {
            auto it = encoder_.find(tok);
            if (it != encoder_.end()) {
                result.push_back(it->second);
            }
            // Unknown tokens are silently dropped.
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
// decode
// ---------------------------------------------------------------------------
// Converts token IDs back to text by concatenating token strings and then
// mapping Unicode characters back to raw bytes.
// ---------------------------------------------------------------------------
std::string Tokenizer::decode(std::span<const int> tokens) const
{
    if (!loaded_) return {};

    // 1. Concatenate token strings.
    std::string unicode_text;
    for (int id : tokens) {
        if (id >= 0 && id < static_cast<int>(decoder_.size())) {
            unicode_text += decoder_[id];
        }
    }

    // 2. Convert Unicode characters back to bytes.
    std::string result;
    size_t i = 0;
    while (i < unicode_text.size()) {
        // Extract one UTF-8 character.
        uint8_t c = static_cast<uint8_t>(unicode_text[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        if (i + char_len > unicode_text.size()) {
            // Malformed -- just copy the byte.
            result += static_cast<char>(c);
            ++i;
            continue;
        }

        std::string ch = unicode_text.substr(i, char_len);
        auto it = byte_decoder_.find(ch);
        if (it != byte_decoder_.end()) {
            result += static_cast<char>(it->second);
        } else {
            // Not in byte decoder -- pass through.
            result += ch;
        }
        i += char_len;
    }

    return result;
}

} // namespace rnet::training
