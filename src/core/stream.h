#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace rnet::core {

/// SER_ACTION constants for controlling serialize vs unserialize
enum SerAction {
    SER_NETWORK   = (1 << 0),
    SER_DISK      = (1 << 1),
    SER_GETHASH   = (1 << 2),
};

/// DataStream: in-memory byte stream for serialization.
/// Supports both reading and writing.
class DataStream {
public:
    using value_type = uint8_t;
    using size_type = size_t;
    using iterator = std::vector<uint8_t>::iterator;
    using const_iterator = std::vector<uint8_t>::const_iterator;

    DataStream() = default;

    explicit DataStream(std::span<const uint8_t> data)
        : data_(data.begin(), data.end()), read_pos_(0) {}

    explicit DataStream(std::vector<uint8_t> data)
        : data_(std::move(data)), read_pos_(0) {}

    DataStream(const DataStream&) = default;
    DataStream(DataStream&&) noexcept = default;
    DataStream& operator=(const DataStream&) = default;
    DataStream& operator=(DataStream&&) noexcept = default;

    /// Write bytes to the stream
    void write(const void* src, size_t len) {
        const auto* p = static_cast<const uint8_t*>(src);
        data_.insert(data_.end(), p, p + len);
    }

    /// Write a single byte
    void write_byte(uint8_t b) {
        data_.push_back(b);
    }

    /// Read bytes from the stream
    void read(void* dst, size_t len) {
        if (read_pos_ + len > data_.size()) {
            throw std::runtime_error(
                "DataStream::read: end of data");
        }
        std::memcpy(dst, data_.data() + read_pos_, len);
        read_pos_ += len;
    }

    /// Peek at bytes without advancing read position
    void peek(void* dst, size_t len) const {
        if (read_pos_ + len > data_.size()) {
            throw std::runtime_error(
                "DataStream::peek: end of data");
        }
        std::memcpy(dst, data_.data() + read_pos_, len);
    }

    /// Skip bytes
    void skip(size_t len) {
        if (read_pos_ + len > data_.size()) {
            throw std::runtime_error(
                "DataStream::skip: end of data");
        }
        read_pos_ += len;
    }

    /// Ignore bytes (alias for skip)
    void ignore(size_t len) { skip(len); }

    /// Check if all data has been read
    bool eof() const { return read_pos_ >= data_.size(); }

    /// Bytes remaining to read
    size_t remaining() const {
        return data_.size() - read_pos_;
    }

    /// Total size of written data
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

    /// Access underlying data
    const uint8_t* data() const { return data_.data(); }
    uint8_t* data() { return data_.data(); }

    /// Span of all data
    std::span<const uint8_t> span() const {
        return {data_.data(), data_.size()};
    }

    /// Span of unread data
    std::span<const uint8_t> unread_span() const {
        return {data_.data() + read_pos_, remaining()};
    }

    /// Vector access
    const std::vector<uint8_t>& vch() const { return data_; }
    std::vector<uint8_t>& vch() { return data_; }

    /// Iterator access
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    /// Clear all data and reset read position
    void clear() {
        data_.clear();
        read_pos_ = 0;
    }

    /// Reserve capacity
    void reserve(size_t n) { data_.reserve(n); }

    /// Rewind read position to beginning
    void rewind() { read_pos_ = 0; }

    /// Get/set read position
    size_t tell() const { return read_pos_; }
    void seek(size_t pos) { read_pos_ = pos; }

    /// Hex string of contents
    std::string to_hex() const;

    /// Compact string representation for debugging
    std::string to_string() const {
        return "[DataStream size=" + std::to_string(data_.size()) +
               " read_pos=" + std::to_string(read_pos_) + "]";
    }

    /// Serialization operators
    template<typename T>
    DataStream& operator<<(const T& obj) {
        obj.serialize(*this);
        return *this;
    }

    template<typename T>
    DataStream& operator>>(T& obj) {
        obj.unserialize(*this);
        return *this;
    }

private:
    std::vector<uint8_t> data_;
    size_t read_pos_ = 0;
};

/// Create a DataStream from a hex string
DataStream DataStream_from_hex(std::string_view hex);

/// SpanReader: read-only stream over a span of bytes.
class SpanReader {
public:
    SpanReader() = default;

    explicit SpanReader(std::span<const uint8_t> data)
        : data_(data) {}

    void read(void* dst, size_t len) {
        if (pos_ + len > data_.size()) {
            throw std::runtime_error(
                "SpanReader::read: end of data");
        }
        std::memcpy(dst, data_.data() + pos_, len);
        pos_ += len;
    }

    void skip(size_t len) {
        if (pos_ + len > data_.size()) {
            throw std::runtime_error(
                "SpanReader::skip: end of data");
        }
        pos_ += len;
    }

    void ignore(size_t len) { skip(len); }

    bool eof() const { return pos_ >= data_.size(); }
    size_t remaining() const { return data_.size() - pos_; }
    size_t size() const { return data_.size(); }
    size_t tell() const { return pos_; }
    void seek(size_t pos) { pos_ = pos; }
    void rewind() { pos_ = 0; }

    std::span<const uint8_t> unread_span() const {
        return data_.subspan(pos_);
    }

    template<typename T>
    SpanReader& operator>>(T& obj) {
        obj.unserialize(*this);
        return *this;
    }

private:
    std::span<const uint8_t> data_;
    size_t pos_ = 0;
};

/// VectorWriter: write-only stream appending to an external vector.
class VectorWriter {
public:
    explicit VectorWriter(std::vector<uint8_t>& vec)
        : vec_(vec) {}

    void write(const void* src, size_t len) {
        const auto* p = static_cast<const uint8_t*>(src);
        vec_.insert(vec_.end(), p, p + len);
    }

    void write_byte(uint8_t b) { vec_.push_back(b); }
    size_t size() const { return vec_.size(); }

    template<typename T>
    VectorWriter& operator<<(const T& obj) {
        obj.serialize(*this);
        return *this;
    }

private:
    std::vector<uint8_t>& vec_;
};

/// SizeCounter: counts bytes that would be written without storing them.
class SizeCounter {
public:
    void write(const void*, size_t len) { size_ += len; }
    void write_byte(uint8_t) { size_ += 1; }
    size_t size() const { return size_; }

    template<typename T>
    SizeCounter& operator<<(const T& obj) {
        obj.serialize(*this);
        return *this;
    }

private:
    size_t size_ = 0;
};

/// AutoFile: stream wrapper around a FILE*.
/// Owns the file handle and closes it on destruction.
class AutoFile {
public:
    AutoFile() = default;
    explicit AutoFile(FILE* file) : file_(file) {}

    ~AutoFile() { close(); }

    AutoFile(const AutoFile&) = delete;
    AutoFile& operator=(const AutoFile&) = delete;

    AutoFile(AutoFile&& other) noexcept : file_(other.file_) {
        other.file_ = nullptr;
    }

    AutoFile& operator=(AutoFile&& other) noexcept {
        if (this != &other) {
            close();
            file_ = other.file_;
            other.file_ = nullptr;
        }
        return *this;
    }

    void write(const void* src, size_t len) {
        if (!file_) {
            throw std::runtime_error("AutoFile::write: null file");
        }
        if (std::fwrite(src, 1, len, file_) != len) {
            throw std::runtime_error("AutoFile::write: write failed");
        }
    }

    void read(void* dst, size_t len) {
        if (!file_) {
            throw std::runtime_error("AutoFile::read: null file");
        }
        if (std::fread(dst, 1, len, file_) != len) {
            throw std::runtime_error("AutoFile::read: read failed");
        }
    }

    void write_byte(uint8_t b) { write(&b, 1); }

    void flush() {
        if (file_) std::fflush(file_);
    }

    void close() {
        if (file_) {
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    bool is_null() const { return file_ == nullptr; }
    FILE* get() const { return file_; }
    FILE* release() {
        FILE* f = file_;
        file_ = nullptr;
        return f;
    }

    int64_t tell() const {
        if (!file_) return -1;
#ifdef _WIN32
        return _ftelli64(file_);
#else
        return ftello(file_);
#endif
    }

    bool seek(int64_t offset, int origin = SEEK_SET) {
        if (!file_) return false;
#ifdef _WIN32
        return _fseeki64(file_, offset, origin) == 0;
#else
        return fseeko(file_, offset, origin) == 0;
#endif
    }

    template<typename T>
    AutoFile& operator<<(const T& obj) {
        obj.serialize(*this);
        return *this;
    }

    template<typename T>
    AutoFile& operator>>(T& obj) {
        obj.unserialize(*this);
        return *this;
    }

private:
    FILE* file_ = nullptr;
};

/// BufferedWriter: wraps a stream with a write buffer for performance.
template<typename Inner, size_t BUF_SIZE = 8192>
class BufferedWriter {
public:
    explicit BufferedWriter(Inner& inner)
        : inner_(inner), pos_(0) {}

    ~BufferedWriter() {
        flush();
    }

    void write(const void* src, size_t len) {
        const auto* p = static_cast<const uint8_t*>(src);
        while (len > 0) {
            size_t avail = BUF_SIZE - pos_;
            if (len <= avail) {
                std::memcpy(buf_ + pos_, p, len);
                pos_ += len;
                return;
            }
            std::memcpy(buf_ + pos_, p, avail);
            pos_ = BUF_SIZE;
            flush();
            p += avail;
            len -= avail;
        }
    }

    void write_byte(uint8_t b) {
        if (pos_ >= BUF_SIZE) flush();
        buf_[pos_++] = b;
    }

    void flush() {
        if (pos_ > 0) {
            inner_.write(buf_, pos_);
            pos_ = 0;
        }
    }

    template<typename T>
    BufferedWriter& operator<<(const T& obj) {
        obj.serialize(*this);
        return *this;
    }

private:
    Inner& inner_;
    uint8_t buf_[BUF_SIZE];
    size_t pos_;
};

/// LimitedReader: wraps a reader with a byte limit.
template<typename Inner>
class LimitedReader {
public:
    LimitedReader(Inner& inner, size_t limit)
        : inner_(inner), remaining_(limit) {}

    void read(void* dst, size_t len) {
        if (len > remaining_) {
            throw std::runtime_error(
                "LimitedReader: exceeded read limit");
        }
        inner_.read(dst, len);
        remaining_ -= len;
    }

    void skip(size_t len) {
        if (len > remaining_) {
            throw std::runtime_error(
                "LimitedReader: exceeded skip limit");
        }
        inner_.skip(len);
        remaining_ -= len;
    }

    size_t remaining() const { return remaining_; }
    bool eof() const { return remaining_ == 0; }

    template<typename T>
    LimitedReader& operator>>(T& obj) {
        obj.unserialize(*this);
        return *this;
    }

private:
    Inner& inner_;
    size_t remaining_;
};

}  // namespace rnet::core
