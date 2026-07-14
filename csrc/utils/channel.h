// Copyright 2024 Tencent Inc.  All rights reserved.

#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>

#include "utils.h"

namespace ksana_llm {

constexpr size_t kChannelDefaultCapacity = 1ul << 48;

// ChannelObject is the underlying type of Channel
template <typename T>
class ChannelObject {
 public:
  DELETE_COPY_AND_MOVE(ChannelObject);

  ChannelObject(size_t capacity, std::string_view name) : capacity_(capacity), name_(name) {}

  // close
  bool IsClosed() { return is_closed_; }  // lock mutex_ is unnecessary
  void Close() {
    std::unique_lock<std::mutex> lck(mutex_);
    is_closed_ = true;
    Notify();
  }

  // see Channel::Read
  size_t Read(T* dst, size_t expected_read_num) {
    if (expected_read_num == 0ul) {
      return 0ul;
    }
    size_t readed_count = 0;
    std::unique_lock<std::mutex> lck(mutex_);
    reading_elem_cnt_ += expected_read_num;
    while (readed_count < expected_read_num && WaitUntilReadReady(lck)) {
      size_t n = std::min(expected_read_num - readed_count, data_.size());
      FOR_RANGE(size_t, i, 0ul, n) {
        dst[readed_count++] = std::move(data_.front());
        data_.pop_front();
      }
      reading_elem_cnt_ -= n;
    }
    reading_elem_cnt_ -= expected_read_num - readed_count;  // unread
    Notify();
    return readed_count;
  }

  // see Channel::WriteMovable
  size_t WriteMovable(T* src, size_t expected_write_num) {
    return Write(expected_write_num, [this, src](size_t i) { data_.push_back(std::move(src[i])); });
  }

  // see Channel::WriteCopyable
  size_t WriteCopyable(const T* src, size_t expected_write_num) {
    return Write(expected_write_num, [this, src](size_t i) { data_.push_back(src[i]); });
  }

  // see Channel::PrintWaitRate
  void PrintWaitRate() {
    std::unique_lock<std::mutex> lck(mutex_);
    // TODO(karlluo): do something print

    read_cnt_ = 0;
    write_cnt_ = 0;
    sum_chan_size_reading_ = 0;
    sum_chan_size_writing_ = 0;
    reader_wait_cnt_ = 0;
    writer_wait_cnt_ = 0;
  }

  // see Channel::Size
  size_t Size() {
    std::unique_lock<std::mutex> lck(mutex_);
    return data_.size();
  }

 private:
  size_t Write(size_t expected_write_num, std::function<void(size_t)> push) {
    if (expected_write_num == 0ul) {
      return 0ul;
    }
    size_t written_count = 0;
    std::unique_lock<std::mutex> lck(mutex_);
    while (written_count < expected_write_num && WaitUntilWriteReady(lck)) {
      size_t n = std::min(expected_write_num - written_count, capacity_ + reading_elem_cnt_ - data_.size());
      FOR_RANGE(size_t, i, 0ul, n) { push(written_count++); }
    }
    Notify();
    return written_count;
  }

  bool WaitUntilReadReady(std::unique_lock<std::mutex>& lck) {
    read_cnt_ += 1ul;
    sum_chan_size_reading_ += data_.size();
    bool need_add_wait_cnt = true;
    while (data_.empty() && !is_closed_) {
      if (waiting_writer_cnt_ != 0) {
        write_cond_.notify_one();
      }
      waiting_reader_cnt_ += 1ul;
      read_cond_.wait(lck);
      waiting_reader_cnt_ -= 1ul;
      if (need_add_wait_cnt) {
        reader_wait_cnt_ += 1ul;
        need_add_wait_cnt = false;
      }
    }
    return !data_.empty();
  }

  bool WaitUntilWriteReady(std::unique_lock<std::mutex>& lck) {
    write_cnt_ += 1ul;
    sum_chan_size_writing_ += data_.size();
    bool need_add_wait_cnt = true;
    while (data_.size() >= capacity_ + reading_elem_cnt_ && !is_closed_) {
      if (waiting_reader_cnt_ != 0) {
        read_cond_.notify_one();
      }
      waiting_writer_cnt_ += 1ul;
      write_cond_.wait(lck);
      waiting_writer_cnt_ -= 1ul;
      if (need_add_wait_cnt) {
        writer_wait_cnt_ += 1ul;
        need_add_wait_cnt = false;
      }
    }
    return data_.size() < capacity_ + reading_elem_cnt_;
  }

  void Notify() {
    if (waiting_reader_cnt_ > 0 && (!data_.empty() || is_closed_)) {
      read_cond_.notify_one();
    }
    if (waiting_writer_cnt_ > 0 && (data_.size() < capacity_ + reading_elem_cnt_ || is_closed_)) {
      write_cond_.notify_one();
    }
  }

  std::deque<T> data_;
  size_t capacity_;
  std::string name_;
  bool is_closed_{false};

  std::mutex mutex_;
  std::condition_variable read_cond_;
  std::condition_variable write_cond_;

  size_t reading_elem_cnt_{0};
  int32_t waiting_reader_cnt_{0};
  int32_t waiting_writer_cnt_{0};

  size_t read_cnt_{0};
  size_t sum_chan_size_reading_{0};
  size_t reader_wait_cnt_{0};

  size_t write_cnt_{0};
  size_t sum_chan_size_writing_{0};
  size_t writer_wait_cnt_{0};
};

// like copyable chan in go
template <typename T>
class Channel {
 public:
  Channel() : Channel(kChannelDefaultCapacity, "") {}
  explicit Channel(size_t capacity) : Channel(capacity, "") {}
  explicit Channel(std::string_view name) : Channel(kChannelDefaultCapacity, name) {}
  Channel(size_t capacity, std::string_view name) { channel_obj_.reset(new ChannelObject<T>(capacity, name)); }
  // close
  void Close() { channel_obj_->Close(); }
  bool IsClosed() { return channel_obj_->IsClosed(); }

  // read n elements into the buffer starting at x, these methods will block the
  // current thread until the channel is closed if there is no enough elements
  // in this channel currently. The number of elements read is returned.
  bool Read(T* x) { return Read(x, 1ul) == 1ul; }
  size_t Read(T* x, size_t n) { return channel_obj_->Read(x, n); }

  // write n elements from the buffer starting at x, these methods will block
  // the current thread util the channel is closed if there is no enough
  // capacity currently. The number of elements written is returned.
  bool Write(const T& x) { return Write(&x, 1ul) == 1ul; }
  bool Write(T&& x) { return WriteMovable(&x, 1ul) == 1ul; }
  size_t Write(const T* x, size_t n) { return channel_obj_->WriteCopyable(x, n); }
  size_t WriteMovable(T* x, size_t n) { return channel_obj_->WriteMovable(x, n); }

  // read_wait_rate: rate of waiting on reading
  // write_wait_rate: rate of waiting on writing
  // read_avg_realtime_size: average size of this channel on reading
  // write_avg_realtime_size: average size of this channel on writing
  void PrintWaitRate() { return channel_obj_->PrintWaitRate(); }

  size_t Size() { return channel_obj_->Size(); }

 private:
  std::shared_ptr<ChannelObject<T>> channel_obj_;
};

}  // namespace ksana_llm
