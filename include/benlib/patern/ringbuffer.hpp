/**
 * @file ringbuffer.hpp
 * @author Bensuperpc (bensuperpc@gmail.com)
 * @brief
 * @version 1.0.0
 * @date 2024-07-06
 *
 * MIT License
 *
 */

#ifndef BENLIB_PATERN_RINGBUFFER_HPP_
#define BENLIB_PATERN_RINGBUFFER_HPP_

#include <deque>

namespace benlib {
namespace patern {

template <typename T>
class RingBuffer {
   public:
    explicit RingBuffer() = delete;
    explicit RingBuffer(size_t size) {
        _maxSize = size;
        _container = std::deque<T>(_maxSize);
    }
    virtual ~RingBuffer() { _container.clear(); }
    size_t size() const noexcept { return _container.size(); }
    void pushBack(T& element) {
        if (_container.size() >= _maxSize) {
            _container.erase(_container.begin());
        }

        _container.push_back(element);
    }
    void pushBack(T&& element) {
        if (_container.size() >= _maxSize) {
            _container.erase(_container.begin());
        }

        _container.push_back(std::forward<T>(element));
    }
    void pushFront(T& element) {
        if (_container.size() >= _maxSize) {
            _container.pop_back();
        }

        _container.push_front(element);
    }
    void pushFront(T&& element) {
        if (_container.size() >= _maxSize) {
            _container.pop_back();
        }

        _container.push_front(std::forward<T>(element));
    }
    void popFront() {
        if (_container.size() > 0) {
            _container.erase(_container.begin());
        }
    }
    void popBack() {
        if (_container.size() > 0) {
            _container.pop_back();
        }
    }
    void setMaxSize(size_t size) {
        if (size < _container.size()) {
            _container.erase(_container.begin(), _container.begin() + (_container.size() - size));
        }
        _maxSize = size;
    }
    size_t getMaxSize() const noexcept { return _maxSize; }
    T& operator[](size_t index) { return _container[index]; }
    void clear() { _container.clear(); }
    std::deque<T>& data() { return _container; }

   private:
    std::deque<T> _container = std::deque<T>();
    std::deque<T>::size_type _maxSize = 0;
};

}  // namespace patern
}  // namespace benlib
#endif
