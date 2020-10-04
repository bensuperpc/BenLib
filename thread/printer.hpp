#if __cplusplus >= 201703L
#    ifndef thread_printer_hpp
#        define thread_printer_hpp

#        include <cstdio>
#        include <mutex>
#        include <sstream>
#        include <string>
#        include <thread>
#        include <type_traits>

namespace thread::printer
{
namespace detail
{
extern std::mutex mutex;

void print();
void print(char character);
void print(const char *string);
void print(const std::string &string);

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value == false>::type print(const T &t);

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value == true>::type print(const T &t);

template <typename T, typename U, typename... O> void print(const T &t, const U &u, const O &... other);
} // namespace detail

template <typename T, typename... O> void print(const T &t, const O &... other);

//--

namespace detail
{
template <typename T> typename std::enable_if<std::is_arithmetic<T>::value == false>::type print(const T &t)
{
    std::ostringstream ostringstream;
    ostringstream << t;
    print(ostringstream.str());
}

template <typename T> typename std::enable_if<std::is_arithmetic<T>::value == true>::type print(const T &t)
{
    print(std::to_string(t));
}

template <typename T, typename U, typename... O> void print(const T &t, const U &u, const O &... other)
{
    print(t);
    print(u, other...);
}
} // namespace detail

template <typename T, typename... O> void print(const T &t, const O &... other)
{
    std::lock_guard<std::mutex> lockGuard(detail::mutex);
    detail::print(t);
    detail::print(other...);
    std::fflush(stdout);
}
} // namespace thread::printer
#    endif
#endif
