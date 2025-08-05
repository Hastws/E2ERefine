#ifndef INCLUDE_AUTOALG_LOG_H
#define INCLUDE_AUTOALG_LOG_H

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace autoalg {
class LogStream {
 public:
  LogStream(std::ostream &out, const std::string &level,
            const std::string &module)
      : ostream_(out) {
    // 当前时间
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now.time_since_epoch()) %
                  1000;

    // 线程 ID
    std::stringstream tid_ss;
    tid_ss << std::this_thread::get_id();

    // 格式化日志前缀
    stringstream_ << "[" << level << "] ";
    stringstream_ << "["
                  << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    stringstream_ << "." << std::setw(3) << std::setfill('0') << millis.count()
                  << "] ";
    stringstream_ << "[TID " << tid_ss.str() << "] ";
    stringstream_ << "[" << module << "] ";
  }

  ~LogStream() {
    stringstream_ << std::endl;
    ostream_ << stringstream_.str();
  }

  template <typename T>
  LogStream &operator<<(const T &val) {
    stringstream_ << val;
    return *this;
  }

 private:
  std::ostream &ostream_;
  std::stringstream stringstream_;
};
}  // namespace autoalg

#define DEBUG(module) autoalg::LogStream(std::cout, "DEBUG", #module)
#define INFO(module) autoalg::LogStream(std::cout, "INFO ", #module)
#define WARN(module) autoalg::LogStream(std::cerr, "WARN ", #module)
#define ERROR(module) autoalg::LogStream(std::cerr, "ERROR", #module)
#define FATAL(module) autoalg::LogStream(std::cerr, "FATAL", #module)

#endif  // INCLUDE_AUTOALG_LOG_H
