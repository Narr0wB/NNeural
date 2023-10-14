
#ifndef LOG_H
#define LOG_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <csignal>

class Log {
    private:
        static std::shared_ptr<spdlog::logger> s_CoreLogger;

    public:
        static void Init();

        inline static std::shared_ptr<spdlog::logger>& GetLogger() { return s_CoreLogger; }
};

#define LOG_ERROR(...)  do { Log::GetLogger()->error(__VA_ARGS__); exit(EXIT_FAILURE); } while (0)
#define LOG_WARN(...)   Log::GetLogger()->warn(__VA_ARGS__)
#define LOG_INFO(...)   Log::GetLogger()->info(__VA_ARGS__)

// #define ASSERT(condition) do  { if (!(condition)) {LOG_ERROR("ASSERTION FAILED! {} {}", __FILE__, __LINE__);}; } while (0)

#endif // LOG_H
