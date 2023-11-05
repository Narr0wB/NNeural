
#include "Log.h"

std::shared_ptr<spdlog::logger> Log::s_CoreLogger;

void Log::Init(){
    spdlog::set_pattern("%^[%T] %n: %v%$");
    s_CoreLogger = spdlog::stdout_color_mt("CORE");
    s_CoreLogger->set_level(spdlog::level::trace);
}

template <typename T>
void log_print_vector(int logger_level, std::vector<T> vector, std::string separator) {
    if (vector.size() == 0) {
        return;
    }

    switch (logger_level) {
        case LERROR: {

            std::string fmt_string;

            for (size_t t = 0; t < vector.size() - 1; ++t) {
                fmt_string += ("{}" + separator);
            }

            fmt_string += "{}";

            break;
        }

        case LWARN: {
            break;
        }

        case LINFO: {
            break;
        }

        default: break;
    }
}