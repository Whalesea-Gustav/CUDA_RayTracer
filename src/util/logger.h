#pragma once

#include <vector>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace gprender {

    class GPLog
    {
    public:
        static void init(spdlog::level::level_enum level = spdlog::level::trace)
        {
            std::vector<spdlog::sink_ptr> sinks;
            sinks.emplace_back(
                    std::make_shared<spdlog::sinks::stdout_color_sink_mt>());
            sinks.emplace_back(std::make_shared<spdlog::sinks::basic_file_sink_mt>(
                    "logs/GPMesh.log", true));

            sinks[0]->set_pattern("%^[%T] %n: %v%$");
            sinks[1]->set_pattern("[%T] [%l] %n: %v");

            m_logger = std::make_shared<spdlog::logger>(
                    "GPMesh", begin(sinks), end(sinks));
            spdlog::register_logger(m_logger);
            m_logger->set_level(level);
            m_logger->flush_on(level);
        }

        inline static std::shared_ptr<spdlog::logger>& get_logger()
        {
            return m_logger;
        }


    private:
        inline static std::shared_ptr<spdlog::logger> m_logger;
    };
}  // namespace gprender

#define GPRENDER_TRACE(...) ::gprender::GPLog::get_logger()->trace(__VA_ARGS__)
#define GPRENDER_INFO(...) ::gprender::GPLog::get_logger()->info(__VA_ARGS__)
#define GPRENDER_WARN(...)                                                      \
    ::gprender::GPLog::get_logger()->warn("Line {} File {}", __LINE__, __FILE__); \
    ::gprender::GPLog::get_logger()->warn(__VA_ARGS__)
#define GPRENDER_ERROR(...)                                                      \
    ::gprender::GPLog::get_logger()->error("Line {} File {}", __LINE__, __FILE__); \
    ::gprender::GPLog::get_logger()->error(__VA_ARGS__)
#define GPRENDER_CRITICAL(...)                    \
    ::gprender::GPLog::get_logger()->critical(      \
        "Line {} File {}", __LINE__, __FILE__); \
    ::gprender::GPLog::get_logger()->critical(__VA_ARGS__)
