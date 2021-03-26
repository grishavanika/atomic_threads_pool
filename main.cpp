#define XX_ATOMIC_POOL_NO_SUSPENDED_MASK() 1

#if (XX_ATOMIC_POOL_NO_SUSPENDED_MASK())
#  include "atomic_threads_pool_no_suspended_mask.h"
#else
#  include "atomic_threads_pool.h"
#endif

#define XX_ATOMIC_POOL() 1
#define XX_IDEAL_SCHEDULE() 1
#define XX_TIME_INCLUDE_SCHEDULING() 0

constexpr std::size_t kThreadsCount = 5;
constexpr std::size_t kCount = 1'000'000;

#if (XX_IDEAL_SCHEDULE())
static_assert((kCount % kThreadsCount) == 0);
#endif


// For SimplestThreadsPool implementation below.
#include <queue>
#include <functional>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>

// For test code in main().
#include <iostream>
#include <chrono>

struct SimplestThreadsPool
{
    using Task = std::function<void ()>;
    using OneShotEvent = nn::detail::OneShotEvent;
    using CountedEvent = nn::detail::CountedEvent;

    std::size_t _threads_count;
    std::queue<Task> _tasks;
    std::mutex _mutex;
    std::condition_variable _on_new_task;
    std::unique_ptr<std::thread[]> _threads;
    std::atomic_bool _exit;
    OneShotEvent _on_start_all;

    explicit SimplestThreadsPool(std::size_t threads_count)
        : _threads_count(threads_count)
        , _threads(std::make_unique<std::thread[]>(threads_count))
        , _exit(true)
    {
        CountedEvent threads_started(_threads_count);
        for (std::size_t thread_id = 0; thread_id < _threads_count; ++thread_id)
        {
            _threads[thread_id] = std::thread(
                  &SimplestThreadsPool::thread_loop, this
                , thread_id
                , std::ref(threads_started));
        }
        threads_started.wait_all();
    }

    ~SimplestThreadsPool()
    {
        stop_all_once();
    }

    void start_all_once()
    {
        assert(_exit);
        _exit = false;
        _on_start_all.notify_all();
    }

    void stop_all_once()
    {
        if (!_threads)
        {
            return;
        }
        _exit = true;
        _on_new_task.notify_all();
        for (std::size_t i = 0; i < _threads_count; ++i)
        {
            _threads[i].join();
        }
        _threads.reset();
    }

    void schedule(Task&& task)
    {
        {
            std::lock_guard lock(_mutex);
            _tasks.push(std::move(task));
        }
        _on_new_task.notify_one();
    }

    void thread_loop(std::size_t /*thread_id*/
        , CountedEvent& threads_started)
    {
        threads_started.notify();
        _on_start_all.wait_once();

        while (not _exit.load(std::memory_order_relaxed))
        {
            Task t;
            {
                std::unique_lock lock(_mutex);
                if (_tasks.empty())
                {
                    _on_new_task.wait(lock, [=]
                    {
                        return _exit.load(std::memory_order_relaxed)
                            || (not _tasks.empty());
                    });
                }
                if (_exit.load(std::memory_order_relaxed))
                {
                    break;
                }
                assert(not _tasks.empty());
                t = std::move(_tasks.front());
                _tasks.pop();
            }
            // No lock hold.
            std::move(t)();
        }
    }
};

int main()
{
#if (XX_ATOMIC_POOL())
    using Pool = nn::AtomicThreadsPool;
#else
    using Pool = SimplestThreadsPool;
#endif

    Pool pool(kThreadsCount);
    nn::detail::CountedEvent work_end(kCount);

    auto work = [&]()
    {
        work_end.notify();
    };

#if (XX_TIME_INCLUDE_SCHEDULING())
#if (XX_IDEAL_SCHEDULE())
#  error Conflict in options: can't do unsafe schedule when pool is started.
#endif
    const auto s = std::chrono::high_resolution_clock::now();
    pool.start_all_once();
#endif

#if (XX_IDEAL_SCHEDULE())
#if (!XX_ATOMIC_POOL())
#  error Conflict in options: can't do ideal schedule for standard threads pool.
#endif
    for (std::size_t i = 0; i < (kCount / kThreadsCount); ++i)
    {
        for (std::size_t thread_id = 0; thread_id < kThreadsCount; ++thread_id)
        {
            pool.schedule_unsafe(work, nn::ThreadIndex(thread_id));
        }
    }
#else
    for (std::size_t i = 0; i < kCount; ++i)
    {
        pool.schedule(work);
    }
#endif

#if (!XX_TIME_INCLUDE_SCHEDULING())
    const auto s = std::chrono::high_resolution_clock::now();
    pool.start_all_once();
#endif

    work_end.wait_all();

    const auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(e - s);
    std::cout << d.count() << " " << "ms" << "\n";

    pool.stop_all_once();
}
