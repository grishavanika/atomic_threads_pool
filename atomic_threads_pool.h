// Implementation of threads pool with "Single-Thread Queue" idea
// from CppCon 2015: Fedor Pikus PART 2 "Live Lock-Free or Deadlock (Practical Lock-free Programming)"
// talk : https ://youtu.be/1obZeHnAwz4.
#pragma once

#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <future>
#include <optional>
#include <functional>

#include <cassert>
#include <cstdint>

#if defined(_MSC_VER)
// https://docs.microsoft.com/en-us/cpp/intrinsics/bitscanforward-bitscanforward64
#  pragma intrinsic(_BitScanForward64)
#endif

namespace nn
{

    using Task = std::function<void()>;

    namespace detail
    {
        struct ThreadQueue
        {
            std::queue<Task> _q;

            void push(Task task)
            {
                _q.push(std::move(task));
            }

            std::size_t size() const
            {
                return _q.size();
            }

            Task pop()
            {
                Task t = std::move(_q.front());
                _q.pop();
                return t;
            }
        };

        struct Event
        {
            std::mutex _mutex;
            std::condition_variable _event;
            bool _signaled = false;

            void wait_signaled()
            {
                std::unique_lock lock(_mutex);
                if (std::exchange(_signaled, false))
                {
                    return;
                }
                _event.wait(lock, [this] { return _signaled; });
                _signaled = false;
            }

            void signal_one()
            {
                std::unique_lock lock(_mutex);
                if (_signaled)
                {
                    return;
                }
                _signaled = true;
                _event.notify_one();
            }
        };

        static std::optional<std::size_t> find_first_set_bit(std::uint64_t v)
        {
#if defined(_MSC_VER)
            using U64 = unsigned __int64;
            static_assert(sizeof(v) == sizeof(U64));
            unsigned long first_set = 0;
            const unsigned char found = _BitScanForward64(&first_set, U64(v));
            if (found)
            {
                return std::make_optional(std::size_t(first_set));
            }
            return std::nullopt;
#else
            using U64 = unsigned long;
            static_assert(sizeof(v) == sizeof(U64));
            const int set = __builtin_ffsl(U64(v));
            if (set > 0)
            {
                return std::make_optional(std::size_t(set - 1));
            }
            return std::nullopt;
#endif
        }
    } // namespace detail

    class AtomicThreadsPool
    {
    public:
        explicit AtomicThreadsPool(std::size_t threads_count);
        ~AtomicThreadsPool();

        AtomicThreadsPool(const AtomicThreadsPool&) = delete;
        AtomicThreadsPool& operator=(const AtomicThreadsPool&) = delete;
        AtomicThreadsPool(AtomicThreadsPool&&) = delete;
        AtomicThreadsPool& operator=(AtomicThreadsPool&&) = delete;

        void start_all();
        void stop_all();

        void schedule(Task&& task, std::size_t thread_id_hint = 0);

    private:
        enum class PopStatus
        {
            Ok,
            Locked,
            Empty,
        };

        enum class StealStatus
        {
            Ok,
            SomeOrAllLocked,
            AllEmpty,
        };

        enum class SuspendStatus
        {
            Failed,
            FromActiveToSuspended,
            FromSuspendedToSuspended,
        };

        void create_all();

        void execute_from_best(Task&& task, std::size_t thread_id_hint);

        void thread_loop(std::size_t thread_id
            , std::shared_future<void> on_start
            , std::promise<void>& on_all_created
            , std::atomic_size_t& created_now);

        PopStatus try_pop_task(std::size_t thread_id, Task& task);
        bool try_push_task(std::size_t thread_id, Task&& task);

        StealStatus try_steal_task_from_others(std::size_t ignore_thread_id, Task& task);

        SuspendStatus try_suspend_thread(std::size_t thread_id);
        bool try_resume_any_thread();
        void resume_thread(std::size_t thread_id, std::uint64_t suspended_threads_mask);

    private:
        static thread_local std::size_t _self_thread_id;

        struct ThreadState
        {
            detail::ThreadQueue _unsafe_queue;
            std::atomic<detail::ThreadQueue*> _atomic_queue = &_unsafe_queue;
            detail::Event _on_new_task;
            std::thread _thread;
        };

        const std::size_t _threads_count;
        const std::uint64_t _all_suspended_mask;

        std::unique_ptr<ThreadState[]> _states;
        std::atomic_uint64_t _suspended_threads;
        std::atomic_bool _exit;

        std::promise<void> _on_start;
    };

    /*static*/ thread_local std::size_t AtomicThreadsPool::_self_thread_id = std::size_t(-1);

    /*explicit*/ AtomicThreadsPool::AtomicThreadsPool(std::size_t threads_count)
        : _threads_count(threads_count)
        , _all_suspended_mask((std::uint64_t(1) << _threads_count) - 1)
        , _states(std::make_unique<ThreadState[]>(threads_count))
        , _suspended_threads(0)
        , _exit(true)
        , _on_start()
    {
        assert(_threads_count > 0);
        // Because of `_suspended_threads` bookkeeping (one bit per uint64_t).
        assert(_threads_count < 64);

        create_all();
    }

    AtomicThreadsPool::~AtomicThreadsPool()
    {
        stop_all();
    }

    void AtomicThreadsPool::create_all()
    {
        assert(_exit);

        // Start & wait all started threads.
        _on_start = {};
        auto start_event = _on_start.get_future().share();

        std::promise<void> on_all_created;
        auto all_created_event = on_all_created.get_future().share();

        std::atomic_size_t started_now = 0;
        for (std::size_t thread_id = 0; thread_id < _threads_count; ++thread_id)
        {
            _states[thread_id]._thread = std::thread(
                &AtomicThreadsPool::thread_loop, this
                , thread_id
                , start_event
                , std::ref(on_all_created)
                , std::ref(started_now));
        }

        // Wait for everyone to start.
        (void)all_created_event.get();
    }

    void AtomicThreadsPool::start_all()
    {
        assert(_exit);
        _exit = false;
        _on_start.set_value();
    }

    void AtomicThreadsPool::stop_all()
    {
        if (not _states)
        {
            // Was already stopped by someone. Just to be nice in d-tor.
            return;
        }

        assert(not _exit);
        _exit = true;
        for (std::size_t i = 0; i < _threads_count; ++i)
        {
            _states[i]._on_new_task.signal_one();
        }
        for (std::size_t i = 0; i < _threads_count; ++i)
        {
            _states[i]._thread.join();
        }

        _states.reset();
    }

    void AtomicThreadsPool::schedule(Task&& task, std::size_t thread_id_hint /*= 0*/)
    {
        execute_from_best(std::move(task), thread_id_hint);
    }

    void AtomicThreadsPool::execute_from_best(Task&& task, std::size_t thread_id_hint)
    {
        assert(task);
        assert(thread_id_hint < _threads_count);

        if (std::size_t thread_id = _self_thread_id; thread_id != std::size_t(-1))
        {
            // We are inside some worker thread.
            // Try to push to its own queue.
            while (not try_push_task(thread_id, std::move(task)))
            {
                // This can happen if some other worker wants to steal
                // from this queue. Try another queue now.
                thread_id = ((thread_id + 1) % _threads_count);
            }

            // We know by definition that at least one thread is active
            // (current one). Still, we need to try to resume some other
            // thread if we know for sure it's suspended.
            // This is to avoid case when only our thread is active and
            // produces multiple tasks; without resuming the thread
            // we end up in a situation were single-thread consumes all the work.
            (void)try_resume_any_thread();
            // Note: in case there are no suspended threads (or we think so -
            // some or all other threads may go  to sleep right now) - we are fine
            // because at least our current thread is active.
            // 
            // assert(running_threads_count >= 1);
            return;
        }

        const std::uint64_t suspended_threads = _suspended_threads.load(std::memory_order_relaxed);
        const std::optional<std::size_t> suspended_thread_id = detail::find_first_set_bit(suspended_threads);
        std::size_t thread_id = suspended_thread_id.value_or(thread_id_hint);
        while (not try_push_task(thread_id, std::move(task)))
        {
            // This can happen if some other worker wants to steal
            // from this queue. Try another queue.
            thread_id = ((thread_id + 1) % _threads_count);
        }

        if (suspended_thread_id)
        {
            // We knew there _was_ some suspended thread. Try to resume it.
            // If it was already resumed - that's fine, the thread will just grab our task
            // and we are good to go (no need to double-check suspended thread after we inserted the task).
            resume_thread(*suspended_thread_id, suspended_threads);
            // assert(running_threads_count >= 1);
            return;
        }

        // We need to resume someone because we did insert into
        // queue thinking all threads were running. But all of them
        // may go to sleep and miss new task.
        (void)try_resume_any_thread();
        // Edge-case: no threads may be resumed there. And that's fine because right
        // before going to sleep `thread_loop()` goes and checks queues again
        // after raising `_suspended_threads` flags.
    }

    void AtomicThreadsPool::thread_loop(std::size_t thread_id
        , std::shared_future<void> on_start
        , std::promise<void>& on_all_created
        , std::atomic_size_t& created_now)
    {
        _self_thread_id = thread_id; // Set to thread-local.

        if (++created_now == _threads_count)
        {
            on_all_created.set_value();
        }

        (void)on_start.get();

        Task task;
        auto execute = [](Task& work)
        {
            Task process = std::exchange(work, {});
            std::move(process)();
        };

        while (not _exit.load(std::memory_order_relaxed))
        {
            bool steal_from_others = false;
            switch (try_pop_task(thread_id, task))
            {
            case PopStatus::Ok:
                execute(task);
                break;
            case PopStatus::Locked:
                // Try again.
                steal_from_others = false;
                break;
            case PopStatus::Empty:
                steal_from_others = true;
                break;
            }

            if (not steal_from_others)
            {
                continue;
            }

            const StealStatus status = try_steal_task_from_others(thread_id, task);
            switch (status)
            {
            case StealStatus::Ok:
                execute(task);
                break;
            case StealStatus::SomeOrAllLocked:
                break;
            case StealStatus::AllEmpty:
                (void)try_suspend_thread(thread_id);
                break;
            }
        }
    }

    auto AtomicThreadsPool::try_pop_task(std::size_t thread_id, Task& task)
        -> PopStatus
    {
        assert(thread_id < _threads_count);

        detail::ThreadQueue* queue = _states[thread_id]._atomic_queue
            .exchange(nullptr, std::memory_order_acquire);
        PopStatus status = PopStatus::Ok;

        if (not queue)
        {
            status = PopStatus::Locked;
        }
        else if (queue->size() == 0)
        {
            status = PopStatus::Empty;
        }
        else
        {
            task = queue->pop();
        }

        if (queue)
        {
            _states[thread_id]._atomic_queue
                .store(queue, std::memory_order_release);
        }
        return status;
    }

    bool AtomicThreadsPool::try_push_task(std::size_t thread_id, Task&& task)
    {
        assert(thread_id < _threads_count);

        detail::ThreadQueue* queue = _states[thread_id]._atomic_queue
            .exchange(nullptr, std::memory_order_acquire);
        if (queue)
        {
            queue->push(std::move(task));
            _states[thread_id]._atomic_queue
                .store(queue, std::memory_order_release);
            return true;
        }
        return false;
    }

    auto AtomicThreadsPool::try_steal_task_from_others(std::size_t ignore_thread_id, Task& task)
        -> StealStatus
    {
        assert(_threads_count > 0);
        std::size_t empty_count = 0;

        // Simplest-possible way to go thru all other queues.
        // Other options include random-selection of the queue.
        for (std::size_t i = (ignore_thread_id + 1); i < (_threads_count + ignore_thread_id); ++i)
        {
            const std::size_t thread_id = (i % _threads_count);
            bool found = false;
            switch (try_pop_task(thread_id, task))
            {
            case PopStatus::Locked:
                break;
            case PopStatus::Empty:
                ++empty_count;
                break;
            case PopStatus::Ok:
                found = true;
                break;
            }
            if (found)
            {
                return StealStatus::Ok;
            }
        }

        return (empty_count == (_threads_count - 1)) // without ignored one.
            ? StealStatus::AllEmpty
            : StealStatus::SomeOrAllLocked;
    }

    auto AtomicThreadsPool::try_suspend_thread(std::size_t thread_id)
        -> SuspendStatus
    {
        assert(thread_id < 64);

        std::uint64_t expected = _suspended_threads.load(std::memory_order::relaxed);
        const std::uint64_t thread_mask = (std::uint64_t(1) << std::uint64_t(thread_id));
        const std::uint64_t desired = (expected | thread_mask);
        if (_suspended_threads.compare_exchange_weak(
            expected
            , desired
            , std::memory_order_acq_rel))
        {
            if ((expected & thread_mask) == 0)
            {
                // We went from "active" thread to "suspended".
                // Don't go to sleep now, give a chance to thread
                // to check queues again. This is needed to handle
                // possible edge-case in execute_from_best(), see "Edge-case" note.
                return SuspendStatus::FromActiveToSuspended;
            }

            // Race between `_suspended_threads` set and event wait is fine
            // (someone can re-set mask we set just now and make
            // event signaled _before_ we go to actual wait;
            // this is handled by flag inside event itself;
            // wait below will be no-op in this case).
            _states[thread_id]._on_new_task.wait_signaled();

            // Note: there we can't expect for the bit to be zero.
            // Event may become signaled simply to wake-up all threads
            // to check & stop the work, as example.
            return SuspendStatus::FromSuspendedToSuspended;
        }

        // In case we failed to set, do not try to do strong CAS
        // again. We simply let the thread try to steal tasks from
        // possibly empty queues and go to try-suspend again.
        return SuspendStatus::Failed;
    }

    bool AtomicThreadsPool::try_resume_any_thread()
    {
        const std::uint64_t suspended_threads = _suspended_threads.load(std::memory_order_relaxed);
        const std::optional<std::size_t> thread_id = detail::find_first_set_bit(suspended_threads);
        if (thread_id)
        {
            resume_thread(*thread_id, suspended_threads);
            return true;
        }
        return false;
    }

    void AtomicThreadsPool::resume_thread(
        std::size_t thread_id, std::uint64_t suspended_threads_mask)
    {
        assert(thread_id < 64);
        auto make_mask = [this, &thread_id](std::uint64_t current)
        {
            return (current & ~(std::uint64_t(1) << thread_id))
                & _all_suspended_mask;
        };

        std::uint64_t expected = suspended_threads_mask;
        std::uint64_t desired = make_mask(expected);
        while (!_suspended_threads.compare_exchange_weak(
            expected
            , desired
            , std::memory_order_acq_rel))
        {
            desired = make_mask(expected);
        }

        // Even if (possible) thread was not suspended because bit was not set,
        // we still need to signal; this is to avoid case when threads
        // want to go to sleep right now.
        _states[thread_id]._on_new_task.signal_one();
    }

} // namespace nn

