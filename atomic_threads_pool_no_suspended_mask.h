// Implementation of threads pool with "Single-Thread Queue" idea
// from CppCon 2015: Fedor Pikus PART 2 "Live Lock-Free or Deadlock (Practical Lock-free Programming)"
// talk: https://youtu.be/1obZeHnAwz4.
#pragma once

#include <atomic>
#include <queue>
#include <thread>
#include <mutex>
#include <future>
#include <functional>
#include <new> // obviously, for hardware_destructive_interference_size.

#include <cassert>
#include <cstdint>

namespace nn
{

    using Task = std::function<void()>;

    namespace detail
    {
        struct ThreadQueue
        {
            std::queue<Task> _q;

            void push(Task&& task)
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

        // Kind-a auto-reset Win32 event.
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

        struct OneShotEvent
        {
            std::promise<void> _trigger;
            std::shared_future<void> _wait = _trigger.get_future().share();

            void wait_once() { _wait.get(); }
            void notify_all() { _trigger.set_value(); }
        };

        struct CountedEvent : private OneShotEvent
        {
            const std::size_t _limit;
            std::atomic_size_t _count;

            explicit CountedEvent(std::size_t limit)
                : _limit(limit)
                , _count(0) {}

            void notify()
            {
                const std::size_t count = _count.fetch_add(1, std::memory_order_relaxed) + 1;
                assert(count <= _limit);
                if (count == _limit)
                {
                    notify_all();
                }
            }

            void wait_all() { wait_once(); }
        };
    } // namespace detail

    using ThreadIndex = std::uint32_t;

    class AtomicThreadsPool
    {
    public:
        explicit AtomicThreadsPool(std::size_t threads_count);
        ~AtomicThreadsPool();

        AtomicThreadsPool(const AtomicThreadsPool&) = delete;
        AtomicThreadsPool& operator=(const AtomicThreadsPool&) = delete;
        AtomicThreadsPool(AtomicThreadsPool&&) = delete;
        AtomicThreadsPool& operator=(AtomicThreadsPool&&) = delete;

        // Can be called only once per threads pool lifetime.
        void start_all_once();
        void stop_all_once();

        void schedule(Task&& task);
        void schedule_unsafe(Task&& task, ThreadIndex thread_id);

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
            None,
            FromEnabledToDisabled,
            FromDisabledToSuspended,
        };

        void execute_from_best(Task&& task);

        void thread_loop(ThreadIndex thread_id
            , detail::CountedEvent& threads_started);

        PopStatus try_pop_task(ThreadIndex thread_id, Task& task);
        bool try_push_task(ThreadIndex thread_id, Task&& task);

        StealStatus try_steal_task_from_others(ThreadIndex ignore_thread_id, Task& task);

        SuspendStatus try_suspend_thread(ThreadIndex thread_id, bool was_disabled);
        bool try_resume_any_thread();
        void resume_thread(ThreadIndex thread_id);

        void mark_thread_enabled(ThreadIndex thread_id);

    private:
        static constexpr ThreadIndex kNoThreadId = ThreadIndex(-1);
        static thread_local ThreadIndex _self_thread_id;

        struct ThreadState
        {
            detail::ThreadQueue _unsafe_queue;
            char _no_false_sharing1[std::hardware_destructive_interference_size];

            std::atomic<detail::ThreadQueue*> _atomic_queue = &_unsafe_queue;
            char _no_false_sharing2[std::hardware_destructive_interference_size];

            std::atomic_bool _suspended = false;
            char _no_false_sharing3[std::hardware_destructive_interference_size];

            detail::Event _on_new_task;
            std::thread _thread;
        };

        const std::uint32_t _threads_count;

        std::unique_ptr<ThreadState[]> _states;
        detail::OneShotEvent _on_start_all;

        std::atomic_bool _exit;
    };

    /*static*/ thread_local ThreadIndex AtomicThreadsPool::_self_thread_id
        = AtomicThreadsPool::kNoThreadId;

    /*explicit*/ AtomicThreadsPool::AtomicThreadsPool(std::size_t threads_count)
        : _threads_count(std::uint32_t(threads_count))
        , _states(std::make_unique<ThreadState[]>(threads_count))
        , _exit(true)
        , _on_start_all()
    {
        assert(_threads_count > 0);

        detail::CountedEvent threads_started(_threads_count);
        for (ThreadIndex thread_id = 0; thread_id < _threads_count; ++thread_id)
        {
            _states[thread_id]._thread = std::thread(
                  &AtomicThreadsPool::thread_loop, this
                , thread_id
                , std::ref(threads_started));
        }
        threads_started.wait_all();
    }

    AtomicThreadsPool::~AtomicThreadsPool()
    {
        stop_all_once();
        _states.reset();
    }

    void AtomicThreadsPool::start_all_once()
    {
        const bool exit_was_set = _exit.exchange(false);
        assert(exit_was_set);
        _on_start_all.notify_all();
    }

    void AtomicThreadsPool::stop_all_once()
    {
        const bool exit_was_set = _exit.exchange(true);
        if (exit_was_set)
        {
            return;
        }

        for (ThreadIndex i = 0; i < _threads_count; ++i)
        {
            _states[i]._on_new_task.signal_one();
        }
        for (ThreadIndex i = 0; i < _threads_count; ++i)
        {
            _states[i]._thread.join();
        }
    }

    void AtomicThreadsPool::schedule(Task&& task)
    {
        execute_from_best(std::move(task));
    }

    void AtomicThreadsPool::schedule_unsafe(Task&& task, ThreadIndex thread_id)
    {
        assert(thread_id < _threads_count);
        _states[thread_id]._unsafe_queue.push(std::move(task));
    }

    void AtomicThreadsPool::execute_from_best(Task&& task)
    {
        assert(task);

        if (ThreadIndex thread_id = _self_thread_id; thread_id != kNoThreadId)
        {
            // We are inside some worker thread.
            // Try to push to its own queue.
            while (not try_push_task(thread_id, std::move(task)))
            {
                // This can happen if some other worker wants to steal
                // from this queue. Try another queue now.
                thread_id = ThreadIndex((thread_id + 1) % _threads_count);
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

        ThreadIndex maybe_suspended = 0;
        bool found_suspended = false;
        for (ThreadIndex id = 0; id < _threads_count; ++id)
        {
            if (_states[id]._suspended.load(std::memory_order_relaxed))
            {
                maybe_suspended = id;
                found_suspended = true;
                break;
            }
        }

        ThreadIndex thread_id = maybe_suspended;
        while (not try_push_task(thread_id, std::move(task)))
        {
            // This can happen if some other worker wants to steal
            // from this queue. Try another queue.
            thread_id = ThreadIndex((thread_id + 1) % _threads_count);
        }

        if (found_suspended)
        {
            // We knew there _was_ some suspended thread. Try to resume it.
            // If it was already resumed - that's fine, the thread will just grab our task
            // and we are good to go (no need to double-check suspended thread after we inserted the task).
            resume_thread(maybe_suspended);
            // assert(running_threads_count >= 1);
            return;
        }

        // We need to resume someone because we did insert into
        // queue thinking all threads were running. But all of them
        // may go to sleep and miss new task.
        (void)try_resume_any_thread();
        // Edge-case: no threads may be resumed there. And that's fine because right
        // before going to sleep `thread_loop()` goes and checks queues again
        // after raising `_suspended` flag.
    }

    void AtomicThreadsPool::thread_loop(ThreadIndex thread_id
        , detail::CountedEvent& threads_started)
    {
        _self_thread_id = thread_id; // Set to thread-local.

        threads_started.notify();
        _on_start_all.wait_once();

        Task task;
        auto execute = [](Task& work)
        {
            // Makes sure task lifetime is ended right after execution.
            Task process = std::exchange(work, {});
            std::move(process)();
        };

        SuspendStatus last_suspend = SuspendStatus::None;
        auto ensure_thread_enabled = [this, thread_id](SuspendStatus& status)
        {
            if (std::exchange(status, SuspendStatus::None)
                == SuspendStatus::FromEnabledToDisabled)
            {
                mark_thread_enabled(thread_id);
            }
        };


        while (not _exit.load(std::memory_order_relaxed))
        {
            bool steal_from_others = false;
            switch (try_pop_task(thread_id, task))
            {
            case PopStatus::Ok:
                ensure_thread_enabled(last_suspend);
                execute(task);
                break;
            case PopStatus::Locked:
                ensure_thread_enabled(last_suspend);
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
                ensure_thread_enabled(last_suspend);
                execute(task);
                break;
            case StealStatus::SomeOrAllLocked:
                ensure_thread_enabled(last_suspend);
                break;
            case StealStatus::AllEmpty:
            {
                const bool already_disabled = (last_suspend == SuspendStatus::FromEnabledToDisabled);
                last_suspend = try_suspend_thread(thread_id, already_disabled);
                break;
            }
            }
        }
    }

    auto AtomicThreadsPool::try_pop_task(ThreadIndex thread_id, Task& task)
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

    bool AtomicThreadsPool::try_push_task(ThreadIndex thread_id, Task&& task)
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

    auto AtomicThreadsPool::try_steal_task_from_others(ThreadIndex ignore_thread_id, Task& task)
        -> StealStatus
    {
        assert(_threads_count > 0);
        std::uint32_t empty_count = 0;

        // Simplest-possible way to go thru all other queues.
        // Other options include random-selection of the queue.
        for (ThreadIndex i = (ignore_thread_id + 1); i < (_threads_count + ignore_thread_id); ++i)
        {
            const ThreadIndex thread_id = (i % _threads_count);
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

    auto AtomicThreadsPool::try_suspend_thread(ThreadIndex thread_id, bool was_disabled)
        -> SuspendStatus
    {
        bool was_suspended = was_disabled;
        if (_states[thread_id]._suspended.compare_exchange_weak(
              was_suspended
            , true
            , std::memory_order_acq_rel))
        {
            if (not was_suspended)
            {
                // We went from "active" thread to "suspended".
                // Don't go to sleep now, give a chance to thread
                // to check queues again. This is needed to handle
                // possible edge-case in execute_from_best(), see "Edge-case" note.
                return SuspendStatus::FromEnabledToDisabled;
            }

            // Race between `_suspended` set and event wait is fine
            // (someone can re-set mask we set just now and make
            // event signaled _before_ we go to actual wait;
            // this is handled by flag inside event itself;
            // wait below will be no-op in this case).
            _states[thread_id]._on_new_task.wait_signaled();

            // Note: there we can't expect `_suspended` be false.
            // Event may become signaled simply to wake-up all threads
            // to check & stop the work, as example.
            return SuspendStatus::FromDisabledToSuspended;
        }

        // In case we failed to set, do not try to do strong CAS
        // again. We simply let the thread try to steal tasks from
        // possibly empty queues and go to try-suspend again.
        return SuspendStatus::None;
    }

    bool AtomicThreadsPool::try_resume_any_thread()
    {
        for (ThreadIndex thread_id = 0; thread_id < _threads_count; ++thread_id)
        {
            if (_states[thread_id]._suspended.load(std::memory_order_relaxed))
            {
                resume_thread(thread_id);
                return true;
            }
        }
        return false;
    }

    void AtomicThreadsPool::mark_thread_enabled(ThreadIndex thread_id)
    {
        bool was_suspended = true;
        while (!_states[thread_id]._suspended.compare_exchange_weak(
              was_suspended
            , false
            , std::memory_order_acq_rel))
        {
            // No-op.
        }
    }

    void AtomicThreadsPool::resume_thread(ThreadIndex thread_id)
    {
        mark_thread_enabled(thread_id);
        // Even if (possible) thread was not suspended because bit was not set,
        // we still need to signal; this is to avoid case when threads
        // want to go to sleep right now.
        _states[thread_id]._on_new_task.signal_one();
    }

} // namespace nn

