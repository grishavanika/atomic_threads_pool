
Implementation of threads pool with "Single-Thread Queue" idea from
CppCon 2015: Fedor Pikus PART 2 "Live Lock-Free or Deadlock (Practical Lock-free Programming)" talk:
<https://youtu.be/1obZeHnAwz4>.

Note: toy examples (scheduling no-op task to single-queue outside thread pool worker)
shows poor performance compared to simplest-possible thread pool (one lock + one queue)
for a lot of reasons:

 - probably wrong implementation of thread's suspend/resume cycle
 - suspend/resume cycle that has O(N) - when N is a number of threads - complexity
 - non-randomized O(N) (same as above) complexity to steal a task from other threads

Ideal case (when all the work is scheduled before threads start, evenly between queues)
shows 2-3 times better results for, again, toy example with empty task.
