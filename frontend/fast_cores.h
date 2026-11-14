#pragma once
// Android: keeps the calling thread off a phone's small processor cores.
//
// A phone or handheld has small cores beside its fast ones, and the system's scheduler likes to
// move a thread that waits for part of each frame onto a small one, at a low clock. The
// emulation thread and the thread that draws and presents the frames both wait like that. On a
// Mangmi Air X (four Cortex-A53 cores at 2.0 GHz, four Cortex-A73 at 2.1 GHz) a frame's
// emulation took from 6 to 14 ms in the same place of the same game, more than a frame's
// 16.7 ms with the drawing beside it. The small cores are those whose top clock is the lowest of
// all, when the cores differ. (The widescreen GPU thread does the same:
// melonDS/src/GPU3D_WideGpu_GL.cpp.) SSR_ANYCORE=1 leaves the threads alone.
#ifdef __ANDROID__
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>

namespace FastCores
{
// Whether the thread is now kept to the faster cores; what was done goes to the log as `who`.
inline bool KeepThisThread(const char* who)
{
    if (getenv("SSR_ANYCORE")) return false;
    long top[64] {}, lowest = 0, highest = 0;
    int n = 0;
    for (; n < 64; n++)
    {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", n);
        FILE* f = fopen(path, "r");
        if (!f) break;
        if (fscanf(f, "%ld", &top[n]) != 1) top[n] = 0;
        fclose(f);
        if (!n || top[n] < lowest) lowest = top[n];
        if (top[n] > highest) highest = top[n];
    }
    if (n < 2 || lowest == highest || lowest <= 0) return false;
    cpu_set_t set;
    CPU_ZERO(&set);
    int kept = 0;
    for (int i = 0; i < n; i++)
        if (top[i] > lowest)
        {
            CPU_SET(i, &set);
            kept++;
        }
    if (!kept || sched_setaffinity(0, sizeof(set), &set) != 0) return false;
    fprintf(stderr, "[cores] %s: kept to the %d faster cores of %d\n", who, kept, n);
    return true;
}

// Android's performance hints (Android 13 and later, libandroid's APerformanceHint): each frame
// the system is told how long the emulation took against the time it should take, and sets the
// cores' clocks from that rather than from how busy they were a moment ago. The scheduler's
// own guess lowers the clocks while the threads wait between frames, so the next frame starts
// slow. The target is three quarters of a frame, which leaves room for a heavier frame.
// Phones without it (older Android, or no support in the system) carry on as before.
// SSR_NOHINT=1 leaves it off.
class WorkHint
{
public:
    // ms: the emulation's time for the frame just run; tid: the emulation thread (0: this one).
    void Report(double ms, int tid)
    {
        if (!Tried)
        {
            Tried = true;
            Start(tid);
        }
        if (Session && ms > 0) ReportFn(Session, (int64_t)(ms * 1e6));
    }

private:
    void Start(int tid)
    {
        if (getenv("SSR_NOHINT")) return;
        void* lib = dlopen("libandroid.so", RTLD_NOW);
        if (!lib) return;
        using GetManager = void* (*)();
        using CreateSession = void* (*)(void*, const int32_t*, size_t, int64_t);
        auto getManager = reinterpret_cast<GetManager>(dlsym(lib, "APerformanceHint_getManager"));
        auto createSession = reinterpret_cast<CreateSession>(dlsym(lib, "APerformanceHint_createSession"));
        ReportFn = reinterpret_cast<ReportFnType>(dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
        void* manager = getManager && createSession && ReportFn ? getManager() : nullptr;
        if (!manager)
        {
            fprintf(stderr, "[cores] performance hints: not on this system\n");
            return;
        }
        int32_t tids[2] = { (int32_t)gettid(), (int32_t)tid };
        const size_t count = tid && tid != tids[0] ? 2 : 1;
        Session = createSession(manager, tids, count, (int64_t)(1e9 / 60 * 3 / 4));
        fprintf(stderr, "[cores] performance hints: %s\n", Session ? "on, for a frame's emulation in 12.5 ms" : "refused by the system");
    }
    using ReportFnType = int (*)(void*, int64_t);
    ReportFnType ReportFn = nullptr;
    void* Session = nullptr;
    bool Tried = false;
};
}
#endif
