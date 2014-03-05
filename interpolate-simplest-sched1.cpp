#include <Halide.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <map>
#include <string>

// How many times to run (and take min)
#ifndef AUTOTUNE_TRIALS
#define AUTOTUNE_TRIALS 3
#endif

// Limit in seconds to try running for (0 = no limit)
#ifndef AUTOTUNE_LIMIT
#define AUTOTUNE_LIMIT 0
#endif

// Size to run with
#ifndef AUTOTUNE_N
#define AUTOTUNE_N 1024, 1024
#endif

inline void _autotune_timing_stub(Halide::Func& func) {
    func.compile_jit();
    func.infer_input_bounds(AUTOTUNE_N);
    timeval t1, t2;
    double rv = 0;
    const unsigned int timeout = AUTOTUNE_LIMIT;
    alarm(timeout);
    for (int i = 0; i < AUTOTUNE_TRIALS; i++) {
      gettimeofday(&t1, NULL);
      func.realize(AUTOTUNE_N);
      gettimeofday(&t2, NULL);
      alarm(0); // disable alarm
      double t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;
      if(i == 0 || t < rv)
        rv = t;
    }
    printf("{\"time\": %.10f}\n", rv);
    exit(0);
}


#ifndef AUTOTUNE_HOOK
#define AUTOTUNE_HOOK(x)
#endif

#ifndef BASELINE_HOOK
#define BASELINE_HOOK(x)
#endif

#include "Halide.h"

#define AUTOTUNE_HOOK(x)
#define BASELINE_HOOK(x)

using namespace Halide;

#include <iostream>
#include <limits>

#include <sys/time.h>

using std::vector;

double now() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    static bool first_call = true;
    static time_t first_sec = 0;
    if (first_call) {
        first_call = false;
        first_sec = tv.tv_sec;
    }
    assert(tv.tv_sec >= first_sec);
    return (tv.tv_sec - first_sec) + (tv.tv_usec / 1000000.0);
}

int main(int argc, char **argv) {
    ImageParam input(Float(32), 3, "input");

    Func upsampled;
    Func upsampledx;
    Var x("x"), y("y"), c("c");

    Func clamped("clamped");
    clamped(x, y, c) = input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1), c);

    upsampledx = Func("upsampledx");
    upsampled = Func("upsampled");
    upsampledx(x, y, c) = select((x % 2) == 0,
                                    clamped(x/2, y, c),
                                    0.5f * (clamped(x/2, y, c) +
                                            clamped(x/2+1, y, c)));
    upsampled(x, y, c) = select((y % 2) == 0,
                                   upsampledx(x, y/2, c),
                                   0.5f * (upsampledx(x, y/2, c) +
                                           upsampledx(x, y/2+1, c)));

    Func final = upsampled;
    {
        std::map<std::string, Halide::Internal::Function> funcs = Halide::Internal::find_transitive_calls((final).function());

        Halide::Var _x0, _y1, _c2, _x3, _y4, _c5, _x6, _c8, _x12, _y13, _x15, _y16, _x18, _c20, _x21, _y22, _c23, _y25, _x27, _y28;
        clamped.compute_root();
        upsampled
            .split(x, x, _x21, 4)
            .split(y, y, _y22, 8)
            .reorder(_y22, _x21, y, c, x)
            .compute_root();
        upsampledx.compute_at(upsampled, _y22);

        _autotune_timing_stub(final);
    };
}
