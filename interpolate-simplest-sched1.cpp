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

    const unsigned int levels = 2;

    Func downsampled[levels];
    Func downx[levels];
    Func interpolated[levels];
    Func upsampled[levels];
    Func upsampledx[levels];
    Var x("x"), y("y"), c("c");

    downsampled[0] = Func("downsampled");
    downx[0] = Func("downx");
    interpolated[0] = Func("interpolated");
    upsampled[0] = Func("upsampled");
    upsampledx[0] = Func("upsampledx");

    Func clamped("clamped");
    clamped(x, y, c) = input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1), c);

    // This triggers a bug in llvm 3.3 (3.2 and trunk are fine), so we
    // rewrite it in a way that doesn't trigger the bug. The rewritten
    // form assumes the input alpha is zero or one.
    // downsampled[0](x, y, c) = select(c < 3, clamped(x, y, c) * clamped(x, y, 3), clamped(x, y, 3));
    #if 0
    // downsampled[0](x, y, c) = clamped(x, y, c) * clamped(x, y, 3);
    downsampled[0] = clamped;

    // for (unsigned int l = 1; l < levels; ++l) {
        downx[1] = Func("downx");
        downsampled[1] = Func("downsampled");
        downx[1](x, y, c) = (downsampled[0](x*2-1, y, c) +
                             2.0f * downsampled[0](x*2, y, c) +
                             downsampled[0](x*2+1, y, c)) * 0.25f;
        downsampled[1](x, y, c) = (downx[1](x, y*2-1, c) +
                                   2.0f * downx[1](x, y*2, c) +
                                   downx[1](x, y*2+1, c)) * 0.25f;
    // }
    #endif
    interpolated[1] = Func("interpolated"); // keep names stable
    #if 0
    interpolated[1](x, y, c) = downsampled[1](x, y, c);
    #else
    interpolated[1] = clamped;
    #endif
    // for (unsigned int l = levels-2; l < levels; --l) {
        upsampledx[0] = Func("upsampledx");
        upsampled[0] = Func("upsampled");
        interpolated[0] = Func("interpolated");
        upsampledx[0](x, y, c) = select((x % 2) == 0,
                                        interpolated[0+1](x/2, y, c),
                                        0.5f * (interpolated[0+1](x/2, y, c) +
                                                interpolated[0+1](x/2+1, y, c)));
        upsampled[0](x, y, c) = select((y % 2) == 0,
                                       upsampledx[0](x, y/2, c),
                                       0.5f * (upsampledx[0](x, y/2, c) +
                                               upsampledx[0](x, y/2+1, c)));
        // interpolated[0](x, y, c) = downsampled[0](x, y, c) + (1.0f - downsampled[0](x, y, 3)) * upsampled[0](x, y, c);
        interpolated[0](x, y, c) = upsampled[0](x, y, c);
    // }

    Func final = interpolated[0];
    {
        std::map<std::string, Halide::Internal::Function> funcs = Halide::Internal::find_transitive_calls((final).function());

        Halide::Var _x0, _y1, _c2, _x3, _y4, _c5, _x6, _c8, _x12, _y13, _x15, _y16, _x18, _c20, _x21, _y22, _c23, _y25, _x27, _y28;
        Halide::Func(funcs["clamped"])
        .compute_root()
        ;
        #if 0
        Halide::Func(funcs["downsampled"])
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$2"])
        .compute_root()
        ;
        Halide::Func(funcs["downx$2"])
        .compute_root()
        ;
        #endif
        Halide::Func(funcs["interpolated$3"])
        .compute_root()
        ;
        Halide::Func(funcs["upsampledx$2"])
        .compute_at(Halide::Func(funcs["upsampled$2"]), _y22)
        ;
        Halide::Func(funcs["upsampled$2"])
        .split(x, x, _x21, 4)
        .split(y, y, _y22, 8)
        .reorder(_y22, _x21, y, c, x)
        .compute_root()
        ;
        Halide::Func(funcs["interpolated$2"])
        .compute_root()
        ;

        _autotune_timing_stub(final);
    };
}
