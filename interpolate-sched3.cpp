#include <Halide.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

#include <map>
#include <string>

// How many times to run (and take min)
// #define AUTOTUNE_TRIALS 3

// Limit in seconds to try running for (0 = no limit)
// #define AUTOTUNE_LIMIT 0

// Size to run with
// #define AUTOTUNE_N 1024, 1024

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

    const unsigned int levels = 10;

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
    downsampled[0](x, y, c) = clamped(x, y, c) * clamped(x, y, 3);

    for (unsigned int l = 1; l < levels; ++l) {
        downx[l] = Func("downx");
        downsampled[l] = Func("downsampled");
        downx[l](x, y, c) = (downsampled[l-1](x*2-1, y, c) +
                             2.0f * downsampled[l-1](x*2, y, c) +
                             downsampled[l-1](x*2+1, y, c)) * 0.25f;
        downsampled[l](x, y, c) = (downx[l](x, y*2-1, c) +
                                   2.0f * downx[l](x, y*2, c) +
                                   downx[l](x, y*2+1, c)) * 0.25f;
    }
    interpolated[levels-1] = Func("interpolated");
    interpolated[levels-1](x, y, c) = downsampled[levels-1](x, y, c);
    for (unsigned int l = levels-2; l < levels; --l) {
        upsampledx[l] = Func("upsampledx");
        upsampled[l] = Func("upsampled");
        interpolated[l] = Func("interpolated");
        upsampledx[l](x, y, c) = select((x % 2) == 0,
                                        interpolated[l+1](x/2, y, c),
                                        0.5f * (interpolated[l+1](x/2, y, c) +
                                                interpolated[l+1](x/2+1, y, c)));
        upsampled[l](x, y, c) = select((y % 2) == 0,
                                       upsampledx[l](x, y/2, c),
                                       0.5f * (upsampledx[l](x, y/2, c) +
                                               upsampledx[l](x, y/2+1, c)));
        interpolated[l](x, y, c) = downsampled[l](x, y, c) + (1.0f - downsampled[l](x, y, 3)) * upsampled[l](x, y, c);
    }

    Func normalize("normalize");
    normalize(x, y, c) = interpolated[0](x, y, c) / interpolated[0](x, y, 3);

    Func final("final");
    final(x, y, c) = normalize(x, y, c);
    {
        std::map<std::string, Halide::Internal::Function> funcs = Halide::Internal::find_transitive_calls((final).function());

        Halide::Var _x0, _y1, _c2, _x3, _y4, _x6, _y7, _x9, _y10, _c11, _x12, _y13, _x15, _y16, _x18, _c20, _x21, _y22, _c23, _x24, _y25, _x27, _y28, _c29, _y31, _c32, _x33, _y34, _c35, _y37, _c38, _x39, _y40, _c41, _x42, _x45, _y46, _x48, _c50, _x51, _y52, _c53, _x54, _y55, _c56, _x57, _y58, _c59, _x60, _y61, _c62, _y64, _y67, _c68, _x69, _c71, _x72, _y73, _x75, _y76, _c77, _x78, _y79, _x81, _y82, _c83, _x84, _y85, _x87, _y88, _x90, _y91, _x93, _y94, _x96, _c98, _x99, _c101, _y103, _x105, _y106, _c107, _c110, _x111, _y112, _c113, _x114, _y115, _x117, _y118, _c119, _x120, _y121, _c122, _x123, _c125, _x126, _y127, _x129, _y130, _c131, _x132, _x135, _y136, _c137, _y139, _c140, _x141, _c143, _x144, _c146, _x147, _y148, _c149;
        Halide::Func(funcs["clamped"])
        .split(x, x, _x0, 2)
        .split(y, y, _y1, 8)
        .split(c, c, _c2, 4)
        .reorder(_x0, _y1, x, _c2, y, c)
        .reorder_storage(c, y, x)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled"])
        .split(x, x, _x3, 8)
        .split(y, y, _y4, 8)
        .reorder(c, _x3, _y4, x, y)
        .reorder_storage(x, c, y)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$10"])
        .split(x, x, _x6, 4)
        .split(y, y, _y7, 8)
        .reorder(_y7, _x6, x, c, y)
        .reorder_storage(x, y, c)
        .vectorize(_y7, 2)
        .compute_at(Halide::Func(funcs["interpolated$2"]), _y67)
        ;
        Halide::Func(funcs["downsampled$2"])
        .split(x, x, _x9, 8)
        .split(y, y, _y10, 2)
        .split(c, c, _c11, 2)
        .reorder(_x9, _y10, x, _c11, c, y)
        .reorder_storage(y, c, x)
        .vectorize(_x9, 4)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$3"])
        .split(x, x, _x12, 16)
        .split(y, y, _y13, 8)
        .reorder(_x12, _y13, x, y, c)
        .reorder_storage(c, x, y)
        .vectorize(_x12, 4)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$4"])
        .split(x, x, _x15, 2)
        .split(y, y, _y16, 8)
        .reorder(_y16, y, _x15, x, c)
        .reorder_storage(c, y, x)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$5"])
        .split(x, x, _x18, 16)
        .split(c, c, _c20, 8)
        .reorder(_x18, _c20, x, y, c)
        .reorder_storage(x, c, y)
        .vectorize(_x18, 4)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$6"])
        .split(x, x, _x21, 4)
        .split(y, y, _y22, 8)
        .split(c, c, _c23, 4)
        .reorder(_x21, _y22, _c23, x, y, c)
        .reorder_storage(y, x, c)
        .vectorize(_x21, 2)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$7"])
        .split(x, x, _x24, 8)
        .split(y, y, _y25, 2)
        .reorder(_x24, x, c, _y25, y)
        .reorder_storage(y, x, c)
        .vectorize(_x24, 2)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$8"])
        .split(x, x, _x27, 4)
        .split(y, y, _y28, 2)
        .split(c, c, _c29, 64)
        .reorder(_c29, _y28, y, c, _x27, x)
        .reorder_storage(c, y, x)
        .vectorize(_c29, 8)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$9"])
        .split(y, y, _y31, 4)
        .split(c, c, _c32, 2)
        .reorder(_y31, y, _c32, c, x)
        .reorder_storage(c, x, y)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["downx$10"])
        .split(x, x, _x33, 8)
        .split(y, y, _y34, 16)
        .split(c, c, _c35, 8)
        .reorder(_y34, _c35, _x33, y, c, x)
        .reorder_storage(x, c, y)
        .vectorize(_y34, 8)
        .compute_at(Halide::Func(funcs["upsampledx$4"]), c)
        ;
        Halide::Func(funcs["downx$2"])
        .split(y, y, _y37, 32)
        .split(c, c, _c38, 8)
        .reorder(_y37, _c38, c, y, x)
        .reorder_storage(x, c, y)
        .vectorize(_y37, 4)
        .compute_root()
        ;
        Halide::Func(funcs["downx$3"])
        .split(x, x, _x39, 4)
        .split(y, y, _y40, 4)
        .split(c, c, _c41, 2)
        .reorder(_x39, _y40, y, _c41, c, x)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["downsampled$3"]), _x12)
        ;
        Halide::Func(funcs["downx$4"])
        .split(x, x, _x42, 8)
        .reorder(c, _x42, y, x)
        .reorder_storage(c, y, x)
        .vectorize(c, 4)
        .compute_at(Halide::Func(funcs["downsampled$4"]), _y16)
        ;
        Halide::Func(funcs["downx$5"])
        .split(x, x, _x45, 4)
        .split(y, y, _y46, 16)
        .reorder(_y46, y, _x45, c, x)
        .reorder_storage(c, y, x)
        .vectorize(_y46, 2)
        .compute_at(Halide::Func(funcs["downsampled$5"]), x)
        ;
        Halide::Func(funcs["downx$6"])
        .split(x, x, _x48, 64)
        .split(c, c, _c50, 8)
        .reorder(_x48, _c50, c, x, y)
        .reorder_storage(y, x, c)
        .vectorize(_x48, 8)
        .compute_at(Halide::Func(funcs["downsampled$6"]), _x21)
        ;
        Halide::Func(funcs["downx$7"])
        .split(x, x, _x51, 4)
        .split(y, y, _y52, 4)
        .split(c, c, _c53, 16)
        .reorder(_c53, _y52, _x51, c, x, y)
        .reorder_storage(c, x, y)
        .vectorize(_c53, 8)
        .compute_root()
        ;
        Halide::Func(funcs["downx$8"])
        .split(x, x, _x54, 8)
        .split(y, y, _y55, 4)
        .split(c, c, _c56, 4)
        .reorder(_x54, _y55, _c56, y, x, c)
        .reorder_storage(x, y, c)
        .vectorize(_x54, 4)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["downx$9"])
        .split(x, x, _x57, 2)
        .split(y, y, _y58, 16)
        .split(c, c, _c59, 8)
        .reorder(_y58, y, _c59, _x57, c, x)
        .reorder_storage(x, c, y)
        .vectorize(_y58, 8)
        .compute_at(Halide::Func(funcs["downsampled$9"]), _y31)
        ;
        Halide::Func(funcs["interpolated$10"])
        .split(x, x, _x60, 16)
        .split(y, y, _y61, 4)
        .split(c, c, _c62, 4)
        .reorder(_x60, _c62, x, _y61, y, c)
        .reorder_storage(y, c, x)
        .vectorize(_x60, 4)
        .compute_at(Halide::Func(funcs["upsampledx$10"]), _y121)
        ;
        Halide::Func(funcs["interpolated$11"])
        .split(y, y, _y64, 4)
        .reorder(_y64, y, x, c)
        .reorder_storage(x, y, c)
        .compute_at(Halide::Func(funcs["final"]), c)
        ;
        Halide::Func(funcs["interpolated$2"])
        .split(y, y, _y67, 16)
        .split(c, c, _c68, 4)
        .reorder(_y67, x, _c68, y, c)
        .reorder_storage(c, x, y)
        .vectorize(_y67, 4)
        .compute_at(Halide::Func(funcs["upsampledx$2"]), _x123)
        ;
        Halide::Func(funcs["interpolated$3"])
        .split(x, x, _x69, 8)
        .split(c, c, _c71, 16)
        .reorder(_c71, _x69, y, x, c)
        .reorder_storage(y, x, c)
        .vectorize(_c71, 2)
        .compute_at(Halide::Func(funcs["upsampledx$4"]), x)
        ;
        Halide::Func(funcs["interpolated$4"])
        .split(x, x, _x72, 8)
        .split(y, y, _y73, 16)
        .reorder(_y73, c, y, _x72, x)
        .reorder_storage(x, c, y)
        .vectorize(_y73, 2)
        .compute_at(Halide::Func(funcs["upsampledx$4"]), _x129)
        ;
        Halide::Func(funcs["interpolated$5"])
        .split(x, x, _x75, 8)
        .split(y, y, _y76, 2)
        .split(c, c, _c77, 8)
        .reorder(_c77, c, _y76, _x75, y, x)
        .reorder_storage(y, x, c)
        .vectorize(_c77, 4)
        .compute_at(Halide::Func(funcs["upsampledx$6"]), c)
        ;
        Halide::Func(funcs["interpolated$6"])
        .split(x, x, _x78, 4)
        .split(y, y, _y79, 4)
        .reorder(_y79, _x78, c, y, x)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["upsampledx$6"]), _c137)
        ;
        Halide::Func(funcs["interpolated$7"])
        .split(x, x, _x81, 8)
        .split(y, y, _y82, 8)
        .split(c, c, _c83, 8)
        .reorder(_x81, _y82, y, x, _c83, c)
        .reorder_storage(c, x, y)
        .vectorize(_x81, 2)
        .compute_at(Halide::Func(funcs["upsampledx$7"]), _y139)
        ;
        Halide::Func(funcs["interpolated$8"])
        .split(x, x, _x84, 8)
        .split(y, y, _y85, 2)
        .reorder(c, _x84, _y85, y, x)
        .reorder_storage(y, x, c)
        .vectorize(c, 4)
        .compute_at(Halide::Func(funcs["upsampledx$8"]), _c143)
        ;
        Halide::Func(funcs["interpolated$9"])
        .split(x, x, _x87, 16)
        .split(y, y, _y88, 4)
        .reorder(_x87, _y88, c, x, y)
        .reorder_storage(x, c, y)
        .vectorize(_x87, 2)
        .compute_at(Halide::Func(funcs["upsampledx$10"]), c)
        ;
        Halide::Func(funcs["normalize"])
        .split(x, x, _x90, 8)
        .split(y, y, _y91, 2)
        .reorder(_x90, _y91, x, c, y)
        .reorder_storage(y, c, x)
        .vectorize(_x90, 4)
        .compute_at(Halide::Func(funcs["final"]), _c149)
        ;
        Halide::Func(funcs["upsampled$10"])
        .split(x, x, _x93, 4)
        .split(y, y, _y94, 32)
        .reorder(_y94, _x93, x, c, y)
        .reorder_storage(y, x, c)
        .vectorize(_y94, 8)
        .compute_at(Halide::Func(funcs["interpolated$11"]), _y64)
        ;
        Halide::Func(funcs["upsampled$2"])
        .split(x, x, _x96, 2)
        .split(c, c, _c98, 16)
        .reorder(_c98, _x96, x, c, y)
        .reorder_storage(c, y, x)
        .vectorize(_c98, 4)
        .compute_at(Halide::Func(funcs["interpolated$3"]), _c71)
        ;
        Halide::Func(funcs["upsampled$3"])
        .split(x, x, _x99, 32)
        .split(c, c, _c101, 2)
        .reorder(_x99, x, _c101, y, c)
        .reorder_storage(c, y, x)
        .vectorize(_x99, 8)
        .compute_at(Halide::Func(funcs["interpolated$4"]), c)
        ;
        Halide::Func(funcs["upsampled$4"])
        .split(y, y, _y103, 4)
        .reorder(_y103, x, y, c)
        .reorder_storage(c, y, x)
        .vectorize(_y103, 2)
        .compute_at(Halide::Func(funcs["upsampledx$6"]), x)
        ;
        Halide::Func(funcs["upsampled$5"])
        .split(x, x, _x105, 8)
        .split(y, y, _y106, 4)
        .split(c, c, _c107, 2)
        .reorder(_x105, _y106, _c107, c, x, y)
        .reorder_storage(y, x, c)
        .vectorize(_x105, 2)
        .compute_at(Halide::Func(funcs["interpolated$6"]), _y79)
        ;
        Halide::Func(funcs["upsampled$6"])
        .split(c, c, _c110, 4)
        .reorder(x, _c110, c, y)
        .reorder_storage(c, x, y)
        .vectorize(x, 8)
        .compute_at(Halide::Func(funcs["upsampledx$8"]), y)
        ;
        Halide::Func(funcs["upsampled$7"])
        .split(x, x, _x111, 4)
        .split(y, y, _y112, 2)
        .split(c, c, _c113, 2)
        .reorder(_c113, _y112, y, c, _x111, x)
        .reorder_storage(c, x, y)
        .compute_at(Halide::Func(funcs["interpolated$8"]), c)
        ;
        Halide::Func(funcs["upsampled$8"])
        .split(x, x, _x114, 8)
        .split(y, y, _y115, 32)
        .reorder(_y115, _x114, c, x, y)
        .reorder_storage(x, c, y)
        .vectorize(_y115, 4)
        .compute_at(Halide::Func(funcs["upsampledx$10"]), x)
        ;
        Halide::Func(funcs["upsampled$9"])
        .split(x, x, _x117, 8)
        .split(y, y, _y118, 8)
        .split(c, c, _c119, 2)
        .reorder(_x117, _y118, _c119, x, y, c)
        .reorder_storage(x, y, c)
        .vectorize(_x117, 2)
        .compute_at(Halide::Func(funcs["upsampledx$10"]), y)
        ;
        Halide::Func(funcs["upsampledx$10"])
        .split(x, x, _x120, 8)
        .split(y, y, _y121, 8)
        .split(c, c, _c122, 4)
        .reorder(_y121, _c122, y, _x120, c, x)
        .reorder_storage(y, x, c)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["upsampledx$2"])
        .split(x, x, _x123, 64)
        .split(c, c, _c125, 2)
        .reorder(_x123, _c125, y, x, c)
        .reorder_storage(c, y, x)
        .vectorize(_x123, 8)
        .compute_at(Halide::Func(funcs["upsampled$2"]), _c98)
        ;
        Halide::Func(funcs["upsampledx$3"])
        .split(x, x, _x126, 2)
        .split(y, y, _y127, 8)
        .reorder(_y127, c, _x126, x, y)
        .reorder_storage(y, x, c)
        .vectorize(_y127, 2)
        .compute_at(Halide::Func(funcs["upsampledx$4"]), _c131)
        ;
        Halide::Func(funcs["upsampledx$4"])
        .split(x, x, _x129, 8)
        .split(y, y, _y130, 8)
        .split(c, c, _c131, 4)
        .reorder(_y130, _x129, y, _c131, x, c)
        .reorder_storage(x, y, c)
        .vectorize(_y130, 4)
        .compute_root()
        ;
        Halide::Func(funcs["upsampledx$5"])
        .split(x, x, _x132, 8)
        .reorder(_x132, c, y, x)
        .reorder_storage(c, y, x)
        .vectorize(_x132, 4)
        .compute_at(Halide::Func(funcs["upsampledx$6"]), _x135)
        ;
        Halide::Func(funcs["upsampledx$6"])
        .split(x, x, _x135, 4)
        .split(y, y, _y136, 8)
        .split(c, c, _c137, 2)
        .reorder(_y136, _c137, _x135, y, c, x)
        .reorder_storage(y, x, c)
        .vectorize(_y136, 2)
        .compute_root()
        ;
        Halide::Func(funcs["upsampledx$7"])
        .split(y, y, _y139, 8)
        .split(c, c, _c140, 4)
        .reorder(_y139, _c140, y, x, c)
        .reorder_storage(x, y, c)
        .compute_at(Halide::Func(funcs["upsampledx$8"]), x)
        ;
        Halide::Func(funcs["upsampledx$8"])
        .split(x, x, _x141, 32)
        .split(c, c, _c143, 8)
        .reorder(_x141, _c143, c, x, y)
        .reorder_storage(x, c, y)
        .vectorize(_x141, 4)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["upsampledx$9"])
        .split(x, x, _x144, 8)
        .split(c, c, _c146, 32)
        .reorder(_c146, _x144, x, c, y)
        .reorder_storage(c, y, x)
        .vectorize(_c146, 4)
        .compute_at(Halide::Func(funcs["upsampledx$10"]), _x120)
        ;
        Halide::Func(funcs["final"])
        .split(x, x, _x147, 64)
        .split(y, y, _y148, 8)
        .split(c, c, _c149, 4)
        .reorder(_x147, x, _c149, _y148, c, y)
        .reorder_storage(y, x, c)
        .vectorize(_x147, 8)
        .compute_root()
        ;
        

        _autotune_timing_stub(final);
    };

    int sched;
    char *target = getenv("HL_TARGET");
    if (target && std::string(target) == "ptx") {
        sched = 4;
    } else {
        sched = 2;
    }

    switch (sched) {
    case 0:
    {
        //std::cout << "Flat schedule." << std::endl;
        for (unsigned int l = 0; l < levels; ++l) {
            downsampled[l].compute_root();
            interpolated[l].compute_root();
        }
        final.compute_root();
        break;
    }
    case 1:
    {
        //std::cout << "Flat schedule with vectorization." << std::endl;
        for (unsigned int l = 0; l < levels; ++l) {
            downsampled[l].compute_root().vectorize(x,4);
            interpolated[l].compute_root().vectorize(x,4);
        }
        final.compute_root();
        break;
    }
    case 2:
    {
        Var xi, yi;
        //std::cout << "Flat schedule with parallelization + vectorization." << std::endl;
        clamped.compute_root().parallel(y).reorder(c, x, y).reorder_storage(c, x, y).vectorize(c, 4);
        for (unsigned int l = 1; l < levels-1; ++l) {
            if (l > 0) downsampled[l].compute_root().parallel(y).reorder(c, x, y).reorder_storage(c, x, y).vectorize(c, 4);
            interpolated[l].compute_root().parallel(y).reorder(c, x, y).reorder_storage(c, x, y).vectorize(c, 4);
            interpolated[l].unroll(x, 2).unroll(y, 2);
        }
        final.reorder(c, x, y).bound(c, 0, 3).parallel(y);
        final.tile(x, y, xi, yi, 2, 2).unroll(xi).unroll(yi);
        break;
    }
    case 3:
    {
        //std::cout << "Flat schedule with vectorization sometimes." << std::endl;
        for (unsigned int l = 0; l < levels; ++l) {
            if (l + 4 < levels) {
                Var yo,yi;
                downsampled[l].compute_root().vectorize(x,4);
                interpolated[l].compute_root().vectorize(x,4);
            } else {
                downsampled[l].compute_root();
                interpolated[l].compute_root();
            }
        }
        final.compute_root();
        break;
    }
    case 4:
    {
        //std::cout << "GPU schedule." << std::endl;

        // Some gpus don't have enough memory to process the entire
        // image, so we process the image in tiles.
        Var yo, yi, xo, xi;
        final.reorder(c, x, y).bound(c, 0, 3).vectorize(x, 4);
        final.tile(x, y, xo, yo, xi, yi, input.width()/4, input.height()/4);
        normalize.compute_at(final, xo).reorder(c, x, y).cuda_tile(x, y, 16, 16).unroll(c);

        // Start from level 1 to save memory - level zero will be computed on demand
        for (unsigned int l = 1; l < levels; ++l) {
            int tile_size = 32 >> l;
            if (tile_size < 1) tile_size = 1;
            if (tile_size > 16) tile_size = 16;
            downsampled[l].compute_root().cuda_tile(x, y, c, tile_size, tile_size, 4);
            interpolated[l].compute_at(final, xo).cuda_tile(x, y, c, tile_size, tile_size, 4);
        }

        break;
    }
    default:
        assert(0 && "No schedule with this number.");
    }

    BASELINE_HOOK(final);

#if 0
    // JIT compile the pipeline eagerly, so we don't interfere with timing
    final.compile_jit();

    // Image<float> in_png = load<float>(argv[1]);
    Image<float> out(2048, 2048, 3);
    // assert(in_png.channels() == 4);
    // input.set(in_png);
    final.infer_input_bounds(out);

    std::cout << "Running... " << std::endl;
    double min = std::numeric_limits<double>::infinity();
    const unsigned int iters = 20;

    for (unsigned int x = 0; x < iters; ++x) {
        double before = now();
        final.realize(out);
        double after = now();
        double amt = after - before;

        std::cout << "   " << amt * 1000 << std::endl;
        if (amt < min) min = amt;

    }
    std::cout << " took " << min * 1000 << " msec." << std::endl;

    // vector<Argument> args;
    // args.push_back(input);
    // final.compile_to_assembly("test.s", args);
    // save(out, argv[2]);
#endif
}
