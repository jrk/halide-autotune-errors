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
    ImageParam input(Float(32), 3);

    const unsigned int levels = 10;

    Func downsampled[levels];
    Func downx[levels];
    Func interpolated[levels];
    Func upsampled[levels];
    Func upsampledx[levels];
    Var x("x"), y("y"), c("c");

    Func clamped;
    clamped(x, y, c) = input(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1), c);

    // This triggers a bug in llvm 3.3 (3.2 and trunk are fine), so we
    // rewrite it in a way that doesn't trigger the bug. The rewritten
    // form assumes the input alpha is zero or one.
    // downsampled[0](x, y, c) = select(c < 3, clamped(x, y, c) * clamped(x, y, 3), clamped(x, y, 3));
    downsampled[0](x, y, c) = clamped(x, y, c) * clamped(x, y, 3);

    for (unsigned int l = 1; l < levels; ++l) {
        downx[l](x, y, c) = (downsampled[l-1](x*2-1, y, c) +
                             2.0f * downsampled[l-1](x*2, y, c) +
                             downsampled[l-1](x*2+1, y, c)) * 0.25f;
        downsampled[l](x, y, c) = (downx[l](x, y*2-1, c) +
                                   2.0f * downx[l](x, y*2, c) +
                                   downx[l](x, y*2+1, c)) * 0.25f;
    }
    interpolated[levels-1](x, y, c) = downsampled[levels-1](x, y, c);
    for (unsigned int l = levels-2; l < levels; --l) {
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

//std::cout << "Finished function setup." << std::endl;

    {
    {
        std::map<std::string, Halide::Internal::Function> funcs = Halide::Internal::find_transitive_calls((final).function());

        Halide::Var _x0, _y1, _x3, _y4, _x6, _x12, _y13, _c14, _x15, _y16, _c17, _x18, _y19, _x21, _c23, _x24, _c26, _x27, _y28, _c29, _x30, _y31, _c32, _y34, _c35, _y37, _c38, _x39, _y40, _c41, _x42, _x45, _c47, _x48, _y49, _c50, _x51, _y52, _c53, _x54, _y55, _c56, _x57, _y58, _x60, _y61, _x63, _c65, _x66, _x69, _y70, _x72, _y73, _x75, _y76, _c77, _x78, _y79, _x81, _c83, _x84, _c86, _x87, _y88, _c89, _x90, _y91, _c92, _x93, _y94, _c95, _x96, _y97, _x102, _y103, _c104, _x105, _c107, _y109, _c110, _x111, _x114, _y115, _x117, _y118, _c119, _c122, _x123, _y124, _c125, _x126, _y127, _c128, _x129, _y130, _c131, _x132, _y133, _x135, _y139, _c140, _x141, _c143, _x144, _y145, _c146, _x147, _y148, _c149;
        Halide::Func(funcs["f0"])
        .split(x, x, _x0, 4)
        .split(y, y, _y1, 2)
        .reorder(c, _y1, _x0, y, x)
        .reorder_storage(y, x, c)
        .vectorize(c, 4)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["f1"])
        .split(x, x, _x3, 16)
        .split(y, y, _y4, 4)
        .reorder(_x3, _y4, x, y, c)
        .reorder_storage(c, x, y)
        .vectorize(_x3, 8)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["f11"])
        .split(x, x, _x6, 8)
        .reorder(y, c, _x6, x)
        .reorder_storage(y, x, c)
        .compute_at(Halide::Func(funcs["f1"]), _x3)
        ;
        Halide::Func(funcs["f12"])
        .reorder(y, x, c)
        .reorder_storage(x, c, y)
        .vectorize(y, 4)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["f13"])
        .split(x, x, _x12, 8)
        .split(y, y, _y13, 8)
        .split(c, c, _c14, 8)
        .reorder(_c14, _x12, _y13, c, y, x)
        .reorder_storage(x, y, c)
        .vectorize(_c14, 4)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["f14"])
        .split(x, x, _x15, 2)
        .split(y, y, _y16, 8)
        .split(c, c, _c17, 4)
        .reorder(_y16, _x15, y, _c17, x, c)
        .reorder_storage(c, y, x)
        .compute_root()
        ;
        Halide::Func(funcs["f15"])
        .split(x, x, _x18, 8)
        .split(y, y, _y19, 4)
        .reorder(_y19, _x18, x, c, y)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["f5"]), _y127)
        ;
        Halide::Func(funcs["f16"])
        .split(x, x, _x21, 32)
        .split(c, c, _c23, 8)
        .reorder(_x21, _c23, x, y, c)
        .reorder_storage(x, y, c)
        .vectorize(_x21, 4)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["f17"])
        .split(x, x, _x24, 8)
        .split(c, c, _c26, 4)
        .reorder(_c26, _x24, c, y, x)
        .reorder_storage(c, x, y)
        .vectorize(_c26, 2)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["f18"])
        .split(x, x, _x27, 2)
        .split(y, y, _y28, 8)
        .split(c, c, _c29, 16)
        .reorder(_c29, _x27, c, _y28, x, y)
        .reorder_storage(y, x, c)
        .vectorize(_c29, 2)
        .compute_at(Halide::Func(funcs["f8"]), _c140)
        ;
        Halide::Func(funcs["f19"])
        .split(x, x, _x30, 2)
        .split(y, y, _y31, 8)
        .split(c, c, _c32, 64)
        .reorder(_c32, _x30, _y31, x, c, y)
        .reorder_storage(y, c, x)
        .vectorize(_c32, 8)
        .compute_at(Halide::Func(funcs["f9"]), _x141)
        ;
        Halide::Func(funcs["f2"])
        .split(y, y, _y34, 8)
        .split(c, c, _c35, 2)
        .reorder(_y34, _c35, y, c, x)
        .reorder_storage(y, x, c)
        .compute_root()
        ;
        Halide::Func(funcs["f20"])
        .split(y, y, _y37, 8)
        .split(c, c, _c38, 8)
        .reorder(_y37, y, _c38, x, c)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["final"]), _y148)
        ;
        Halide::Func(funcs["f21"])
        .split(x, x, _x39, 4)
        .split(y, y, _y40, 8)
        .split(c, c, _c41, 4)
        .reorder(_x39, _y40, y, _c41, x, c)
        .reorder_storage(y, x, c)
        .compute_at(Halide::Func(funcs["final"]), y)
        ;
        Halide::Func(funcs["f22"])
        .split(x, x, _x42, 16)
        .reorder(_x42, c, x, y)
        .reorder_storage(c, x, y)
        .vectorize(_x42, 2)
        .compute_at(Halide::Func(funcs["f41"]), _x102)
        ;
        Halide::Func(funcs["f23"])
        .split(x, x, _x45, 2)
        .split(c, c, _c47, 64)
        .reorder(_c47, c, _x45, y, x)
        .reorder_storage(c, y, x)
        .vectorize(_c47, 8)
        .compute_at(Halide::Func(funcs["f42"]), _c107)
        ;
        Halide::Func(funcs["f24"])
        .split(x, x, _x48, 4)
        .split(y, y, _y49, 32)
        .split(c, c, _c50, 2)
        .reorder(_y49, _x48, y, _c50, x, c)
        .reorder_storage(y, c, x)
        .vectorize(_y49, 4)
        .compute_at(Halide::Func(funcs["f41"]), y)
        ;
        Halide::Func(funcs["f25"])
        .split(x, x, _x51, 4)
        .split(y, y, _y52, 64)
        .split(c, c, _c53, 2)
        .reorder(_y52, y, _x51, _c53, c, x)
        .reorder_storage(y, x, c)
        .vectorize(_y52, 8)
        .compute_at(Halide::Func(funcs["f44"]), y)
        ;
        Halide::Func(funcs["f26"])
        .split(x, x, _x54, 2)
        .split(y, y, _y55, 32)
        .split(c, c, _c56, 2)
        .reorder(_y55, y, _x54, _c56, x, c)
        .reorder_storage(x, c, y)
        .vectorize(_y55, 8)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["f27"])
        .split(x, x, _x57, 4)
        .split(y, y, _y58, 2)
        .reorder(_x57, c, x, _y58, y)
        .reorder_storage(c, x, y)
        .compute_at(Halide::Func(funcs["f46"]), _x117)
        ;
        Halide::Func(funcs["f28"])
        .split(x, x, _x60, 32)
        .split(y, y, _y61, 4)
        .reorder(_x60, _y61, x, y, c)
        .reorder_storage(c, y, x)
        .vectorize(_x60, 4)
        .compute_at(Halide::Func(funcs["f47"]), c)
        ;
        Halide::Func(funcs["f29"])
        .split(x, x, _x63, 2)
        .split(c, c, _c65, 16)
        .reorder(_c65, _x63, y, c, x)
        .reorder_storage(c, x, y)
        .vectorize(_c65, 4)
        .compute_root()
        ;
        Halide::Func(funcs["f3"])
        .split(x, x, _x66, 16)
        .reorder(_x66, x, y, c)
        .reorder_storage(y, c, x)
        .vectorize(_x66, 2)
        .compute_root()
        ;
        Halide::Func(funcs["f30"])
        .split(x, x, _x69, 16)
        .split(y, y, _y70, 8)
        .reorder(_x69, _y70, x, c, y)
        .reorder_storage(c, x, y)
        .vectorize(_x69, 4)
        .compute_at(Halide::Func(funcs["final"]), c)
        ;
        Halide::Func(funcs["f31"])
        .split(x, x, _x72, 8)
        .split(y, y, _y73, 2)
        .reorder(_x72, _y73, x, y, c)
        .reorder_storage(y, x, c)
        .vectorize(_x72, 2)
        .compute_at(Halide::Func(funcs["final"]), x)
        ;
        Halide::Func(funcs["f32"])
        .split(x, x, _x75, 8)
        .split(y, y, _y76, 8)
        .split(c, c, _c77, 8)
        .reorder(_c77, _x75, _y76, y, x, c)
        .reorder_storage(x, c, y)
        .compute_at(Halide::Func(funcs["f22"]), _x42)
        ;
        Halide::Func(funcs["f33"])
        .split(x, x, _x78, 8)
        .split(y, y, _y79, 8)
        .reorder(_y79, _x78, y, x, c)
        .reorder_storage(x, y, c)
        .compute_at(Halide::Func(funcs["f41"]), _y103)
        ;
        Halide::Func(funcs["f34"])
        .split(x, x, _x81, 2)
        .split(c, c, _c83, 8)
        .reorder(_c83, c, _x81, x, y)
        .reorder_storage(x, y, c)
        .compute_at(Halide::Func(funcs["f24"]), _y49)
        ;
        Halide::Func(funcs["f35"])
        .split(x, x, _x84, 8)
        .split(c, c, _c86, 8)
        .reorder(_x84, _c86, c, y, x)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["f25"]), y)
        ;
        Halide::Func(funcs["f36"])
        .split(x, x, _x87, 8)
        .split(y, y, _y88, 2)
        .split(c, c, _c89, 8)
        .reorder(_c89, _x87, c, _y88, y, x)
        .reorder_storage(y, c, x)
        .vectorize(_c89, 4)
        .compute_at(Halide::Func(funcs["f26"]), y)
        ;
        Halide::Func(funcs["f37"])
        .split(x, x, _x90, 8)
        .split(y, y, _y91, 4)
        .split(c, c, _c92, 4)
        .reorder(_x90, x, _c92, _y91, c, y)
        .reorder_storage(x, y, c)
        .compute_at(Halide::Func(funcs["f26"]), c)
        ;
        Halide::Func(funcs["f38"])
        .split(x, x, _x93, 8)
        .split(y, y, _y94, 4)
        .split(c, c, _c95, 64)
        .reorder(_c95, _y94, y, _x93, c, x)
        .reorder_storage(x, c, y)
        .vectorize(_c95, 8)
        .compute_at(Halide::Func(funcs["f47"]), x)
        ;
        Halide::Func(funcs["f4"])
        .split(x, x, _x96, 64)
        .split(y, y, _y97, 8)
        .reorder(_x96, _y97, x, y, c)
        .reorder_storage(y, x, c)
        .vectorize(_x96, 8)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["f40"])
        .reorder(y, c, x)
        .reorder_storage(y, x, c)
        .compute_at(Halide::Func(funcs["f30"]), _x69)
        ;
        Halide::Func(funcs["f41"])
        .split(x, x, _x102, 4)
        .split(y, y, _y103, 4)
        .split(c, c, _c104, 8)
        .reorder(_x102, _c104, _y103, y, c, x)
        .reorder_storage(x, c, y)
        .vectorize(_x102, 2)
        .compute_root()
        ;
        Halide::Func(funcs["f42"])
        .split(x, x, _x105, 4)
        .split(c, c, _c107, 64)
        .reorder(_c107, c, y, _x105, x)
        .reorder_storage(x, c, y)
        .vectorize(_c107, 8)
        .compute_at(Halide::Func(funcs["f41"]), _c104)
        ;
        Halide::Func(funcs["f43"])
        .split(y, y, _y109, 4)
        .split(c, c, _c110, 4)
        .reorder(_y109, _c110, c, x, y)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["f33"]), _y79)
        ;
        Halide::Func(funcs["f44"])
        .split(x, x, _x111, 32)
        .reorder(_x111, y, c, x)
        .reorder_storage(y, x, c)
        .vectorize(_x111, 4)
        .compute_at(Halide::Func(funcs["f41"]), c)
        ;
        Halide::Func(funcs["f45"])
        .split(x, x, _x114, 4)
        .split(y, y, _y115, 4)
        .reorder(_y115, _x114, y, x, c)
        .reorder_storage(c, x, y)
        .compute_at(Halide::Func(funcs["f41"]), x)
        ;
        Halide::Func(funcs["f46"])
        .split(x, x, _x117, 8)
        .split(y, y, _y118, 2)
        .split(c, c, _c119, 16)
        .reorder(_c119, _x117, _y118, c, x, y)
        .reorder_storage(x, y, c)
        .vectorize(_c119, 2)
        .compute_at(Halide::Func(funcs["f26"]), x)
        ;
        Halide::Func(funcs["f47"])
        .split(c, c, _c122, 16)
        .reorder(_c122, y, c, x)
        .reorder_storage(c, x, y)
        .vectorize(_c122, 2)
        .compute_root()
        ;
        Halide::Func(funcs["f48"])
        .split(x, x, _x123, 4)
        .split(y, y, _y124, 8)
        .split(c, c, _c125, 2)
        .reorder(_c125, _y124, _x123, c, y, x)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["f38"]), _c95)
        ;
        Halide::Func(funcs["f5"])
        .split(x, x, _x126, 2)
        .split(y, y, _y127, 16)
        .split(c, c, _c128, 4)
        .reorder(_y127, _x126, y, _c128, c, x)
        .reorder_storage(y, c, x)
        .vectorize(_y127, 2)
        .compute_root()
        ;
        Halide::Func(funcs["f50"])
        .split(x, x, _x129, 4)
        .split(y, y, _y130, 4)
        .split(c, c, _c131, 8)
        .reorder(_y130, _c131, _x129, x, c, y)
        .reorder_storage(x, c, y)
        .vectorize(_y130, 2)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["f6"])
        .split(x, x, _x132, 32)
        .split(y, y, _y133, 4)
        .reorder(_x132, x, _y133, y, c)
        .reorder_storage(y, x, c)
        .vectorize(_x132, 4)
        .compute_root()
        ;
        Halide::Func(funcs["f7"])
        .split(x, x, _x135, 4)
        .reorder(c, y, _x135, x)
        .reorder_storage(c, x, y)
        .vectorize(c, 4)
        .compute_root()
        ;
        Halide::Func(funcs["f8"])
        .split(y, y, _y139, 64)
        .split(c, c, _c140, 2)
        .reorder(_y139, _c140, c, y, x)
        .reorder_storage(x, y, c)
        .vectorize(_y139, 8)
        .compute_root()
        ;
        Halide::Func(funcs["f9"])
        .split(x, x, _x141, 4)
        .split(c, c, _c143, 8)
        .reorder(_x141, _c143, c, y, x)
        .reorder_storage(y, c, x)
        .compute_at(Halide::Func(funcs["f29"]), _c65)
        ;
        Halide::Func(funcs["normalize"])
        .split(x, x, _x144, 4)
        .split(y, y, _y145, 4)
        .split(c, c, _c146, 16)
        .reorder(_c146, _y145, _x144, c, y, x)
        .reorder_storage(x, c, y)
        .vectorize(_c146, 2)
        .compute_at(Halide::Func(funcs["final"]), _c149)
        ;
        Halide::Func(funcs["final"])
        .split(x, x, _x147, 4)
        .split(y, y, _y148, 2)
        .split(c, c, _c149, 8)
        .reorder(_c149, _y148, _x147, c, y, x)
        .reorder_storage(x, c, y)
        .vectorize(_c149, 2)
        .compute_root()
        ;
        

        _autotune_timing_stub(final);
    };
    }

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

    {
    BASELINE_HOOK(final);
    }

    /*
    // JIT compile the pipeline eagerly, so we don't interfere with timing
    final.compile_jit();

    Image<float> in_png = load<float>(argv[1]);
    Image<float> out(in_png.width(), in_png.height(), 3);
    assert(in_png.channels() == 4);
    input.set(in_png);

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

    vector<Argument> args;
    args.push_back(input);
    final.compile_to_assembly("test.s", args);

    save(out, argv[2]);
    */
}
