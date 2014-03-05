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

    const unsigned int levels = 3;

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

        Halide::Var _x0, _c2, _x3, _c5, _c8, _x9, _y10, _c11, _x12, _y13, _c14, _x15, _y16, _y19, _c20, _x21, _y22, _c23, _y25, _c26, _x27, _y28, _c32, _c35, _x36, _y37, _c38, _x39, _y40, _c41, _x42, _y43, _c44;
        Halide::Func(funcs["clamped"])
        .split(x, x, _x0, 2)
        .split(c, c, _c2, 16)
        .reorder(_c2, _x0, x, y, c)
        .reorder_storage(c, y, x)
        .vectorize(_c2, 8)
        .parallel(c)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled"])
        .split(x, x, _x3, 8)
        .split(c, c, _c5, 16)
        .reorder(_c5, _x3, c, y, x)
        .reorder_storage(x, y, c)
        .vectorize(_c5, 4)
        .parallel(x)
        .compute_root()
        ;
        Halide::Func(funcs["downsampled$2"])
        .split(c, c, _c8, 16)
        .reorder(_c8, c, y, x)
        .reorder_storage(c, x, y)
        .vectorize(_c8, 2)
        .compute_at(Halide::Func(funcs["interpolated$3"]), y)
        ;
        Halide::Func(funcs["downsampled$3"])
        .split(x, x, _x9, 4)
        .split(y, y, _y10, 4)
        .split(c, c, _c11, 8)
        .reorder(_x9, _c11, _y10, c, x, y)
        .reorder_storage(c, x, y)
        .compute_at(Halide::Func(funcs["interpolated$2"]), _y19)
        ;
        Halide::Func(funcs["downx$2"])
        .split(x, x, _x12, 8)
        .split(y, y, _y13, 2)
        .split(c, c, _c14, 2)
        .reorder(_c14, _y13, _x12, x, c, y)
        .reorder_storage(y, c, x)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["downx$3"])
        .split(x, x, _x15, 4)
        .split(y, y, _y16, 64)
        .reorder(_y16, y, _x15, c, x)
        .reorder_storage(y, x, c)
        .vectorize(_y16, 8)
        .compute_at(Halide::Func(funcs["interpolated$3"]), y)
        ;
        Halide::Func(funcs["interpolated$2"])
        .split(y, y, _y19, 32)
        .split(c, c, _c20, 4)
        .reorder(_y19, x, _c20, y, c)
        .reorder_storage(x, y, c)
        .vectorize(_y19, 4)
        .compute_at(Halide::Func(funcs["interpolated$3"]), _c23)
        ;
        Halide::Func(funcs["interpolated$3"])
        .split(x, x, _x21, 2)
        .split(y, y, _y22, 16)
        .split(c, c, _c23, 4)
        .reorder(_y22, _x21, _c23, c, x, y)
        .reorder_storage(c, x, y)
        .vectorize(_y22, 4)
        .parallel(y)
        .compute_root()
        ;
        Halide::Func(funcs["interpolated$4"])
        .split(y, y, _y25, 2)
        .split(c, c, _c26, 16)
        .reorder(_c26, c, _y25, x, y)
        .reorder_storage(y, c, x)
        .vectorize(_c26, 2)
        .compute_at(Halide::Func(funcs["final"]), c)
        ;
        Halide::Func(funcs["normalize"])
        .split(x, x, _x27, 2)
        .split(y, y, _y28, 32)
        .reorder(_y28, _x27, x, y, c)
        .reorder_storage(c, x, y)
        .vectorize(_y28, 4)
        .compute_at(Halide::Func(funcs["final"]), _c44)
        ;
        Halide::Func(funcs["upsampled$2"])
        .split(c, c, _c32, 8)
        .reorder(_c32, x, c, y)
        .reorder_storage(y, x, c)
        .vectorize(_c32, 4)
        .compute_at(Halide::Func(funcs["interpolated$3"]), _y22)
        ;
        Halide::Func(funcs["upsampled$3"])
        .split(c, c, _c35, 32)
        .reorder(_c35, y, x, c)
        .reorder_storage(x, c, y)
        .vectorize(_c35, 8)
        .compute_at(Halide::Func(funcs["final"]), _y43)
        ;
        Halide::Func(funcs["upsampledx$2"])
        .split(x, x, _x36, 2)
        .split(y, y, _y37, 4)
        .split(c, c, _c38, 16)
        .reorder(_c38, _y37, c, _x36, y, x)
        .reorder_storage(y, x, c)
        .vectorize(_c38, 4)
        .compute_at(Halide::Func(funcs["interpolated$3"]), _x21)
        ;
        Halide::Func(funcs["upsampledx$3"])
        .split(x, x, _x39, 2)
        .split(y, y, _y40, 4)
        .split(c, c, _c41, 8)
        .reorder(_y40, _x39, y, _c41, c, x)
        .reorder_storage(x, y, c)
        .compute_at(Halide::Func(funcs["final"]), y)
        ;
        Halide::Func(funcs["final"])
        .split(x, x, _x42, 8)
        .split(y, y, _y43, 4)
        .split(c, c, _c44, 2)
        .reorder(_c44, _x42, c, x, _y43, y)
        .reorder_storage(x, c, y)
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
