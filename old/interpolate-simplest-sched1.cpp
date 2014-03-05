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

// Size to run with - override Makefile setting - this is hacked down to 2D
// #ifndef AUTOTUNE_N
#undef AUTOTUNE_N
#define AUTOTUNE_N 2048, 2048
// #endif

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

using namespace Halide;

#include <iostream>
#include <limits>

#include <sys/time.h>

using std::vector;

int main(int argc, char **argv) {
    ImageParam input(Float(32), 2, "input");

    Func upsampled("upsampled");
    Func upsampledx("upsampledx");
    Var x("x"), y("y");

    Func clamped("clamped");
    clamped(x, y) = input(x, y);

    upsampledx(x, y) = select((x % 2) == 0,
                              clamped(x, y),
                              clamped(x+1, y));
    upsampled(x, y) = upsampledx(x, y);

    Var xi("xi"), yi("yi");
    clamped.compute_root(); // passes if this is removed, switched to inline
    upsampled
        .split(y, y, yi, 8)
        .reorder(yi, y, x)
        .compute_root();
    upsampledx.compute_at(upsampled, yi);

    _autotune_timing_stub(upsampled);
}
