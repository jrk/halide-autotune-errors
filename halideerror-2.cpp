#include <Halide.h>
#include <stdio.h>
#include <sys/time.h>

#define AUTOTUNE_N 1024,1024
#define AUTOTUNE_TRIALS 3

inline void _autotune_timing_stub(Halide::Func& func) {
    func.compile_jit();
    func.infer_input_bounds(AUTOTUNE_N);
    timeval t1, t2;
    double rv = 0;
    for (int i = 0; i < AUTOTUNE_TRIALS; i++) {
      gettimeofday(&t1, NULL);
      func.realize(AUTOTUNE_N);
      gettimeofday(&t2, NULL);
      double t = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;
      if(i == 0 || t < rv)
        rv = t;
    }
    printf("{\"time\": %.10f}\n", rv);
    exit(0);
}

#include <Halide.h>
using namespace Halide;

int main(int argc, char **argv) {

    ImageParam in_img(UInt(16), 2);
    Func blur_x("blur_x"), blur_y("blur_y");
    Var x("x"), y("y"), xi("xi"), yi("yi");

    Func input("input");
    input(x,y) = in_img(clamp(x, 1, in_img.width()-1),
                        clamp(y, 1, in_img.height())-1);

    // The algorithm
    blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;

Halide::Var _x0, _x2, _y3, _x4, _y5;
input.compute_at(blur_y, _y5);
blur_x.compute_at(blur_y, _x4);
blur_y
.split(x, x, _x4, 4)
.split(y, y, _y5, 8)
.reorder(y, _y5, x, _x4)
.compute_root();


_autotune_timing_stub(blur_y);;

    return 0;
}
