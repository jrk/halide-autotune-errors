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

#include "Halide.h"
#include <stdio.h>

#define AUTOTUNE_HOOK(x)
#define BASELINE_HOOK(x)

using namespace Halide;

int main(int argc, char **argv) {
  // if (argc < 2) {
  //     printf("Usage: bilateral_grid <s_sigma>\n");
  //     // printf("Spatial sigma is a compile-time parameter, please provide it as an argument.\n"
  //     //        "(llvm's ptx backend doesn't handle integer mods by non-consts yet)\n");
  //     return 0;
  // }

    ImageParam input(Float(32), 2);
    float r_sigma = 0.1;
   // int s_sigma = atoi(argv[1]);
    int s_sigma = 4;
    Var x("x"), y("y"), z("z"), c("c");

    // Add a boundary condition 
    Func clamped("clamped");
    clamped(x, y) = input(clamp(x, 0, input.width()-1),
                          clamp(y, 0, input.height()-1));

    // Construct the bilateral grid 
    RDom r(0, s_sigma, 0, s_sigma);
    Expr val = clamped(x * s_sigma + r.x - s_sigma/2, y * s_sigma + r.y - s_sigma/2);
    val = clamp(val, 0.0f, 1.0f);
    Expr zi = cast<int>(val * (1.0f/r_sigma) + 0.5f);
    Func grid("grid"), histogram("histogram");    
    histogram(x, y, zi, c) += select(c == 0, val, 1.0f);

    // Introduce a dummy function, so we can schedule the histogram within it
    grid(x, y, z, c) = histogram(x, y, z, c);

    // Blur the grid using a five-tap filter
    Func blurx("blurx"), blury("blury"), blurz("blurz");
    blurx(x, y, z) = grid(x-2, y, z) + grid(x-1, y, z)*4 + grid(x, y, z)*6 + grid(x+1, y, z)*4 + grid(x+2, y, z);
    blury(x, y, z) = blurx(x, y-2, z) + blurx(x, y-1, z)*4 + blurx(x, y, z)*6 + blurx(x, y+1, z)*4 + blurx(x, y+2, z);
    blurz(x, y, z) = blury(x, y, z-2) + blury(x, y, z-1)*4 + blury(x, y, z)*6 + blury(x, y, z+1)*4 + blury(x, y, z+2);

    // Take trilinear samples to compute the output
    val = clamp(clamped(x, y), 0.0f, 1.0f);
    Expr zv = val * (1.0f/r_sigma);
    zi = cast<int>(zv);
    Expr zf = zv - zi;
    Expr xf = cast<float>(x % s_sigma) / s_sigma;
    Expr yf = cast<float>(y % s_sigma) / s_sigma;
    Expr xi = x/s_sigma;
    Expr yi = y/s_sigma;
    Func interpolated("interpolated");
    interpolated(x, y) = 
        lerp(lerp(lerp(blurz(xi, yi, zi), blurz(xi+1, yi, zi), xf),
                  lerp(blurz(xi, yi+1, zi), blurz(xi+1, yi+1, zi), xf), yf),
             lerp(lerp(blurz(xi, yi, zi+1), blurz(xi+1, yi, zi+1), xf),
                  lerp(blurz(xi, yi+1, zi+1), blurz(xi+1, yi+1, zi+1), xf), yf), zf);

    // Normalize
    Func bilateral_grid("bilateral_grid");
    bilateral_grid(x, y) = interpolated(x, y, 0)/interpolated(x, y, 1);

Halide::Var _x0, _y1, _x2, _y3, _z4, _c5, _x6, _y7, _z8, _x9, _z11, _x12, _y13, _z14, _x15, _y16, _x17, _y18;
clamped .split(x, x, _x0, 32) .split(y, y, _y1, 16)
.reorder(_x0, x, _y1, y) .reorder_storage(y, x)
.vectorize(_x0, 2)
.compute_root()
;
grid .split(x, x, _x2, 4) .split(y, y, _y3, 2) .split(z, z, _z4, 16) .split(c, c, _c5, 4)
.reorder(_x2, _y3, _z4, z, _c5, x, y, c) .reorder_storage(z, x, y, c)
.vectorize(_x2, 2)
.compute_at(blurx, _x6)
;
blurx .split(x, x, _x6, 4) .split(y, y, _y7, 32) .split(z, z, _z8, 8)
.reorder(_y7, y, _x6, _z8, x, z) .reorder_storage(z, y, x)
.vectorize(_y7, 2)
.compute_root()
;
blury .split(x, x, _x9, 16) .split(z, z, _z11, 8)
.reorder(_x9, x, _z11, y, z) .reorder_storage(y, x, z)

.compute_at(blurz, _y13)
;
blurz .split(x, x, _x12, 4) .split(y, y, _y13, 4) .split(z, z, _z14, 2)
.reorder(_y13, y, _z14, z, _x12, x) .reorder_storage(z, y, x)

.compute_at(bilateral_grid, x)
;
interpolated .split(x, x, _x15, 16) .split(y, y, _y16, 2)
.reorder(_y16, _x15, y, x) .reorder_storage(x, y)

.compute_at(bilateral_grid, _y18)
;
bilateral_grid .split(x, x, _x17, 64) .split(y, y, _y18, 8)
.reorder(_x17, _y18, x, y) .reorder_storage(y, x)
.vectorize(_x17, 8)
.compute_root()
;


_autotune_timing_stub(bilateral_grid);;

    char *target = getenv("HL_TARGET");
    if (target && std::string(target) == "ptx") {

        // GPU schedule
        grid.compute_root().reorder(z, c, x, y).cuda_tile(x, y, 8, 8);

        // Compute the histogram into shared memory before spilling it to global memory
        histogram.store_at(grid, Var("blockidx")).compute_at(grid, Var("threadidx"));

        blurx.compute_root().cuda_tile(x, y, z, 16, 16, 1);
        blury.compute_root().cuda_tile(x, y, z, 16, 16, 1);
        blurz.compute_root().cuda_tile(x, y, z, 8, 8, 4);
        bilateral_grid.compute_root().cuda_tile(x, y, s_sigma, s_sigma);
    } else {

        // CPU schedule
        grid.compute_root().reorder(c, z, x, y).parallel(y);
        histogram.compute_at(grid, x).unroll(c);
        blurx.compute_root().parallel(z).vectorize(x, 4);
        blury.compute_root().parallel(z).vectorize(x, 4);
        blurz.compute_root().parallel(z).vectorize(x, 4);
        bilateral_grid.compute_root().parallel(y).vectorize(x, 4);
    }

    BASELINE_HOOK(bilateral_grid);

   //bilateral_grid.compile_to_file("bilateral_grid", r_sigma, input);

    return 0;
}



