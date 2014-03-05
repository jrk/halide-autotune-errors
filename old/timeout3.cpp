#include <Halide.h>
#include <stdio.h>
#include <sys/time.h>

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

Halide::Var _x0, _x2, _y3, _c4, _x5, _y6, _z7, _c8, _x9, _y10, _z11, _x12, _y13, _z14, _x15, _y16, _z17, _x18, _y19, _x20, _y21;
clamped .split(x, x, _x0, 128)
.reorder(_x0, x, y) .reorder_storage(x, y)
.unroll(_x0, 32) .vectorize(_x0, 2)
.compute_root()
;
histogram .split(x, x, _x2, 16) .split(y, y, _y3, 2) .split(c, c, _c4, 8)
.reorder(_x2, _c4, x, _y3, c, y) .reorder_storage(c, y, x)
.vectorize(_x2, 4)
.compute_at(blurx, _z11)
;
grid .split(x, x, _x5, 16) .split(y, y, _y6, 512) .split(z, z, _z7, 8) .split(c, c, _c8, 2)
.reorder(_y6, _c8, y, _z7, c, z, _x5, x) .reorder_storage(c, x, y, z)
.unroll(_y6, 128) .vectorize(_y6, 16)
.compute_at(blurx, _y10)
;
blurx .split(x, x, _x9, 2) .split(y, y, _y10, 512) .split(z, z, _z11, 4)
.reorder(_y10, _z11, z, y, _x9, x) .reorder_storage(x, y, z)
.unroll(_y10, 64) .vectorize(_y10, 8) .parallel(x)
.compute_root()
;
blury .split(x, x, _x12, 32) .split(y, y, _y13, 2) .split(z, z, _z14, 4)
.reorder(_x12, x, _z14, _y13, z, y) .reorder_storage(z, y, x)
.unroll(_x12, 8) .vectorize(_x12, 4) .parallel(y)
.compute_root()
;
blurz .split(x, x, _x15, 16) .split(y, y, _y16, 128) .split(z, z, _z17, 8)
.reorder(_y16, _z17, y, z, _x15, x) .reorder_storage(x, z, y)
.unroll(_y16, 32) .vectorize(_y16, 4)
.compute_at(bilateral_grid, _x20)
;
interpolated .split(x, x, _x18, 16) .split(y, y, _y19, 128)
.reorder(_y19, _x18, y, x) .reorder_storage(x, y)
.unroll(_y19, 16) .vectorize(_y19, 2)
.compute_at(bilateral_grid, _y21)
;
bilateral_grid .split(x, x, _x20, 16) .split(y, y, _y21, 512)
.reorder(_y21, _x20, y, x) .reorder_storage(y, x)
.unroll(_y21, 32) .vectorize(_y21, 2) .parallel(x)
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



