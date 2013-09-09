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

Halide::Var _x0, _y1, _x2, _y3, _c4, _x5, _y6, _z7, _c8, _x9, _y10, _z11, _x12, _y16, _z17, _x18, _y19, _x20, _y21;
clamped .split(x, x, _x0, 8) .split(y, y, _y1, 16)
.reorder(_y1, _x0, y, x) .reorder_storage(y, x)
.vectorize(_y1, 8)
.compute_at(histogram, _x2)
;
histogram .split(x, x, _x2, 2) .split(y, y, _y3, 4) .split(c, c, _c4, 8)
.reorder(_x2, _y3, x, _c4, y, c) .reorder_storage(c, y, x)

.compute_root()
;
grid .split(x, x, _x5, 4) .split(y, y, _y6, 4) .split(z, z, _z7, 16) .split(c, c, _c8, 128)
.reorder(_c8, _y6, c, y, _x5, x, _z7, z) .reorder_storage(y, c, x, z)
.vectorize(_c8, 16)
.compute_at(blury, y)
;
blurx .split(x, x, _x9, 16) .split(y, y, _y10, 2) .split(z, z, _z11, 2)
.reorder(_x9, x, _z11, _y10, z, y) .reorder_storage(x, z, y)
.vectorize(_x9, 8)
.compute_at(blury, _x12)
;
blury .split(x, x, _x12, 128)
.reorder(_x12, y, z, x) .reorder_storage(z, y, x)
.vectorize(_x12, 8)
.compute_root()
;
blurz .split(y, y, _y16, 2) .split(z, z, _z17, 128)
.reorder(_z17, z, _y16, x, y) .reorder_storage(y, z, x)
.vectorize(_z17, 16)
.compute_at(bilateral_grid, _y21)
;
interpolated .split(x, x, _x18, 8) .split(y, y, _y19, 2)
.reorder(_x18, x, _y19, y) .reorder_storage(y, x)
.vectorize(_x18, 4)
.compute_at(bilateral_grid, x)
;
bilateral_grid .split(x, x, _x20, 128) .split(y, y, _y21, 16)
.reorder(_x20, x, _y21, y) .reorder_storage(y, x)
.vectorize(_x20, 16)
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



