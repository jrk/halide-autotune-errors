#include "Halide.h"

using namespace Halide;

#include <iostream>
#include <limits>

#include "timing_prefix.h"

#include <sys/time.h>

using std::vector;

int main(int argc, char **argv) {
    ImageParam input(Float(32), 3);

    const unsigned int levels = 10;

#if 0
    std::vector<Func> downsampled;
    std::vector<Func> downx;
    std::vector<Func> interpolated;
    std::vector<Func> upsampled;
    std::vector<Func> upsampledx;
    Var x("x"), y("y"), c("c");

    for(size_t i = 0; i < levels; i++) {
        downsampled.push_back(Func("downsampled"));
        downx.push_back(Func("downx"));
        interpolated.push_back(Func("interpolated"));
        upsampled.push_back(Func("upsampled"));
        upsampledx.push_back(Func("upsampledx"));
    }
#else
    Func downsampled[levels];
    Func downx[levels];
    Func interpolated[levels];
    Func upsampled[levels];
    Func upsampledx[levels];
    Var x("x"), y("y"), c("c");
#endif

    Func clamped("clamped");
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
        std::map<std::string, Halide::Internal::Function> funcs = Halide::Internal::find_transitive_calls((final).function());
Halide::Var _x0, _y1, _y4, _c5, _x6, _y7, _c8, _y10, _c11, _y13, _c14, _x15, _y16, _c17, _x18, _y19, _y22, _c23, _y25, _c26, _y28, _c29, _x30, _x33, _y34, _c35, _y37, _c38, _x39, _y40, _c41, _c44, _x45, _c47, _x48, _y49, _c50, _y52, _x54, _c56, _x57, _y58, _c59, _x60, _y61, _c62, _x63, _y64, _c65, _x66, _y67, _c68, _x69, _y70, _c71, _y73, _c74, _c80, _x81, _y82, _x84, _y85, _c86, _x87, _y88, _c89, _x90, _y91, _c92, _x96, _y97, _c98, _x99, _y100, _c101, _x102, _y103, _c104, _y106, _c107, _x108, _y109, _c110, _x111, _c113, _x114, _c116, _x117, _y118, _c119, _x120, _y121, _c122, _x123, _c125, _x126, _c128, _y130, _c131, _x132, _y133, _y136, _c137, _x138, _y139, _c140, _x141, _y142, _c143, _x144, _y145, _x147, _c149;
Halide::Func(funcs["f0"])
.split(x, x, _x0, 16)
.split(y, y, _y1, 8)
.reorder(_x0, _y1, y, x, c)
.reorder_storage(x, y, c)
.vectorize(_x0, 4)
.compute_root()
;
Halide::Func(funcs["f1"])
.split(y, y, _y4, 8)
.split(c, c, _c5, 2)
.reorder(_y4, _c5, x, y, c)
.reorder_storage(x, c, y)
.vectorize(_y4, 2)
.parallel(c)
.compute_root()
;
Halide::Func(funcs["f11"])
.split(x, x, _x6, 8)
.split(y, y, _y7, 4)
.split(c, c, _c8, 8)
.reorder(_y7, _c8, _x6, c, x, y)
.reorder_storage(c, x, y)
.compute_at(Halide::Func(funcs["f1"]), _c5)
;
Halide::Func(funcs["f12"])
.split(y, y, _y10, 4)
.split(c, c, _c11, 8)
.reorder(_c11, _y10, c, y, x)
.reorder_storage(c, x, y)
.compute_root()
;
Halide::Func(funcs["f13"])
.split(y, y, _y13, 4)
.split(c, c, _c14, 4)
.reorder(_y13, y, _c14, x, c)
.reorder_storage(c, x, y)
.parallel(c)
.compute_root()
;
Halide::Func(funcs["f14"])
.split(x, x, _x15, 4)
.split(y, y, _y16, 2)
.split(c, c, _c17, 8)
.reorder(_c17, _x15, _y16, y, x, c)
.reorder_storage(y, c, x)
.compute_at(Halide::Func(funcs["f4"]), _y97)
;
Halide::Func(funcs["f15"])
.split(x, x, _x18, 8)
.split(y, y, _y19, 4)
.reorder(_x18, _y19, x, c, y)
.reorder_storage(c, y, x)
.compute_root()
;
Halide::Func(funcs["f16"])
.split(y, y, _y22, 8)
.split(c, c, _c23, 4)
.reorder(_y22, _c23, x, c, y)
.reorder_storage(y, c, x)
.compute_root()
;
Halide::Func(funcs["f17"])
.split(y, y, _y25, 64)
.split(c, c, _c26, 4)
.reorder(_y25, _c26, y, x, c)
.reorder_storage(y, c, x)
.vectorize(_y25, 8)
.compute_at(Halide::Func(funcs["f7"]), _y136)
;
Halide::Func(funcs["f18"])
.split(y, y, _y28, 4)
.split(c, c, _c29, 4)
.reorder(_c29, _y28, x, c, y)
.reorder_storage(y, x, c)
.vectorize(_c29, 2)
.compute_at(Halide::Func(funcs["f8"]), _x138)
;
Halide::Func(funcs["f19"])
.split(x, x, _x30, 8)
.reorder(_x30, c, y, x)
.reorder_storage(c, x, y)
.vectorize(_x30, 2)
.compute_at(Halide::Func(funcs["f9"]), _x141)
;
Halide::Func(funcs["f2"])
.split(x, x, _x33, 8)
.split(y, y, _y34, 4)
.split(c, c, _c35, 2)
.reorder(_x33, _y34, x, _c35, c, y)
.reorder_storage(y, c, x)
.vectorize(_x33, 2)
.compute_root()
;
Halide::Func(funcs["f20"])
.split(y, y, _y37, 16)
.split(c, c, _c38, 4)
.reorder(_y37, y, _c38, c, x)
.reorder_storage(y, x, c)
.vectorize(_y37, 4)
.compute_at(Halide::Func(funcs["normalize"]), _y145)
;
Halide::Func(funcs["f21"])
.split(x, x, _x39, 8)
.split(y, y, _y40, 8)
.split(c, c, _c41, 2)
.reorder(_y40, _x39, _c41, x, c, y)
.reorder_storage(c, x, y)
.vectorize(_y40, 4)
.compute_at(Halide::Func(funcs["final"]), x)
;
Halide::Func(funcs["f22"])
.split(c, c, _c44, 8)
.reorder(y, _c44, x, c)
.reorder_storage(c, x, y)
.compute_at(Halide::Func(funcs["f41"]), _x102)
;
Halide::Func(funcs["f23"])
.split(x, x, _x45, 4)
.split(c, c, _c47, 4)
.reorder(_x45, _c47, x, c, y)
.reorder_storage(c, y, x)
.compute_at(Halide::Func(funcs["f42"]), c)
;
Halide::Func(funcs["f24"])
.split(x, x, _x48, 8)
.split(y, y, _y49, 2)
.split(c, c, _c50, 8)
.reorder(_x48, _c50, x, _y49, c, y)
.reorder_storage(c, x, y)
.vectorize(_x48, 2)
.parallel(y)
.compute_root()
;
Halide::Func(funcs["f25"])
.split(y, y, _y52, 4)
.reorder(_y52, c, y, x)
.reorder_storage(c, x, y)
.compute_at(Halide::Func(funcs["f44"]), _x111)
;
Halide::Func(funcs["f26"])
.split(x, x, _x54, 64)
.split(c, c, _c56, 4)
.reorder(_x54, _c56, c, x, y)
.reorder_storage(c, x, y)
.vectorize(_x54, 8)
.compute_at(Halide::Func(funcs["f35"]), c)
;
Halide::Func(funcs["f27"])
.split(x, x, _x57, 8)
.split(y, y, _y58, 2)
.split(c, c, _c59, 16)
.reorder(_c59, c, _x57, _y58, y, x)
.reorder_storage(y, x, c)
.vectorize(_c59, 2)
.compute_at(Halide::Func(funcs["f46"]), _x117)
;
Halide::Func(funcs["f28"])
.split(x, x, _x60, 8)
.split(y, y, _y61, 2)
.split(c, c, _c62, 4)
.reorder(_x60, _y61, _c62, x, c, y)
.reorder_storage(x, y, c)
.compute_at(Halide::Func(funcs["f46"]), y)
;
Halide::Func(funcs["f29"])
.split(x, x, _x63, 2)
.split(y, y, _y64, 2)
.split(c, c, _c65, 4)
.reorder(_c65, _y64, c, _x63, y, x)
.reorder_storage(y, c, x)
.vectorize(_c65, 2)
.compute_at(Halide::Func(funcs["f48"]), y)
;
Halide::Func(funcs["f3"])
.split(x, x, _x66, 32)
.split(y, y, _y67, 4)
.split(c, c, _c68, 4)
.reorder(_x66, _y67, _c68, c, y, x)
.reorder_storage(y, c, x)
.vectorize(_x66, 8)
.compute_root()
;
Halide::Func(funcs["f30"])
.split(x, x, _x69, 2)
.split(y, y, _y70, 16)
.split(c, c, _c71, 8)
.reorder(_y70, _x69, _c71, y, x, c)
.reorder_storage(c, y, x)
.vectorize(_y70, 8)
.compute_at(Halide::Func(funcs["final"]), _x147)
;
Halide::Func(funcs["f31"])
.split(y, y, _y73, 4)
.split(c, c, _c74, 2)
.reorder(x, _c74, _y73, c, y)
.reorder_storage(c, x, y)
.vectorize(x, 8)
.parallel(y)
.compute_root()
;
Halide::Func(funcs["f32"])
.reorder(x, y, c)
.reorder_storage(x, c, y)
.vectorize(x, 4)
.compute_at(Halide::Func(funcs["f41"]), _c104)
;
Halide::Func(funcs["f33"])
.split(c, c, _c80, 8)
.reorder(_c80, c, y, x)
.reorder_storage(c, y, x)
.vectorize(_c80, 2)
.compute_at(Halide::Func(funcs["f41"]), y)
;
Halide::Func(funcs["f34"])
.split(x, x, _x81, 32)
.split(y, y, _y82, 2)
.reorder(_x81, _y82, x, c, y)
.reorder_storage(y, c, x)
.vectorize(_x81, 4)
.compute_at(Halide::Func(funcs["f24"]), _c50)
;
Halide::Func(funcs["f35"])
.split(x, x, _x84, 8)
.split(y, y, _y85, 4)
.split(c, c, _c86, 2)
.reorder(_c86, c, _x84, _y85, x, y)
.reorder_storage(x, y, c)
.compute_at(Halide::Func(funcs["f24"]), c)
;
Halide::Func(funcs["f36"])
.split(x, x, _x87, 4)
.split(y, y, _y88, 4)
.split(c, c, _c89, 8)
.reorder(_c89, _y88, _x87, x, y, c)
.reorder_storage(x, y, c)
.vectorize(_c89, 4)
.compute_at(Halide::Func(funcs["f24"]), y)
;
Halide::Func(funcs["f37"])
.split(x, x, _x90, 32)
.split(y, y, _y91, 8)
.split(c, c, _c92, 4)
.reorder(_x90, _c92, _y91, y, x, c)
.reorder_storage(y, x, c)
.vectorize(_x90, 8)
.compute_at(Halide::Func(funcs["f27"]), c)
;
Halide::Func(funcs["f38"])
.reorder(x, y, c)
.reorder_storage(x, c, y)
.vectorize(x, 8)
.compute_at(Halide::Func(funcs["f46"]), x)
;
Halide::Func(funcs["f4"])
.split(x, x, _x96, 4)
.split(y, y, _y97, 2)
.split(c, c, _c98, 16)
.reorder(_c98, _x96, _y97, c, y, x)
.reorder_storage(x, c, y)
.vectorize(_c98, 2)
.compute_root()
;
Halide::Func(funcs["f40"])
.split(x, x, _x99, 4)
.split(y, y, _y100, 16)
.split(c, c, _c101, 4)
.reorder(_y100, _c101, c, _x99, y, x)
.reorder_storage(x, c, y)
.vectorize(_y100, 2)
.compute_at(Halide::Func(funcs["final"]), y)
;
Halide::Func(funcs["f41"])
.split(x, x, _x102, 8)
.split(y, y, _y103, 4)
.split(c, c, _c104, 4)
.reorder(_x102, _c104, _y103, x, y, c)
.reorder_storage(x, c, y)
.compute_root()
;
Halide::Func(funcs["f42"])
.split(y, y, _y106, 2)
.split(c, c, _c107, 8)
.reorder(_c107, c, _y106, y, x)
.reorder_storage(c, y, x)
.vectorize(_c107, 2)
.compute_at(Halide::Func(funcs["f41"]), _y103)
;
Halide::Func(funcs["f43"])
.split(x, x, _x108, 2)
.split(y, y, _y109, 4)
.split(c, c, _c110, 4)
.reorder(_y109, _x108, _c110, y, x, c)
.reorder_storage(c, y, x)
.vectorize(_y109, 2)
.compute_at(Halide::Func(funcs["f41"]), c)
;
Halide::Func(funcs["f44"])
.split(x, x, _x111, 16)
.split(c, c, _c113, 4)
.reorder(_x111, _c113, x, c, y)
.reorder_storage(y, c, x)
.vectorize(_x111, 2)
.compute_at(Halide::Func(funcs["f24"]), _y49)
;
Halide::Func(funcs["f45"])
.split(x, x, _x114, 8)
.split(c, c, _c116, 16)
.reorder(_c116, _x114, c, x, y)
.reorder_storage(y, x, c)
.vectorize(_c116, 4)
.compute_at(Halide::Func(funcs["f35"]), _c86)
;
Halide::Func(funcs["f46"])
.split(x, x, _x117, 2)
.split(y, y, _y118, 4)
.split(c, c, _c119, 32)
.reorder(_c119, _x117, _y118, c, y, x)
.reorder_storage(x, c, y)
.vectorize(_c119, 4)
.parallel(x)
.compute_root()
;
Halide::Func(funcs["f47"])
.split(x, x, _x120, 2)
.split(y, y, _y121, 2)
.split(c, c, _c122, 4)
.reorder(_x120, _y121, _c122, y, x, c)
.reorder_storage(x, y, c)
.compute_at(Halide::Func(funcs["f37"]), _x90)
;
Halide::Func(funcs["f48"])
.split(x, x, _x123, 4)
.split(c, c, _c125, 4)
.reorder(_c125, _x123, y, c, x)
.reorder_storage(x, y, c)
.compute_root()
;
Halide::Func(funcs["f5"])
.split(x, x, _x126, 32)
.split(c, c, _c128, 2)
.reorder(_x126, _c128, x, c, y)
.reorder_storage(x, c, y)
.vectorize(_x126, 4)
.parallel(y)
.compute_root()
;
Halide::Func(funcs["f50"])
.split(y, y, _y130, 2)
.split(c, c, _c131, 8)
.reorder(x, _y130, _c131, y, c)
.reorder_storage(x, c, y)
.vectorize(x, 2)
.compute_root()
;
Halide::Func(funcs["f6"])
.split(x, x, _x132, 8)
.split(y, y, _y133, 32)
.reorder(_y133, c, _x132, y, x)
.reorder_storage(c, x, y)
.vectorize(_y133, 4)
.compute_root()
;
Halide::Func(funcs["f7"])
.split(y, y, _y136, 16)
.split(c, c, _c137, 4)
.reorder(_y136, x, y, _c137, c)
.reorder_storage(c, x, y)
.vectorize(_y136, 2)
.compute_root()
;
Halide::Func(funcs["f8"])
.split(x, x, _x138, 4)
.split(y, y, _y139, 4)
.split(c, c, _c140, 8)
.reorder(_x138, x, _y139, _c140, y, c)
.reorder_storage(y, x, c)
.parallel(c)
.compute_root()
;
Halide::Func(funcs["f9"])
.split(x, x, _x141, 2)
.split(y, y, _y142, 4)
.split(c, c, _c143, 2)
.reorder(_x141, _y142, _c143, c, x, y)
.reorder_storage(c, y, x)
.compute_at(Halide::Func(funcs["f29"]), _c65)
;
Halide::Func(funcs["normalize"])
.split(x, x, _x144, 8)
.split(y, y, _y145, 8)
.reorder(_y145, _x144, y, x, c)
.reorder_storage(y, x, c)
.vectorize(_y145, 4)
.compute_at(Halide::Func(funcs["final"]), c)
;
Halide::Func(funcs["final"])
.split(x, x, _x147, 4)
.split(c, c, _c149, 16)
.reorder(_c149, c, _x147, y, x)
.reorder_storage(c, x, y)
.vectorize(_c149, 8)
.compute_root()
;

        _autotune_timing_stub(final);
    }
}
