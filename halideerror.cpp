#include <Halide.h>
#include <stdio.h>
#include <sys/time.h>

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

    Halide::Var _x0, _y1, _x2, _y3, _x4, _y5;
    input .reorder_storage(y, x)
    .split(x, x, _x0, 16)
    .split(y, y, _y1, 4)
    .parallel(y)
    .reorder(y, _y1, x, _x0)
    .compute_at(blur_x, _y3);
    blur_x .reorder_storage(y, x)
    .split(x, x, _x2, 2)
    .split(y, y, _y3, 2)
    .reorder(x, _x2, y, _y3)
    .compute_at(blur_y, _y5);
    blur_y .reorder_storage(x, y)
    .split(x, x, _x4, 16)
    .split(y, y, _y5, 16)
    .reorder(x, y, _x4, _y5)
    .compute_root();

    blur_y.infer_input_bounds(1024,1024);
    blur_y.realize(1024,1024);

    return 0;
}
