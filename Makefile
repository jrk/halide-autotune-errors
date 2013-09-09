CXX = g++ -O3

HALIDE_DIR=../FImage
HALIDE_BIN=$(HALIDE_DIR)/bin/$(BUILD_PREFIX)
HALIDE_INC=$(HALIDE_DIR)/include

LDFLAGS = -rdynamic $(HALIDE_BIN)/libHalide.a -lpthread -ldl

binaries := $(patsubst %.cpp,%.exe,$(wildcard *.cpp))
traces := $(patsubst %.exe,%.trace,$(binaries))

all: $(binaries)

%.exe: %.cpp $(HALIDE_BIN) $(HALIDE_INC)
	$(CXX) $< -DAUTOTUNE_N=1024,1024 -DAUTOTUNE_TRIALS=1 $(LDFLAGS) -I$(HALIDE_INC) -o $@

%.run: %.exe
	./$<

%.dbg: %.exe
	gdb ./$<

%.trace: %.exe
	HL_TRACE=1 ./$< > $@
	cat $@

clean:
	rm -f $(binaries) $(traces)
