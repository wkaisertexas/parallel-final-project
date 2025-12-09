
NVCC        = nvcc
NVCC_FLAGS  = -O3
DEBUG_FLAGS = -O0 -G -g -lineinfo
OBJ         = main.o matrix.o kernel0.o kernel1.o kernel2.o kernel3.o kernel4.o
EXE         = spmspm

# testing on the first matrix
ARGS = -f data/matrix0.txt -0

ncu-ui: default
	ncu-ui ./$(EXE) $(ARGS)

ncu: default
	ncu --set full -o profile ./$(EXE) $(ARGS)

default: $(EXE)

debug: NVCC_FLAGS=$(DEBUG_FLAGS)
debug: clean $(EXE)

gdb: debug
	gdb --ex run --args ./$(EXE) $(ARGS)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

$(EXE): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(OBJ) -o $(EXE)

test: default
	./$(EXE) $(ARGS)

test-k1: default
	./$(EXE) -f data/matrix0.txt -1

test-dbg: debug
	./$(EXE) $(ARGS)

clean:
	rm -rf $(OBJ) $(EXE) profile.ncu-rep

