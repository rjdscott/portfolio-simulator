.PHONY: build clean

FC      = gfortran
FFLAGS  = -O3 -march=native -ffast-math -funroll-loops -fopenmp -shared -fPIC
OUTDIR  = build
TARGET  = $(OUTDIR)/libportfolio_fortran.so
SRC     = src/portfolio_compute.f90

$(TARGET): $(SRC)
	mkdir -p $(OUTDIR)
	$(FC) $(FFLAGS) -o $(TARGET) $(SRC)

build: $(TARGET)

clean:
	rm -rf $(OUTDIR)
