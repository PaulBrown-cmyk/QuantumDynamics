#=============================== Makefile ====================================
MPIFC ?= mpifort

FFTW_PREFIX := $(shell brew --prefix fftw)
HDF5_PREFIX := $(shell brew --prefix hdf5)

MODDIR := build/mod
OBJDIR := build/obj

# Preprocess + OpenMP + module output/search paths
FFLAGS  := -cpp -O2 -g -fopenmp -J$(MODDIR) -I$(MODDIR)
FFLAGS  += -I$(FFTW_PREFIX)/include

# Toggle HDF5 by setting USE_HDF5=1 (default on)
USE_HDF5 ?= 1
ifeq ($(USE_HDF5),1)
  FFLAGS  += -DUSE_HDF5 -I$(HDF5_PREFIX)/include
  LDLIBS  += -L$(HDF5_PREFIX)/lib -lhdf5_fortran -lhdf5
  # If you use the HL API:
  # LDLIBS += -lhdf5_hl_fortran -lhdf5_hl
endif
# FFTW link
LDLIBS += -L$(FFTW_PREFIX)/lib -lfftw3_threads -lfftw3 -lm -lpthread

SRC = kinds.f08 mpi_env.f08 timers.f08 params.f08 rng.f08 grid.f08 potentials.f08 \
      fftwrap.f08 langevin.f08 propagator.f08 io_hdf5.f08 main.f08

OBJ = $(patsubst %.f08,$(OBJDIR)/%.o,$(SRC))

all: qle_1d

# Ensure directories exist
$(MODDIR) $(OBJDIR):
	mkdir -p $@

# Compile each source to an object, producing .mod into build/mod
$(OBJDIR)/%.o: %.f08 | $(MODDIR) $(OBJDIR)
	$(MPIFC) $(FFLAGS) -c $< -o $@

# Link
qle_1d: $(OBJ)
	$(MPIFC) $(FFLAGS) -o $@ $(OBJ) $(LDLIBS)

clean:
	rm -rf build qle_1d *.h5 *.dat

