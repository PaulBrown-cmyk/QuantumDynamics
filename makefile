#=============================== Makefile ====================================
MPIFC ?= mpifort

FFTW_PREFIX ?= $(shell brew --prefix fftw 2>/dev/null || echo /usr)
HDF5_PREFIX ?= $(shell brew --prefix hdf5 2>/dev/null || echo /usr)
HDF5_MPI_PREFIX ?= $(shell brew --prefix hdf5-mpi 2>/dev/null || echo /usr)

MODDIR := build/mod
OBJDIR := build/obj

# Preprocess + OpenMP + module output/search paths
FFLAGS  := -cpp -O2 -g -fopenmp -J$(MODDIR) -I$(MODDIR)
FFLAGS  += -I$(FFTW_PREFIX)/include
WARNFLAGS ?=
FFLAGS  += $(WARNFLAGS)

# Toggle HDF5 by setting USE_HDF5=1 (default on)
USE_HDF5 ?= 1
USE_PARALLEL_HDF5 ?= 0
ifeq ($(USE_HDF5),1)
  FFLAGS  += -DUSE_HDF5 -I$(HDF5_PREFIX)/include
  LDLIBS  += -L$(HDF5_PREFIX)/lib -lhdf5_fortran -lhdf5
  # If you use the HL API:
  # LDLIBS += -lhdf5_hl_fortran -lhdf5_hl
endif
ifeq ($(USE_PARALLEL_HDF5),1)
  ifneq ($(USE_HDF5),1)
    $(error USE_PARALLEL_HDF5=1 requires USE_HDF5=1)
  endif
  FFLAGS += -DUSE_PARALLEL_HDF5
endif
# FFTW link
LDLIBS += -L$(FFTW_PREFIX)/lib -lfftw3_threads -lfftw3 -lm -lpthread

SRC = kinds.f08 constants.f08 mpi_env.f08 timers.f08 params.f08 rng.f08 grid.f08 potentials.f08 \
      fftwrap.f08 langevin.f08 propagator.f08 checkpoint.f08 io_hdf5.f08 io_parallel_hdf5.f08 main.f08

OBJ = $(patsubst %.f08,$(OBJDIR)/%.o,$(SRC))

all: qle_1d

# Ensure directories exist
$(MODDIR) $(OBJDIR):
	mkdir -p $@

# Compile each source to an object, producing .mod into build/mod
$(OBJDIR)/%.o: %.f08 | $(MODDIR) $(OBJDIR)
	$(MPIFC) $(FFLAGS) -x f95-cpp-input -c $< -o $@

# Link
qle_1d: $(OBJ)
	$(MPIFC) $(FFLAGS) -o $@ $(OBJ) $(LDLIBS)

$(OBJDIR)/constants.o: $(OBJDIR)/kinds.o
$(OBJDIR)/mpi_env.o: $(OBJDIR)/kinds.o
$(OBJDIR)/timers.o: $(OBJDIR)/kinds.o
$(OBJDIR)/params.o: $(OBJDIR)/kinds.o $(OBJDIR)/constants.o
$(OBJDIR)/rng.o: $(OBJDIR)/kinds.o
$(OBJDIR)/grid.o: $(OBJDIR)/kinds.o
$(OBJDIR)/potentials.o: $(OBJDIR)/kinds.o $(OBJDIR)/params.o $(OBJDIR)/rng.o
$(OBJDIR)/fftwrap.o: $(OBJDIR)/kinds.o
$(OBJDIR)/langevin.o: $(OBJDIR)/kinds.o $(OBJDIR)/params.o $(OBJDIR)/rng.o
$(OBJDIR)/propagator.o: $(OBJDIR)/kinds.o $(OBJDIR)/params.o $(OBJDIR)/langevin.o \
                         $(OBJDIR)/grid.o $(OBJDIR)/potentials.o $(OBJDIR)/fftwrap.o
$(OBJDIR)/checkpoint.o: $(OBJDIR)/kinds.o $(OBJDIR)/params.o $(OBJDIR)/propagator.o \
                         $(OBJDIR)/langevin.o $(OBJDIR)/potentials.o $(OBJDIR)/rng.o
$(OBJDIR)/io_hdf5.o: $(OBJDIR)/kinds.o $(OBJDIR)/params.o $(OBJDIR)/constants.o \
                      $(OBJDIR)/grid.o $(OBJDIR)/potentials.o
$(OBJDIR)/io_parallel_hdf5.o: $(OBJDIR)/kinds.o $(OBJDIR)/params.o $(OBJDIR)/constants.o \
                               $(OBJDIR)/grid.o $(OBJDIR)/potentials.o $(OBJDIR)/mpi_env.o
$(OBJDIR)/main.o: $(filter-out $(OBJDIR)/main.o,$(OBJ))

clean:
	rm -rf build qle_1d *.h5 *.dat

# Module sources are ordered in SRC; do not race module generation.
.NOTPARALLEL:
.PHONY: all clean test test-parallel
test: qle_1d
	FC=$(MPIFC) FFTW_PREFIX=$(FFTW_PREFIX) QLE_EXE=$(CURDIR)/qle_1d \
	  HDF5_ENABLED=$(USE_HDF5) ./tests/run.sh

test-parallel:
	$(MAKE) -B USE_HDF5=1 USE_PARALLEL_HDF5=1 HDF5_PREFIX=$(HDF5_MPI_PREFIX)
	QLE_EXE=$(CURDIR)/qle_1d MPIEXEC=$$(command -v mpiexec) ./tests/run_parallel_hdf5.sh
