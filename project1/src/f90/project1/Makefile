# -*- Makefile -*- # hw1

#FC        = ifort     # F90 compiler on ITA owls
FC        = gfortran   # F90 compiler on krisjand's Dell
OPTIM     = -O3       # Optimization flags; set to -g -C for useful debug information

OUTPUT  = project1


# List of source files to be compiled
OBJS    = linalg.o RNGesus.o problem1.o project1.o common_flags.o

# Linking stage
hw1: $(OBJS)
	$(FC) $(OPTIM) -o $(OUTPUT) $(OBJS)

# Dependencies
hw1.o           : problem1.o RNGesus.o linalg.o common_flags.o
problem1.o      : RNGesus.o linalg.o common_flags.o
linalg.o        : common_flags.o

# Compilation of source files
%.o : %.f90
	$(FC) $(OPTIM) -c $<

# Clean-up command (write "make clean")
.PHONY: clean
clean:
	rm *.mod *.o *~ project1

