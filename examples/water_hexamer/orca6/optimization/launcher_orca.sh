#!/bin/bash

# Define the paths to your ORCA 6.1.1 installation
export ORCA_BASE=$HOME/software
export ORCA611_ROOT=$ORCA_BASE/orca_6_1_1
export OPENMPI418_ROOT=$ORCA_BASE/openmpi-4.1.8

# Save system paths to prevent infinite nesting if sourced multiple times
_SYSTEM_PATH="$PATH"
_SYSTEM_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

# Prepend ORCA and OpenMPI to the system paths
export PATH="$ORCA611_ROOT:$OPENMPI418_ROOT/bin:$_SYSTEM_PATH"
export LD_LIBRARY_PATH="$ORCA611_ROOT:$OPENMPI418_ROOT/lib:$_SYSTEM_LD_LIBRARY_PATH"

echo "ORCA 6.1.1 environment is now active."
mpirun --version

###

# Run QM using the full path
$ORCA611_ROOT/orca opt_conf_1.inp > opt_conf_1.out ; \
$ORCA611_ROOT/orca opt_conf_2.inp > opt_conf_2.out ; \
$ORCA611_ROOT/orca opt_conf_3.inp > opt_conf_3.out ; \
$ORCA611_ROOT/orca opt_conf_415.inp > opt_conf_415.out ; \
$ORCA611_ROOT/orca opt_conf_416.inp > opt_conf_416.out
