#!/bin/bash

# For Gaussian 09:
# Set the root directory for your Gaussian installation
export g09root=$HOME/software/gaussian

# Set the scratch directory for temporary files
# It is highly recommended to use a fast, local disk for this.
export GAUSS_SCRDIR=$HOME/scratches/g09

# Create the scratch directory if it doesn't exist
mkdir -p "$GAUSS_SCRDIR"

# Source the official Gaussian profile to set up the environment
# This script will modify your PATH and define other necessary variables.
if [ -f "$g09root/g09/bsd/g09.profile" ]; then
  . "$g09root/g09/bsd/g09.profile"
else
  echo "Error: Gaussian profile script not found at $g09root/g09/bsd/g09.profile"
  return 1
fi

echo "Gaussian 09 environment is now active."

###

# Run QM using the full path
g09 < opt_conf_1.gjf > opt_conf_1.log ; \
g09 < opt_conf_2.gjf > opt_conf_2.log ; \
g09 < opt_conf_303.gjf > opt_conf_303.log ; \
g09 < opt_conf_304.gjf > opt_conf_304.log
