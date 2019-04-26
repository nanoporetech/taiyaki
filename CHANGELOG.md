#  Taiyaki
Version numbers: major.minor.patch
* Major version bump indicates a substantial change, e.g. file formats or removal of functionality.
* Minor version bump indicates a change in functionality that may affect users.
* Patch version bump indicates bug-fixes or minor improvements not expected to affect users.

## v4.0.0
* Modified base training and basecalling
* Minor changes to input format to trainer, use `misc/upgrade_mapped_signal.py` to upgrade old data

## v3.1.0
* Basecaller script that uses GPU
* Training walk-through
* Tweaks to optimisation parameters


## v3.0.2
* Improved training parameters
* Use orthonormal initialisation of starting weights

## v3.0.1
* Bug fix: package version did not work in github's source releases

## v3.0.0
Initial release:
* Prepare data for training basecallers by remapping signal to reference sequence
* Train neural networks for flip-flop basecalling and squiggle prediction
* Export basecaller models for use in Guppy
