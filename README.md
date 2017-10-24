# E-BeamPatterns
A collection of scripts that create e-beam patterns for GaAs nanomembrane growth in MBE.

## Introduction
In our group at EPFL in the Laboratory of Semiconductor Materials ([LMSC](https://lmsc.epfl.ch/)) we use molecular beam epitaxy ([MBE](https://en.wikipedia.org/wiki/Molecular_beam_epitaxy)) to fabricate III-V material nanostructures, typically out of GaAs. Traditionally, people in our group have been growing on full or half 2-inch GaAs wafers which each cost >150 CHF. To reduce our material cost per growth I have instead begun to pattern each 2-inch wafer into multiple (up to 24!) chips, each of which can be used for a separate growth. 

The design of these chips is not easy, especially if we use (111) wafers which have a three-fold symmetry and therefore cleave along lines that are at 60 degree angles. These scripts use a modified version of the [gdsCAD](https://github.com/hohlraum/gdsCAD) library to generate these patterns. The outputs are GDS files which are then fractured and written using electron beam (e-beam) lithography onto our wafers.

Here's an example:

![Basel Membranes Wafer](https://github.com/Martin09/E-BeamPatterns/blob/master/Images/BaselMembranes_4.4.1.png "Basel Membranes Wafer v4.41")

## Goal
I put these scripts here to encourage anyone who wants to try their hand at designing wafers using Python. Hopefully these scripts can serve as a starting point for new users that would like to do the same.
