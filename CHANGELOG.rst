=========
Changelog
=========

Version 0.7.6
=============
Added:
- Adding center virtual chord evaluation. (#59)
- Update building and publishing workflow. (#58)

Version 0.7.5
=============
Modified:

- Added possibility to specify absolute Y engine position. (#55)
- Added absolute positioning of kink and absolute trailing edge kink sweep angle. (#56)

Fixed:

- Update of CeRAS Notebook with description of the taxi-out and takeoff phases. (#54)

Version 0.7.4
=============
Modified:

- Added tuning factor for ratio between MLW and MZFW. (#51)

Version 0.7.3
=============
Modified:

- Added scaling factors for tuning HTP and VTP areas. (#49)

Version 0.7.2
=============
Modified:

- Python 3.12 is now supported. (#48)
- Python 3.8, that reached end of life, is no longer supported. (#48)

Version 0.7.1
=============
Fixed:

- Incorrect comments in a datafile for CeRAS notebook would cause crash with FAST-OAD-core 1.8+. (#47)

Version 0.7.0
=============
Fixed:

- Fixed tail geometry (**Warning**: slight but noticeable changes in tail CG, hence in overall data). (#45)

Version 0.6.1
=============
Added:

- Compatibility with Python 3.11. (#43)

Version 0.6.0
=============
Changed:

- More control on calculation of load factors. (#38)
- Update on geometry of tail planes (#36)

Fixed:

- Restored the default solver behaviors of v0.4 for models with inner loop. (#39)

Version 0.5.0
=============
Changed:

- Allowing deactivation of inner solvers. (#33)
- Having ISA temperature offset as input of engine deck. (#31)


Version 0.4.0
=============
Changed:

- FAST-OAD-CS25 is now officially compatible with Python 3.10. Support of Python 3.7 has been abandoned. (#26)
- Trailing edge of wing inner part can now have a non-zero sweep angle.(#27)
- Now the criteria for computation of wing can be controlled through options. (#29)
- Computation of wing geometry now uses sub-models and has now an option for (de)activating the computation of wing thickness. (#29)

Fixed:

- Fixed component for computing global positions of wing chords, which is now a sub-model of geometry module (#28)


Version 0.3.0
=============
Changed:

- OpenMDAO 3.18+ is now required. (#18)
- Submodels have been introduced in module "fastoad.loop.wing_area". (#7)
- Load factors are now explicitly in output data. (#23)

Fixed:

- Unit of variable "data:weight:aircraft:additional_fuel_capacity" has been corrected to "kg". (#18)
- Unit of "*:CL_alpha" variables is now consistently "1/rad". (#21)
- Unit for "data:load_case:lc2:Vc_EAS" has been corrected to "m/s". (#23)

Version 0.2.0
=============
Added:

- Now polar computation in aerodynamics module computes angle of attack as a linear function of CL. (#16)

Version 0.1.4
=============
Fixed:

- Bundled notebooks have been modified to adapt to FAST-OAD 1.4.1, which is now the minimum required version for FAST-OAD-core. (#14)

Version 0.1.3
=============
Fixed:

- in bundled notebooks:

  - Generation of configuration file would fail if several FAST-OAD plugins were installed.
  - Link to CeRAS website has been fixed

Version 0.1.2
=============
Changed:

- Now allowing wing geometry with no kink (#3)

Fixed:

- Fixed deprecation warnings (#4)
- Now allowing versions greater than 0.1 for StdAtm

Version 0.1.1
=============
- Fixed dependency to FAST-OAD

Version 0.1.0
=============
- FAST-OAD CS-25 related models are now in this separate package
