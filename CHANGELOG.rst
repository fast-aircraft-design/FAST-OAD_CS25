=========
Changelog
=========


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
