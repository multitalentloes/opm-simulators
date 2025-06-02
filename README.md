<h1 align="center">
  <img src="doc/logo/opmlogo.png" alt="Logo" width="32" height="32" style="vertical-align: middle;"> 
  Open Porous Media Simulators 
  <img src="doc/logo/opmlogo.png" alt="Logo" width="32" height="32" style="vertical-align: middle;">
</h1>

<p align="center">
  <a href="https://opm-project.org/"><img src="https://img.shields.io/badge/Home_Page-opm--project.org-8888FF" alt="Home Page"></a>
  <a href="https://"><img src="https://img.shields.io/badge/License-GPL_V.3%2B-2ea44f" alt="License"></a>
  <a href="https://opm-project.org/?page_id=955"><img src="https://img.shields.io/badge/docs-manual-999999" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://doi.org/10.1016/j.camwa.2020.05.014"><img src="https://img.shields.io/badge/Cite-10.1016%2Fj.camwa.2020.05.014-FF9900" alt="Cite"></a>
  <a href="https://doi.org/10.5281/zenodo.12637570"><img src="https://img.shields.io/badge/Zenodo_src_backup-10.5281%2Fzenodo.12637570-ff2222" alt="Zenodo"></a>
</p>

> 💡 The `opm-Simulators` repository containes the OPM Flow reservoir simulator which uses automatic differentation and standard input-formats to simulate reservoirs of industrial complexity on real assets by commercial actors. It is cooperatively developed by multiple industry-partners and academic institutions under the GPL 3 license.
> 
> Flow is a fully implicit black-oil simulators which also supports CO2 storage, H2 storage, thermal simulations, solvents, as well as polymers. It works with the Eclipse formats, making it easy to integrate into existing workflows. The automatic differentation is implemented in [opm-common](https://github.com/OPM/opm-common). The manual can be found here on the OPM [home page](https://opm-project.org/?page_id=955).

🌍 Supported Platforms
---------
- ✅ RHEL 8+
- ✅ Ubuntu 22.04 LTS and 24.04 LTS
- ✅ Debian
- ✅ Mac OS X
- 🚫 Windows

⬇️ [Download binaries here](https://opm-project.org/?page_id=36) ⬇️


🛠️ Building from Source
------------

`opm-simulators` Depends on:
- [opm-common](https://github.com/OPM/opm-common)
- [opm-grid](https://github.com/OPM/opm-grid)
- [Dune, and other depencies kept track of on the home page](https://opm-project.org/?page_id=239)

[Follow the build instructions on the home page](http://opm-project.org/?page_id=36).


📖 In-Code Documentation
---------------------
In addition to providing the [manual](https://opm-project.org/?page_id=955) we also document the source code with Doxygen, using the `make doc` command.


⚠️ Reporting Issues
----------------

Issues can either be reported here on [github repository](https://github.com/OPM/opm-simulators/issues), or using the [OPM mailing list](https://opm-project.org/?page_id=358)

To help diagnose build errors, please provide a link to a build log together
with the issue description.

You can capture such a log from the build using the `script' utility, e.g.:

    LOGFILE=$(date +%Y%m%d-%H%M-)build.log ;
    cmake -E cmake_echo_color --cyan --bold "Log file: $LOGFILE" ;
    script -q $LOGFILE -c 'cmake ../opm-core -DCMAKE_BUILD_TYPE=Debug' &&
    script -q $LOGFILE -a -c 'ionice nice make -j 4 -l 3' ||
    cat CMakeCache.txt CMakeFiles/CMake*.log >> $LOGFILE

The resulting file can be uploaded to for instance gist.github.com.

📑 **Citing**
---------------------
To cite OPM Flow we primarily use the following publication introducing the software in the `Computers & Mathematics with Applications` Journal:
```bibtex
@article{OPMFLOW,
 title = {The {Open} {Porous} {Media} {Flow} reservoir simulator},
 journal = {Computers \& Mathematics with Applications},
 volume = {81},
 pages = {159-185},
 year = {2021},
 note = {Development and Application of Open-source Software for Problems with Numerical PDEs},
 issn = {0898-1221},
 doi = {https://doi.org/10.1016/j.camwa.2020.05.014},
 url = {https://www.sciencedirect.com/science/article/pii/S0898122120302182},
 author = {Atgeirr Flø Rasmussen and Tor Harald Sandve and Kai Bao and Andreas Lauser and Joakim Hove and Bård Skaflestad and Robert Klöfkorn and Markus Blatt and Alf Birger Rustad and Ove Sævareid and Knut-Andreas Lie and Andreas Thune}
}
