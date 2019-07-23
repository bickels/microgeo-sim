# microGEO-SIM

μGEO-SIM - micro-geography of bacterial abundance and diversity across spatial scales. Code used to for spatially explicit simulations of bacterial cells moving and growing on hydrated soil surfaces.

---
This repository accommodates computer code for the manuscript *Soil bacterial diversity shaped by microscale processes across biomes* submitted to Nature Communications.

Authors: Samuel Bickel and Dani Or

Affiliation: Soil, Terrestrial and Environmental Physics (STEP); Institute of Biogeochemistry and Pollutant dynamics (IBP); Swiss Federal Institute of Technology (ETH), Zürich

Correspondence to: SB (samuel.bickel@usys.ethz.ch)

---
## System requirements
### Tested on: 
Linux Debian 8 64bit (and Microsoft Windows 7 64bit)
### Dependencies (tested version):
	- python (2.7.14)
		- numpy (1.14.3)
		- numba (0.40.0)
		- scipy (1.1.0)
		- pandas (0.22.0)
		- matplotlib (2.2.2)
## Installation 
Once python and dependencies are installed the script (`run.sh`) can be executed from the terminal after navigating to the same directory. In the linux terminal:
```
~\path\to\script\$ .\run.sh
~\path\to\script\$ ipython evaluate.py
```
Installation (Anaconda distribution recommended) should be possible within less than 30min.

## Demo
Executing the script (`run.sh`) generates several simulations of individual cells of different species growing under varying water contents (0.05-0.4 v/v) and constant carring capacity. Once the simulations finish (~2-3 days on a standard workstation) the analysis script (`evaluate.py`) can be executed to obtain summary statistics of the simulations that are written to excel files and printed on screen.
