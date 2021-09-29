# Example of HPC approach in Python environment using Numba compiler

This repository contains examples of HPC approaches in Python environment for a 5-point stencil test problem, using the high-performance Python compiler Numba, running on the Santos Dumont (SD) supercomputer. Also includes examples running on Google Colab (GC). The implementations are in the Jupyter Notebook files. Other implementations and information can be found in [this other repository](https://github.com/efurlanm/bs21).

SD Notebooks (in Portuguese):

* [SD-acesso.ipynb](http://github.com/efurlanm/tama21/blob/main/acesso_sd.ipynb) - short introduction about setting up SD access and using the local computer
* [SD-numba-cpu.ipynb](http://github.com/efurlanm/tama21/blob/main/numba-cpu.ipynb) - Implementation in Fortran 90, Python, and Numba
* [SD-numba-cpu-mpi.ipynb](http://github.com/efurlanm/tama21/blob/main/numba-cpu-mpi.ipynb) - Parallel implementation using MPI and running on SD using the Slurm task and resource manager
* [SD-numba-sequana-gpu-threads.ipynb](http://github.com/efurlanm/tama21/blob/main/numba-sequana-gpu-threads.ipynb) - CPU and GPU implementation using threads and running on Sequana node
* [SD-numba-sequana-mpi.ipynb](http://github.com/efurlanm/tama21/blob/main/numba-sequana-mpi.ipynb) - CPU implementation using MPI and running on Sequana node

GC Notebooks (similar to those described above):

* [GC-acesso.ipynb](http://github.com/efurlanm/tama21/blob/main/GC-acesso.ipynb)
* [GC-numba-cpu.ipynb](http://github.com/efurlanm/tama21/blob/main/GC-numba-cpu.ipynb)
* [GC-numba-cpu-mpi.ipynb](http://github.com/efurlanm/tama21/blob/main/GC-numba-cpu-mpi.ipynb)
* [GC-numba-cpu.ipynb](http://github.com/efurlanm/tama21/blob/main/GC-numba-cpu.ipynb)




## Links of interest

* LNCC. SD Manual. https://sdumont.lncc.br/support_manual.php
* LNCC. FAQ. https://sdumont.lncc.br/support_FAQ.php
* LNCC. Several videos like architecture, programming, and others. https://sdumont.lncc.br/support_videos.php
* GOMES, ATA. Supercomputador Santos Dumont: o que é e o que faz? http://www.inf.ufrgs.br/erad2018/downloads/palestras/eradrs2018-sdumont.pdf
* GOMES, ATA. Supercomputador SDumont: visões de quem usa (de quem programa) e quem opera. https://interscity.org/assets/slides-Antonio-LNCC-27-11-20.pdf
* ATOS. Santos Dumont Architecture. https://sdumont.lncc.br/media/01_General_overview_of_SANTOS_DUMONT_architecture.pdf
* VALTER JR. Usando o Santos Dumont. https://www.linea.gov.br/wp-content/uploads/lineadbfiles/apresentacao/17%20-%20Usando%20o%20santos%20dumont%20(16_9).pdf
* CENAPAD-RJ. Materiais sobre Computação de Alto Desempenho. http://www.cenapad-rj.lncc.br/tutoriais/materiais-hpc/
* Projeto Cadase USP. Tutoriais SDumont. https://sites.usp.br/cadase/recursos-computacionais/tutoriais-sdumont/
