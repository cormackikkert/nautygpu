README file for nauty-gpu

This code contains nauty 2.7R4, parallelised with GPU acceleration.

# HOW TO USE

```
make 
```

Then you can run `dreadnaut` as before. See nauty's user manual, `nug27.pdf`.

# EXAMPLE
Here is an example of a use case (and how dreadnaut was interfaced for benchmarking). 
For additional graphs to test see https://pallini.di.uniroma1.it/Graphs.html

./dreadnaut
```
< example_graph.dre
i
q
```
(takes about ~1 second)

# VERSIONING
CUDA version: 11.8
GPU: NVIDIA GA106 (RTX 3060)

# ACKNOWLEDGEMENTS
All code is based on the original nauty code, written by:

Brendan McKay: Australian National University; Brendan.McKay@anu.edu.au
Adolfo Piperno: University of Rome "Sapienza"; piperno@di.uniroma1.it
Gunnar Brinkmann: University of Ghent; Gunnar.Brinkmann@UGent.be
Magma Administration: University of Sydney; admin@maths.usyd.edu.au
Patric Ostergard: Aalto Univerity; patric.ostergard@aalto.fi

See COPYRIGHT, for each file's author. I only edited .cu files (and their respective header files). In these .cu files
I only added the GPU specific code. 

The most interesting edits are in `naugraph.cu`, this is where the `refine` function resides.

# CORRECTNESS
I have tested the refinement procedure heavily, as that is the only code I edited. 
I've run lots of graphs, and algorithmically confirmed the outputted partition is 
indeed equitable, and has the same number of colours as the original c version of 
`dreadnaut`. However `dreadnaut` is a huge program, with many different code flows.
I have neglected the `longcode` aspect of the refinement routine, as it pertains to
the global search routine. If there are any errors, it is likely because of this.
