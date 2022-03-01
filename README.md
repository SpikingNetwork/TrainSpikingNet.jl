trainBalancedNet.jl uses recursive least squares to train fluctation-driven
spiking recurrent neural networks to recapitulate arbitrary activity patterns.
See Arthur, Kim, Preibisch, and Darshan (2022) for further details.


# Requirements #

The CPU version of trainBalancedNet.jl can run on any machine.
To use a GPU you'll need Linux or Windows as the code (currently)
requires CUDA, and Nvidia does not support Macs.


# Installation #

Install Julia with [juliaup](https://github.com/JuliaLang/juliaup)
or by manually downloading the latest version from
[julialang.org](https://julialang.org/).

Download the trainBalancedNet.jl repository with either the ZIP link on
github.com or by using git-clone:

```
$ git clone --depth 1 https://github.com/JaneliaSciComp/trainBalancedNet.git
Cloning into 'trainBalancedNet'...
remote: Enumerating objects: 44, done.
remote: Counting objects: 100% (44/44), done.
remote: Compressing objects: 100% (43/43), done.
remote: Total 44 (delta 5), reused 17 (delta 1), pack-reused 0
Unpacking objects: 100% (44/44), done.
```

Modify the unix PATH environment variable to include the path to the Julia
executable as well as this respository:

```
$ echo "export PATH=$PATH:~/bin/julia-1.7.1/bin:~/bin/trainBalancedNet" >> ~/.bashrc
```

Download all of the required packages:

```
$ tbn.sh install
  Activating project at `~/trainBalancedNet`
  Activating project at `~/trainBalancedNet/test`
```

Finally, test that everything works:

```
$ tbn.sh unittest
Test Summary: | Pass  Total
Array         |    5      5
Test Summary: | Pass  Total
Symmetric     |    6      6
Test Summary:   | Pass  Total
SymmetricPacked |    6      6
Test Summary: | Pass  Total
test          |    4      4
Test Summary: | Pass  Total
K=0           |    5      5
Test Summary: | Pass  Total
Ricciardi     |    7      7
```

# Basic Usage #

Edit "src/param.jl" to set your network size, connectivity, stimulus
pattern, etc.  Optionally, make a copy of it:

```
$ cp src/param.jl ~/data

$ vi ~/data/param.jl

$ grep -A11 -m1 innate ~/param.jl 
# innate, train, test time (ms)
train_duration = 1000.0
stim_on        = 800.0
stim_off       = 1000.0
train_time     = stim_off + train_duration

Nsteps = round(Int, train_time/dt)

# network size
Ncells = 4096
Ne = floor(Int, Ncells*0.5)
Ni = ceil(Int, Ncells*0.5)
```

Initialize a model with random weights.  By default, artificial synaptic
targets will be generated consisting of sinusoids.  Optionally, specify the
full path to a JLD2 file containing either the desired synaptic targets or
the corresponding spike rates using the `-x` and `-s` flags, respectively
(but not both!).  Spike rates will be converted to synptic currents using
the method of Ricciardi (XXXX).  In all cases, the synaptic targets are
stored in "xtarg.jld2", which can be subsquently referenced using `-x`.

```
$ tbn.sh init -t auto ~/data
mean excitatory firing rate: 3.427978515625 Hz
mean inhibitory firing rate: 6.153564453125 Hz

$ ls -t ~/data
P.jld2       wpWeightIn.jld2      wpIndexIn.jld2  nc0.jld2        param.jld2
ncpOut.jld2  wpIndexConvert.jld2  xtarg.jld2      w0Weights.jld2  rng-init.jld2
ncpIn.jld2   wpIndexOut.jld2      stim.jld2       w0Index.jld2    param.jl*
```

Now, Train a model by iteratively updating the weights with sequential
presentations of the stimulus.  The trained weights are stored in additional
JLD2 files and the correlations to the targets dumped to the standard output.
Use the `-t` flag to thread the CPU version of train.jl; it has no effect
on the GPU:

```
$ tbn.sh train gpu -n 100 -p 5 ~/data
Loop no. 1
elapsed time: 52.7787561416626 sec
firing rate: 4.5877685546875 Hz
Loop no. 2
elapsed time: 4.295434951782227 sec
firing rate: 4.5635986328125 Hz
Loop no. 3
elapsed time: 4.279933214187622 sec
firing rate: 4.534423828125 Hz
Loop no. 4
elapsed time: 4.239899158477783 sec
firing rate: 4.515625 Hz
Loop no. 5
correlation: 0.017975493144920994
elapsed time: 33.15775799751282 sec
firing rate: 4.5009765625 Hz
<SNIP>
Loop no. 100
correlation: 0.7493257340926016
elapsed time: 11.195262908935547 sec
firing rate: 5.787353515625 Hz

$ ls -t ~/data
P-ckpt100.jld2           ncpIn.jld2           xtarg.jld2      param.jld2
wpWeightIn-ckpt100.jld2  wpWeightIn.jld2      stim.jld2       rng-init.jld2
rng-train.jld2           wpIndexConvert.jld2  nc0.jld2        param.jl*
P.jld2                   wpIndexOut.jld2      w0Weights.jld2
ncpOut.jld2              wpIndexIn.jld2       w0Index.jld2
```

Finally, plot the trainined activities.  The underlying data is stored in
"test.jld2":

```
$ tbn.sh test gpu -n 50 ~/data
trial #1, 53.0 sec
trial #2, 9.34 sec
trial #3, 9.24 sec
<SNIP>
trial #50, 9.24 sec

$ ls -t ~/data
test-psth.svg            rng-train.jld2       wpIndexOut.jld2  w0Index.jld2
test-syninput.svg        P.jld2               wpIndexIn.jld2   param.jld2
test.jld2                ncpOut.jld2          xtarg.jld2       rng-init.jld2
rng-test.jld2            ncpIn.jld2           stim.jld2        param.jl*
P-ckpt100.jld2           wpWeightIn.jld2      nc0.jld2
wpWeightIn-ckpt100.jld2  wpIndexConvert.jld2  w0Weights.jld2
```

![synpatic inputs](/test-syninput.svg)
![PSTH](/test-psth.svg)

Additional options for all of these scripts can be displayed with `-h` or
`--help`:

```
$ tbn.sh train gpu -h
usage: train.jl [-n NLOOPS] [-p PERFORMANCE_INTERVAL]
                [-c SAVE_CHECKPOINTS] [-r RESTORE_FROM_CHECKPOINT]
                [-m MONITOR_RESOURCES_USED] [-h] data_dir

positional arguments:
  data_dir              full path to the directory containing the
                        parameters file

optional arguments:
  -n, --nloops NLOOPS   number of iterations to train (type: Int64,
                        default: 1)
  -p, --performance_interval PERFORMANCE_INTERVAL
                        measure correlation every P training loops.
                        default is never (type: Int64)
  -c, --save_checkpoints SAVE_CHECKPOINTS
                        save learned weights every C training loops.
                        default is to only save the last loop (type:
                        Int64)
  -r, --restore_from_checkpoint RESTORE_FROM_CHECKPOINT
                        continue training from checkpoint R.  default
                        is to start from the beginning (type: Int64)
  -m, --monitor_resources_used MONITOR_RESOURCES_USED
                        measure power, cores, and memory usage every R
                        seconds.  default is never (type: Int64)
  -h, --help            show this help message and exit
```


# File Formats #

All data are stored in JLD2 files, which are HDF5 files with a particular
structure inside designed to store Julia objects.  They should be readable
in any programming language that can read HDF5.

Plots are output as SVG files which should be readable by any internet
browser.


# Intel Math Kernel Library #

The CPU code can be sped up by about 10% on Intel machines using the
drop-in MKL package to replace the default OpenBLAS.  To install it,
change into the trainBalancedNet directory and execute `julia -e 'using Pkg;
Pkg.add("MKL")'`.  Then edit "src/cpu/train.jl" and add the line "using MKL"
just below the line starting with "using LinearAlgebra...".  Alternatively,
MKL can be automatically used for your other Julia code as well by adding
"using MKL" to "~/.julia/config/startup.jl" instead.
