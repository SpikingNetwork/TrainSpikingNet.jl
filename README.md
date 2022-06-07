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
Test Summary: | Pass  Total
Int16         |    3      3
```

# Basic Usage #

Edit "src/param.jl" to set your network size, connectivity, stimulus
pattern, etc.  Optionally, make a copy of it:

```
$ mkdir ~/data

$ cp src/param.jl ~/data

$ vi ~/data/param.jl

$ grep -A11 -m1 innate ~/data/param.jl 
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
the method of Ricciardi (Brunel 2000, J. Comput. Neurosci; Richardson 2007,
Phys. Rev. E).  In all cases, the synaptic targets are stored in "xtarg.jld2",
which can be subsquently referenced using `-x`.

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
$ tbn.sh train gpu -n100 ~/data
Loop no. 1
correlation: -0.023547219725048148
elapsed time: 41.81254005432129 sec
firing rate: 4.606689453125 Hz
Loop no. 2
correlation: -0.019123938089304755
elapsed time: 5.6806960105896 sec
firing rate: 4.608642578125 Hz
Loop no. 3
correlation: -0.014787908839497654
elapsed time: 5.547835111618042 sec
firing rate: 4.6173095703125 Hz
Loop no. 4
correlation: 0.007915293563043593
elapsed time: 5.602427959442139 sec
firing rate: 4.59765625 Hz
Loop no. 5
correlation: 0.010099408049965756
elapsed time: 5.525408029556274 sec
firing rate: 4.60498046875 Hz
<SNIP>
Loop no. 100
correlation: 0.7823279444123159
elapsed time: 10.06592607498169 sec
firing rate: 5.996826171875 Hz

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
$ tbn.sh test gpu -n50 ~/data
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
usage: train.jl [-n NLOOPS] [-c CORRELATION_INTERVAL] [-s]
                [-r RESTORE_FROM_CHECKPOINT]
                [-m MONITOR_RESOURCES_USED] [-h] data_dir

positional arguments:
  data_dir              full path to the directory containing the
                        parameters file

optional arguments:
  -n, --nloops NLOOPS   number of iterations to train (type: Int64,
                        default: 1)
  -c, --correlation_interval CORRELATION_INTERVAL
                        measure correlation every C training loops.
                        default is every loop (type: Int64, default:
                        1)
  -s, --save_best_checkpoint
                        save the learned weights and covariance
                        matrices with the highest measured correlation
                        too.  default is to only save the last one
  -r, --restore_from_checkpoint RESTORE_FROM_CHECKPOINT
                        continue training from checkpoint R.  default
                        is to start from the beginning (type: Int64)
  -m, --monitor_resources_used MONITOR_RESOURCES_USED
                        measure power, cores, and memory usage every M
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


# Plugins #

The network architecture and initialization can be customized through
user-supplied code.  In the parameters file are five pairs of variables
which specify the path to .jl files and the arguments required by
the functions therein.  For example, the `genInitialWeights_file`
and `genInitialWeights_args` variables are a string and a dictionary,
respectively.  The former specifies the path to a .jl file containing
a function called `genInitialWeights()` to which the latter is passed
when `./tbn.sh init ...` is executed.  `genInitialWeights()` defines and
returns `w0Index`, `w0Weights`, and `nc0` which together specify the static
connections between neurons based on the parameters `pree`, `prie`, `prei`,
`prii`, `jee`, `jie`, `jei`, and `jii`.  Should this default code not do what
you want, you can create your own file (e.g. `myCustomGenInitialWeights.jl`)
which defines an alternative function `genInitialWeights()` (must be the
same name) based on a (possibly) alternative set of parameters.  Simply set
`genInitialWeights_file` to the full path to your custom function, and
`getInitialWeights_args` to the required parameters with their values.
Other code that can be plugged in include `genPlasticWeights()`, `genStim()`,
`genTarget()`, and `genFfwdRate()`.
