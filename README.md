TrainSpikingNet.jl uses recursive least squares to train fluctation-driven
spiking recurrent neural networks to recapitulate arbitrary temporal
activity patterns.  See [Arthur, Kim, Chen, Preibisch, and Darshan
(2022)](https://www.biorxiv.org/content/10.1101/2022.09.26.509578v3.full)
for further details.


# Requirements #

The CPU version of TrainSpikingNet.jl can run on any machine.
To use a GPU you'll need Linux or Windows as the code (currently)
requires CUDA, and Nvidia does not support Macs.


# Two user interfaces #

Models can be trained by issuing commands on the Julia REPL.  Alternatively,
there is also a Linux / PowerShell interface that can be more convenient
when batching many jobs to a cluster, or for those not familiar with Julia.
Each are described in the next two sections.

# Installation #

Install Julia with [juliaup](https://github.com/JuliaLang/juliaup).

Or, manually download Julia from [julialang.org](https://julialang.org/).  See
the [platform specific instructions](https://julialang.org/downloads/platform).

## Julia REPL ##

Add TrainSpikingNet to your environment and, optionally, test that everything works:

```
julia> ] add TrainSpikingNet
julia> ] test TrainSpikingNet
```

The tests take about an hour, so be patient or just skip this step.

(Get out of Pkg mode by pressing the Delete key.)

That's it!

## Linux / PowerShell command line ##

First, follow the above installation instructions for the [Julia
REPL](#julia-repl).

Then, set an environment variable to the path to the TrainSpikingNet.jl
repository:

```
$ echo "export TSN_DIR=`julia -e 'using TrainSpikingNet; println(pathof(TrainSpikingNet))'`" >> ~/.bashrc
```

You'll need to either restart the terminal session for this change to take
effect, or source your rc file:

```
source ~/.bashrc
```

# Tutorial #

Here we walk through how to train a default network with 4096 neurons to
learn dummy sinusoidal activity patterns with identical frequencies but
different phases.  As this network is small, and not everyone has a GPU,
we'll just be using the CPU code here.

## Julia REPL ##

First, start Julia with the default number of threads.  You'll need to
do this on the OS command line as clicking on a desktop shortcut does not
provide a means to specify the number of threads, and the default is just one.

```
$ julia --threads auto
```

Then, make a copy of the parameters file:

```
julia> using TrainSpikingNet
julia> mkdir("my-data")
julia> cp(joinpath(dirname(pathof(TrainSpikingNet)), "param.jl"), "my-data/param.jl")
```

The parameters file is Julia code which sets various simulation variables
using constants and user-defined plugins.  To evaluate it, and save the
pertinent data to a JLD2 file, use the `param` command:

```
julia> p = param("my-data");

julia> p.dt  # the simulation time step in millisecods
0.1

julia> p.Ncells  # the number of neurons in the model
4096

julia> p.cellModel_file  # the plugin which defines membrane potential and spiking
"/home/arthurb/.julia/packages/TrainSpikingNet/XYpdq/src/cellModel-LIF.jl"
```

In addition to "param.jld2", alluded to above, there is also now a file
called "rng-init.jld2" in your data folder.  It contains the initial state
of the random number generator used to initialize the model, which can be
used to exactly reproduce an experiment.

```
julia> readdir("my-data")
3-element Vector{String}:
 "param.jl"
 "param.jld2"
 "rng-init.jld2"         # <--
```

Now use `config` to load simulation code that is customized to your particular
model architecture and machine hardware:

```
julia> config("my-data", :cpu)  # 2nd arg can also be :gpu
```

While `param` only needs to be called again if you change "params.jl",
`config` needs to be called each time you restart the Julia REPL.

Now use `init` to pick random synaptic weights and generate synaptic current
targets.  Without any keyword arguments, sinusoids are used for the latter.

```
julia> state = init();
mean excitatory firing rate: 3.427978515625 Hz
mean inhibitory firing rate: 6.153564453125 Hz

julia> size(state.wpWeightIn) 
(4096,)  # ==N

julia> size(state.wpWeightIn[1])
(58,)  # ==2L

julia> size(state.P)
(4096,)

julia> size(state.P[1])
(58,58)
```

Printed to the terminal are the initial (i.e. the unlearned) firing rates.
And saved to disk are several files containing the matrices which define
the neural connectivity:

```
julia> readdir("my-data")
17-element Vector{String}:
 "P.jld2"
 "X_stim.jld2"
 "nc0.jld2"
 "ncpIn.jld2"
 "ncpOut.jld2"
 "param.jl"
 "param.jld2"
 "rateX.jld2"
 "rng-init.jld2"
 "utarg.jld2"
 "w0Index.jld2"
 "w0Weights.jld2"
 "wpIndexConvert.jld2"
 "wpIndexIn.jld2"
 "wpIndexOut.jld2"
 "wpWeightIn.jld2"
 "wpWeightX.jld2"
```

To highlight just a few:  "wpWeightIn.jld2" stores the plastic synaptic
weights, "w0Weights.jld2" stores the static synaptic weights, and "utarg.jld2"
stores the target synaptic currents (sinusoidal in this case).  See the
comments in the code for more details.

Now use `train` to iteratively update the plastic weights with sequential
presentations of the stimulus:

```
julia> weights = train(nloops=100);
loop #  task #  elapsed time (s)  firing rate (Hz)  correlation
     1       1           9.94775           4.60229   -0.0317705
     2       1           7.30732           4.60620   -0.0289302
     3       1           7.19637           4.61389  -0.00390199
     4       1           7.17242           4.62964   0.00310603
     5       1           7.12784           4.63171    0.0127754
     6       1           7.18992           4.63525    0.0234085
     7       1           7.10075           4.64185    0.0309006
     8       1           7.14763           4.63428    0.0424313
     9       1           7.21539           4.64844    0.0573519
    10       1           7.08120           4.64685    0.0692450
<SNIP>
    98       1           7.06703           5.88428     0.769383
    99       1           7.18761           5.87146     0.660614
   100       1           9.60830           5.90820     0.728394

julia> size(weights.wpWeightIn)
(4096,)

julia> size(weights.wpWeightIn[1])
(58,)
```

The correlations to the targets are printed to the terminal, and the trained
weights and updated covariance matrix are stored in additional JLD2 files
suffixed with "-ckpt" for "checkpoint":

```
julia> readdir("my-data")
21-element Vector{String}:
 "P-ckpt100.jld2"             # <--
 "P.jld2"
 "X_stim.jld2"
 "nc0.jld2"
 "ncpIn.jld2"
 "ncpOut.jld2"
 "param.jl"
 "param.jld2"
 "rateX.jld2"
 "rng-init.jld2"
 "rng-train.jld2"
 "utarg.jld2"
 "w0Index.jld2"
 "w0Weights.jld2"
 "wpIndexConvert.jld2"
 "wpIndexIn.jld2"
 "wpIndexOut.jld2"
 "wpWeightIn-ckpt100.jld2"    # <--
 "wpWeightIn.jld2"
 "wpWeightX-ckpt100.jld2"     # <--
 "wpWeightX.jld2"
```

Finally, use `test` to plot the trained activities:

```
julia> activities = test(ntrials=100);
trial #1, task #1: 50.6 sec
trial #2, task #1: 8.94 sec
trial #3, task #1: 8.64 sec
<SNIP>
trial #100, task #1: 8.29 sec

julia> activities.times[1]  # spike times on the first trial for the first 16 neurons
16-element Vector{Vector{UInt16}}:
 [0x00c7, 0x140a, 0x47b7]
 []
 [0x0889, 0x2574, 0x25cd]
 [0x06ce, 0x085e, 0x098c, 0x0bab, 0x0fb9, 0x1053, 0x113d, 0x127d, 0x1643, 0x1f12, 0x2189, 0x21f4, 0x2564, 0x3610, 0x3d2c, 0x3f1f, 0x40ca, 0x433e, 0x455a, 0x4637]
 [0x03d4, 0x0622, 0x0aaf, 0x0cb1, 0x21e2, 0x35ed, 0x393d, 0x404b, 0x4400, 0x4bea, 0x4d02]
 [0x09e3, 0x0bb1, 0x4575]
 []
 [0x093e, 0x0a1e, 0x115a, 0x1296, 0x161c, 0x1865, 0x1b6b, 0x21eb, 0x2257, 0x22d9  …  0x3195, 0x3404, 0x3713, 0x38e1, 0x3dd3, 0x3f89, 0x4310, 0x4685, 0x4a7d, 0x4e04]
 [0x0001, 0x076f, 0x0aca]
 [0x0857]
 [0x0f0f, 0x1183, 0x124b, 0x1389, 0x1e70, 0x32f3, 0x3514, 0x3ed7, 0x42f7]
 [0x117b, 0x4837]
 [0x2340, 0x238e, 0x2401, 0x2624, 0x26c1, 0x308c, 0x31e9, 0x42f2, 0x46e9]
 [0x0001]
 [0x01db, 0x04ab, 0x083f, 0x1633, 0x1fdb, 0x20c9, 0x223a, 0x369d, 0x38d7, 0x3f95, 0x4475]
 [0x06e1, 0x0bdc, 0x1053, 0x13f6, 0x15c1, 0x1706, 0x19ee, 0x1b3b, 0x1ce0, 0x1e16  …  0x3b0e, 0x3e86, 0x3faa, 0x40d4, 0x4356, 0x44fe, 0x48ad, 0x49bb, 0x4bfd, 0x4df1]
```

The `ntrials` argument specifies how many iterations to perform, but this time
there is no learning.  We perform multiple iterations so that peri-stimulus
time histograms (PSTHs) with low firing rate neurons can be averaged over
many trials.

![synpatic inputs](/test-syninput.svg)
![PSTH](/test-psth.svg)

The figures above are saved to "test-{syninput,psth}-task1.pdf" and the
underlying data is stored in "test.jld2":

```
julia> readdir("my-data")
25-element Vector{String}:
 "P-ckpt100.jld2"
 "P.jld2"
 "X_stim.jld2"
 "nc0.jld2"
 "ncpIn.jld2"
 "ncpOut.jld2"
 "param.jl"
 "param.jld2"
 "rateX.jld2"
 "rng-init.jld2"
 "rng-test.jld2"
 "rng-train.jld2"
 "test-psth-task1.pdf"         # <--
 "test-syninput-task1.pdf"     # <--
 "test.jld2"                   # <--
 "utarg.jld2"
 "w0Index.jld2"
 "w0Weights.jld2"
 "wpIndexConvert.jld2"
 "wpIndexIn.jld2"
 "wpIndexOut.jld2"
 "wpWeightIn-ckpt100.jld2"
 "wpWeightIn.jld2"
 "wpWeightX-ckpt100.jld2"
 "wpWeightX.jld2"
```

Should you wish to replot at a later time, there is no need to call `test`
again.  `test` actually only produces the data in "test.jld", and then
immediately calls `plot` to create the figures.  You can do so directly
yourself, bypassing a lengthy call to `test`:

```
julia> plot("my-data/test.jld2", ineurons_to_plot=[1,5,9,13])
```

The `ineurons_to_plot` argument specifies the desired indices, and must be a
subset of that used in `test`, which defaults to `1:16`.  Note that `plot` (and
`test`) will overwrite any figures previously generated by `test` or `plot`, so
rename or move them accordingly.

A few things to note:

* `init`, for now, can only be done on a CPU,

* for small models, training on the CPU is probably faster than with a GPU,
and using fewer CPU threads might be faster than using more if your model
is particularly small,

* if using GPU(s), start Julia with the number of threads set to the number
of GPUs you have in your workstation.

## Linux / PowerShell command line ##

The above tutorial can recapitulated on the Linux command line as follows.

Copying the parameters file:

```
$ mkdir my-data
$ cp $TSN_DIR/src/param.jl my-data
```

Picking random synaptic weights and generating artificial synaptic current
targets of sinusoids:

```
$ julia --threads auto $TSN_DIR/src/init.jl $PWD/my-data
```

Note that `init.jl` automatically performs the equivalent of `param` in
the REPL.

The `--threads` flag, which must come immediately after the `julia` command,
specifies how many CPU threads to use.  The argument after `init.jl`,
`$PWD/my-data` here, must be the full (not relative) path to the desired
`params.jl` file.

Iteratively updating the plastic weights with sequential presentations of
the stimulus:

```
$ julia $TSN_DIR/src/train.jl --nloops 100 --gpu $PWD/my-data
```

Note that `train.jl` automatically performs the equivalent of `config`
in the REPL.

The `--nloops` flag, which must come after `train.jl`, specifies how many
iterations of the training loop to perform.  The `--threads` flag is not
used here as we are now using the GPU version of `train.jl` and have just
a single GPU in our workstation.

Plotting the trained activities:

```
$ julia $TSN_DIR/src/test.jl --ntrials 100 --gpu $PWD/my-data
$ julia $TSN_DIR/src/plot.jl --ineurons_to_plot=[1,5,9,13] $PWD/my-data/test.jld2
```

## Help ##

TrainSpikingNet.jl is documented not just in this README, but also via comments
in the code, as well as docstrings that are accessible both in the Julia
REPL and the OS command line.  Specifically, at the REPL, entering "?train"
results in:

```
help?> train
search: train TrainSpikingNet trailing_ones trailing_zeros AbstractString AbstractUnitRange

  train(; nloops = 1,
          correlation_interval = 1,
          save_best_checkpoint = false,
          restore_from_checkpoint = nothing,
          monitor_resources_used = nothing,
          return_P = false) -> (; weights, P)

  Update the weights using the recursive least squares algorithm nloops times, measuring the
  similarity between the actual and target synaptic currents every correlation_interval
  iterations. To continue training a previous model, specify which of the saved weights to
  start from with restore_from_checkpoint. The learned plastic weights and updated covariance
  matrix are saved as JLD2 files in the data_dir input to the last call to config with the
  checkpoint added as a suffix. The weights and optionally the covariance are also returned as
  a NamedTuple for convenience.
```

Equivalently, on the OS command line:

```
$ julia $TSN_DIR/src/train.jl --help
usage: train.jl [-n NLOOPS] [-c CORRELATION_INTERVAL] [-s]
                [-r RESTORE_FROM_CHECKPOINT]
                [-m MONITOR_RESOURCES_USED] [-g] [-h] data_dir

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
  -g, --gpu             use the GPU
  -h, --help            show this help message and exit
```

Docstrings are similarly available for `param`, `config`, `init`, `test`,
and `plot`.

Don't hesitate to file an issue on Github if you find a bug or have a feature
request.  The best place for usage help is either a GitHub discussion,
https://discourse.julialang.org/, or to contact one of the authors directly.

# Custom Usage #

To train a network for your own purpose you need to specify the neural
architecture, its target activity, and various simulation parameters.
These are done, respectively, through a set of plugin modules that define
adjacency matrices, by supplying a file with the desired synaptic inputs
or PSTHs, and by editing a copy of the default parameters file.

There are five plugin modules that define the architecture.  Each consists
of a .jl file containing a function of a specific name that inputs a custom
dictionary of parameters and outputs one or more arrays.  The path to the
.jl file as well as the parameters are both specified in "param.jl" with
variables ending in _file and _args, respectively.  You as the user write
these five functions to return custom adjacency matrices using parameters
of your choosing.  Defaults are supplied for each plugin.

  * `genPlasticWeights()` specifies the connectivity of the learned synapses.
  The default is "src/genPlasticWeights-erdos-renyi.jl"

  * `genStaticWeights()` specifies the connectivity of the fixed synapses.
  The default is "src/genStaticWeights-erdos-renyi.jl".  This is only used
  if K > 0.

  * `genRateX()` specifies the spike thresholds for the feed-forward neurons.
  The default is "src/genRateX-ornstein-uhlenbeck.jl".  This is only used
  if LX > 0.

  * `genXStim()` specifies the external input applied to each neuron.
  The default is "src/genXStim-ornstein-uhlenbeck.jl"

  * `genUTarget()` specifies the desired synaptic currents to learn.
  The default is "src/genUTarget-sinusoids.jl".  This is only used if a file
  with the targets is not supplied to `init`.

For example, the "genStaticWeights_file" and "genStaticWeights_args"
variables in "src/param.jl" are a string and a dictionary, respectively.
The former specifies the path to a .jl file containing a function called
`genStaticWeights()` to which the latter is passed when `init`
is executed.  `genStaticWeights()` constructs and returns `w0Index`,
`w0Weights`, and `nc0` which together specify the static connections
between neurons based on the parameters `Ncells`, `Ne`, `pree`, `prie`,
`prei`, `prii`, `jee`, `jie`, `jei`, and `jii`.  Should the default code,
contained in "src/genStaticWeights-erdos-renyi.jl", not do what you want,
you can create your own file (e.g. "genStaticWeights-custom.jl") which
defines an alternative definition of `genStaticWeights()` (must be the
same name) based on a (possibly) alternative set of parameters.  Simply set
`genStaticWeights_file` in your custom parameters file to the full path to
your custom function, and `getStaticWeights_args` to the required parameters
with their values.

If your target synaptic inputs are defined algorithmically then you have to
use `genUTarget()`, but if they are stored in a file then you can also supply
its fullpath to `init` using the `utarg_file` argument.  This file will
be copied to `utarg.jld2`.  If your desired temporal activity patterns are
PSTHs instead of synaptic currents, use the `spikerate_file` argument instead
and they will be converted to synaptic currents using the method of Ricciardi
(Brunel 2000, J. Comput. Neurosci; Richardson 2007, Phys. Rev. E) and saved
in `utarg.jld2`.  As this conversion can take awhile, you should subsquently
use the `utarg_file` flag with this newly generated `utarg.jld2` file.

Simulation parameters, like the learning rate, the duration and time step
of the simulation, the floating point precision to use, the seed value
for random number generation, etc., are specified in "src/param.jl".
Simply make a copy and edit.  See the comments therein for more details.

The cell model is also specified in "src/param.jl".  Much like the neural
architecture, this code is pulled out into a plugin, so that it is easy to
use a custom algorithm.  The "cellModel_file" and "cellModel_args" variables
are the path to a .jl file and a dictionary of parameters, respectively.
The former must define six functions, three each for the CPU and GPU:

* `cellModel_timestep!` updates the state variables at each tick of the clock

* `cellModel_spiked[!]` returns whether a neuron spiked or not

* `cellModel_reset!` sets the variables to their initial state

The CPU methods each input `i`, which is the cell index.  The GPU methods
operate on all cells in parallel, and input `bnotrefrac` and `bspike` which
are boolean vectors indicating which neurons are not in the refractory
period and which have spiked, respectively.  Other inputs include the
membrane voltage (`v`), and the external (`X`) and recurrent (`u`) currents.

Six cell models are supplied, five of which
are the Allen Institute's General Leak Integrate
and Fire (GLIF) models (see [Teeter, Iyer, Menon, et al, Mihalas,
2017](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02717-4/MediaObjects/41467_2017_2717_MOESM1_ESM.pdf)).
The sixth, "src/cellModel-LIF.jl", is the default, and is a
performance-optimized version of GLIF1.

Once you have configured all the above, proceed with `init`, `train`,
and `test` as described in the Tutorial.


# File Formats #

All data are stored in JLD2 files, which are HDF5 files with a particular
structure inside designed to store Julia objects.  They should be readable
in any programming language that can read HDF5.


# Intel Math Kernel Library #

The CPU code can be sped up by about 10-50% on Intel machines using the
drop-in MKL package to replace the default OpenBLAS.  Install it like
this from the Julia REPL:

```
julia> ] add MKL
```

Then add `using MKL` to your startup file:

```
julia> edit(joinpath(DEPOT_PATH[1], "config", "startup.jl"))
```

Or equivalently like this on Linux:

```
julia -e 'using Pkg; Pkg.add("MKL")'
echo "using MKL" >> ~/.julia/config/startup.jl
```


# Physical Units #

The state variables in the parameters file can have dimensions assigned
to them using [Unitful.jl](https://github.com/PainterQubits/Unitful.jl).
For example, `dt`, the simulation time step, could be set to `100μs`
instead of the default `0.1` with a comment that it is in milliseconds.
Doing so makes mixing power-of-ten prefixes easy and serves as a guard
against mixing incompatible units, but incurs a performance cost of about
10% depending on the model size and whether a GPU is used or not.  If this
tradeoff is acceptable, use "src/param-units.jl" as a template for your
parameters file instead of "src/params.jl".
