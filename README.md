TrainSpikingNet.jl uses recursive least squares to train fluctation-driven
spiking recurrent neural networks to recapitulate arbitrary temporal
activity patterns.  See [Arthur, Kim, Chen, Preibisch, and Darshan
(2022)](https://www.biorxiv.org/content/10.1101/2022.09.26.509578v1.full)
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

Install Julia with [juliaup](https://github.com/JuliaLang/juliaup)
or by manually downloading the latest version from
[julialang.org](https://julialang.org/).

## Julia REPL ##

Add TrainSpikingNet to your environment and test that everything works:

```
julia> ] add https://github.com/SpikingNetwork/TrainSpikingNet.jl
julia> ] test TrainSpikingNet
```

(Get out of Pkg mode by pressing the Delete key.)

That's it!

## Linux / PowerShell command line ##

If downloaded manually, modify the PATH environment variable to include
the path to the Julia executable.  Like this on Linux:

```
$ echo "export PATH=$PWD/julia-1.8.3/bin:$PATH" >> ~/.bashrc
```

Download the TrainSpikingNet.jl repository with either the [ZIP
link](https://github.com/SpikingNetwork/TrainSpikingNet.jl/archive/refs/heads/master.zip)
on github.com or by using git-clone:

```
$ git clone --depth 1 https://github.com/SpikingNetwork/TrainSpikingNet.jl.git
```

For convenience, add a new environment variable which contains the full
path to the just downloaded TrainSpikingNet directory.  Like this on Linux:

```
$ echo "export TSN_DIR=$PWD/TrainSpikingNet" >> ~/.bashrc
```

Install all of the required packages:

```
$ cd $TSN_DIR
$ julia --project=@.
        -e 'using Pkg;
            Pkg.activate(".");
            Pkg.instantiate();
            Pkg.activate("test");
            Pkg.instantiate()'
```

[Note that on Windows the double-quotes above need to be escaped by preceeding
them with a backslash.]

Finally, (and optionally) test that everything works:

```
cd $TSN_DIR/test
julia --project=@. runtests.jl
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

julia> size(state.wpWeightIn)  # L=29 here; 58=2L
(58, 4096)

julia> size(state.P)
(58, 58)
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
$ weights = train(nloops=100);
Loop no. 1, task no. 1
correlation: -0.03810218983457164
elapsed time: 64.27813005447388 sec
firing rate: 4.238525390625 Hz
Loop no. 2, task no. 1
correlation: -0.009730533926830837
elapsed time: 11.158457040786743 sec
firing rate: 3.5748291015625 Hz
Loop no. 3, task no. 1
correlation: 0.019285263967765184
elapsed time: 10.458786010742188 sec
firing rate: 3.177490234375 Hz
Loop no. 4, task no. 1
correlation: 0.04037737332828045
elapsed time: 10.786484003067017 sec
firing rate: 2.952392578125 Hz
Loop no. 5, task no. 1
correlation: 0.06431122625571872
elapsed time: 10.612313032150269 sec
firing rate: 2.79833984375 Hz
<SNIP>
Loop no. 100, task no. 1
correlation: 0.7138109340783079
elapsed time: 14.510313034057617 sec
firing rate: 2.1953125 Hz

julia> size(weights.wpWeightIn)
(58, 4096)
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
$ activities = test(ntrials=100);
trial #1, task #1: 50.6 sec
trial #2, task #1: 8.94 sec
trial #3, task #1: 8.64 sec
<SNIP>
trial #100, task #1: 8.29 sec

julia> activities.nss[1]  # no. of spikes on the first trial for the first 16 neurons
16-element Vector{UInt32}:
 0x0000000b
 0x00000002
 0x00000006
 0x0000000c
 0x00000009
 0x00000021
 0x0000000a
 0x00000007
 0x00000004
 0x00000001
 0x00000002
 0x00000000
 0x0000000d
 0x00000002
 0x00000008
 0x00000001
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

The `ineurons_to_plot` argument specifies the desired indices, and must be
a subset of that used in `test`, which defaults to `1:16`.

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

TrainSpikingNet is documented not just in this README, but also via comments
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
