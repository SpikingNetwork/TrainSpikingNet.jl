TrainSpikingNet.jl uses recursive least squares to train fluctation-driven
spiking recurrent neural networks to recapitulate arbitrary temporal
activity patterns.  See [Arthur, Kim, Chen, Preibisch, and Darshan
(2022)](https://www.biorxiv.org/content/10.1101/2022.09.26.509578v1.full)
for further details.


# Requirements #

The CPU version of TrainSpikingNet.jl can run on any machine.
To use a GPU you'll need Linux or Windows as the code (currently)
requires CUDA, and Nvidia does not support Macs.


# Installation #

Install Julia with [juliaup](https://github.com/JuliaLang/juliaup)
or by manually downloading the latest version from
[julialang.org](https://julialang.org/).

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
different phases.

First, make a copy of the parameters file.  Like this on Linux:

```
$ mkdir ~/data
$ cp src/param.jl ~/data
```

Now use `init.jl` to pick random synaptic weights and generate artificial
synaptic current targets of sinusoids.

```
$ julia --threads 4 $TSN_DIR/src/init.jl ~/data
mean excitatory firing rate: 3.427978515625 Hz
mean inhibitory firing rate: 6.153564453125 Hz
```

The `--threads` flag, which must come immediately after the `julia` command,
specifies how many CPU threads to use-- here we use only four as this is
a relatively small model.  The argument after `init.jl`, `~/data` here,
is the full (not relative) path to the desired `params.jl` file.

Printed to the terminal are the initial (i.e. the unlearned) firing rates.
And saved to disk are several files containing the matrices which define
the neural connectivity:

```
$ ls -t ~/data
rateX.jld2   wpWeightIn.jld2      wpIndexIn.jld2  w0Weights.jld2  param.jl
P.jld2       wpWeightX.jld2       utarg.jld2      w0Index.jld2
ncpOut.jld2  wpIndexConvert.jld2  X_stim.jld2     param.jld2
ncpIn.jld2   wpIndexOut.jld2      nc0.jld2        rng-init.jld2
```

To highlight just a few:  "wpWeightIn.jld2" stores the plastic synaptic
weights, "w0Weights.jld2" stores the static synaptic weights, and "utarg.jld2"
stores the target synaptic currents (sinusoidal in this case).  See the
comments in the code for more details.

Now use `train.jl` to iteratively update the plastic weights with sequential
presentations of the stimulus:

```
$ julia $TSN_DIR/src/gpu/train.jl --nloops 100 ~/data
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
```

The `--nloops` flag, which must come after `train.jl`, specifies how many iterations
of the training loop to perform.  The `--threads` flag is not used here as we are using
the GPU version of `train.jl`.

The correlations to the targets are printed to the terminal, and the
trained weights are stored in additional JLD2 files suffixed with "-ckpt"
for "checkpoint":

```
$ ls -t ~/data
P-ckpt100.jld2           P.jld2           wpIndexConvert.jld2  nc0.jld2        param.jl
wpWeightIn-ckpt100.jld2  ncpOut.jld2      wpIndexOut.jld2      w0Weights.jld2
wpWeightX-ckpt100.jld2   ncpIn.jld2       wpIndexIn.jld2       w0Index.jld2
rng-train.jld2           wpWeightIn.jld2  utarg.jld2           param.jld2
rateX.jld2               wpWeightX.jld2   X_stim.jld2          rng-init.jld2
```

Finally, use `test.jl` to plot the trained activities:

```
$ julia $TSN_DIR/src/gpu/test.jl --ntrials 100 ~/data
trial #1, task #1: 50.6 sec
trial #2, task #1: 8.94 sec
trial #3, task #1: 8.64 sec
<SNIP>
trial #100, task #1: 8.29 sec
```

The `--ntrials ` flag specifies how many iterations to perform, but this time
there is no learning.  We perform multiple iterations so that peri-stimulus
time histograms (PSTHs) with low firing rate neurons can be averaged over
many trials.

![synpatic inputs](/test-syninput.pdf)
![PSTH](/test-psth.pdf)

The figures above are saved to "test-{syninput,psth}-task1.pdf" and the
underlying data is stored in "test.jld2":

```
$ ls -t ~/data
test-psth-task1.pdf      rng-train.jld2   wpIndexConvert.jld2  w0Index.jld2
test-syninput-task1.pdf  rateX.jld2       wpIndexOut.jld2      param.jld2
test.jld2                P.jld2           wpIndexIn.jld2       rng-init.jld2
rng-test.jld2            ncpOut.jld2      utarg.jld2           param.jl
P-ckpt100.jld2           ncpIn.jld2       X_stim.jld2
wpWeightIn-ckpt100.jld2  wpWeightIn.jld2  nc0.jld2
wpWeightX-ckpt100.jld2   wpWeightX.jld2   w0Weights.jld2
```

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
  the default is "src/genStaticWeights-erdos-renyi.jl".  This is only used
  if K > 0.

  * `genRateX()` specifies the spike thresholds for the feed-forward neurons.
  The default is "src/genRateX-ornstein-uhlenbeck.jl".  This is only used
  if LX > 0.

  * `genXStim()` specifies the external input applied to each neuron.
  The default is "src/genXStim-ornstein-uhlenbeck.jl"

  * `genUTarget()` specifies the desired synaptic currents to learn.
  The default is "src/genUTarget-sinusoids.jl".  This is only used if a file
  with the targets is not supplied on the command line to `init.jl`.

For example, the "genStaticWeights_file" and "genStaticWeights_args"
variables in "src/param.jl" are a string and a dictionary, respectively.
The former specifies the path to a .jl file containing a function called
`genStaticWeights()` to which the latter is passed when `init.jl`
is executed.  `genStaticWeights()` defines and returns `w0Index`,
`w0Weights`, and `nc0` which together specify the static connections
between neurons based on the parameters `Ncells`, `Ne`, `pree`, `prie`,
`prei`, `prii`, `jee`, `jie`, `jei`, and `jii`.  Should the default code,
contained in "src/genStaticWeights-erdos-renyi.jl" not do what you want,
you can create your own file (e.g. "genStaticWeights-custom.jl") which
defines an alternative definition of `genStaticWeights()` (must be the
same name) based on a (possibly) alternative set of parameters.  Simply set
`genStaticWeights_file` in your custom parameters file to the full path to
your custom function, and `getStaticWeights_args` to the required parameters
with their values.

If your target synaptic inputs are defined algorithmically then you have to
use `genUTarget()`, but if they are stored in a file then you can also supply
its fullpath to `init.jl` using the `--utarg_file` argument.  This file will
be copied to `utarg.jld2`.  If your desired temporal activity patterns are
PSTHs instead of synaptic currents, use the `--spikerate_file` flag instead
and they will be converted to synaptic currents using the method of Ricciardi
(Brunel 2000, J. Comput. Neurosci; Richardson 2007, Phys. Rev. E) and saved
in `utarg.jld2`.  As this conversion can take awhile, you should subsquently
use the `--utarg_file` flag with this newly generated `utarg.jld2` file.

Finally, you'll need to make a copy of and edit "src/param.jl" to set various
constants, like the spike threshold, the refractory period, the synapse
and membrane time constants, the learning rate, the duration and time step
of the simulation, the floating point precision to use, the seed value for
random number generation, etc.  See the comments therein for more details.

Then, proceed as above with `init.jl`, `train.jl`, and `test.jl`.

Additional options for all of these scripts can be displayed with the
`--help` flag:

```
$ julia $TSN_DIR/src/gpu/train.jl --help
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


# Intel Math Kernel Library #

The CPU code can be sped up by about 10-50% on Intel machines using the
drop-in MKL package to replace the default OpenBLAS.  Install it like
this on Linux:

```
julia -e 'using Pkg; Pkg.add("MKL")'
echo "using MKL" >> ~/.julia/config/startup.jl
```
