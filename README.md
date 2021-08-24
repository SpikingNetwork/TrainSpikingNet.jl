# Installation #

Install Julia by downloading the latest version from
[julialang.org](https://julialang.org/).  Add `export JULIA_PROJECT=@.`
to your .bashrc file.  `cd` into this git repo directory.  Start Julia and
execute `]instantiate`.  Test that everything works by exiting out of
Julia and executing `julia test/runtests.jl`.

# Basic Usage #

Edit `src/params.jl` to set your network size, connectivity, stimulus
pattern, number of training loops, etc.

Execute `julia src/init.jl <path-to-params.jl> [<path-to-xtargs.jld>]`
to randomly set the weights.  If the full path to a JLD file containing
xtargs is not specified as an additional argument, artificial target
functions will be generated consisting of sinusoids.  The results
are stored in several `.jld` files.

Execute `julia src/{cpu,gpu}/train.jl <path-to-params.jl` to iteratively
update the weights with sequential presentations of the stimulus.
If `performance_interval` in `params.jl` is positive, the correlation
w.r.t. the stimulus is periodically measured.

Use `julia src/test.jl <path-to-params.jl` to plot the trainined activities.
