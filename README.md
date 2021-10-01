# Installation #

Install Julia by downloading the latest version from
[julialang.org](https://julialang.org/).  Add `export JULIA_PROJECT=@.`
to your .bashrc file.  `cd` into this git repo directory.  Start Julia and
execute `]instantiate`.  Test that everything works by exiting out of
Julia and executing `julia test/runtests.jl`.

# Basic Usage #

Edit `src/params.jl` to set your network size, connectivity, stimulus
pattern, etc.  Optionally copy it to another directory.

Execute `julia src/init.jl <dir-with-params.jl> [-x <path-to-xtargs.jld>]`
to randomly set the weights.  If the full path to a JLD file containing
the synaptic targets is not specified as an additional argument, artificial
targets will be generated consisting of sinusoids.  The results are stored
in several JLD files alongside params.jl.

Execute `julia [-t <T>] src/{cpu,gpu}/train.jl [-n <#-of-iterations>]
[-p <test-every-N-iterations>] <dir-with-params.jl` to iteratively update
the weights with sequential presentations of the stimulus.  The trained
weights are stored in additional JLD files and the correlations to the
targets dumped to the standard output.  Use the `-t` flag to thread the
CPU version of train.jl; it has no effect on the GPU.

Use `julia [-t <T>] src/{cpu,gpu}test.jl <dir-with-params.jl [-n
<#-of-repeated-trials]` to plot the trainined activities.  The underlying
data is stored in `test.jld`.  Use `julia src/plot.jl <path-to-test.jld>`
to re-plot without re-computing the trials.

Additional options for all of these scripts can be displayed with `-h` or
`--help`.
