# Installation #

Install Julia by downloading the latest version from
[julialang.org](https://julialang.org/).  Add `export JULIA_PROJECT=@.`
to your .bashrc file.  `cd` into this git repo directory.  Start Julia and
execute `]instantiate`.  Test that everything works by exiting out of
Julia and executing `julia test/runtests.jl`.

# Basic Usage #

Edit `src/params.jl` to set your network size, connectivity, stimulus
pattern, etc.

Execute `julia src/init.jl <path-to-params.jl> [-x <path-to-xtargs.jld>]`
to randomly set the weights.  If the full path to a JLD file containing
the synaptic targets is not specified as an additional argument, artificial
targets will be generated consisting of sinusoids.  The results are stored
in several `.jld` files.

Execute `julia src/{cpu,gpu}/train.jl [-n <#-of-iterations>] [-p
<test-every-N-iterations>] <path-to-params.jl` to iteratively update the
weights with sequential presentations of the stimulus.  The trained weights
are stored in additional JLD files and the correlations to the targes dumped
to the standard output.

Use `julia src/test.jl <path-to-params.jl` to plot the trainined activities.

Additional options for all of these scripts can be displayed with `-h` or
`--help`.
