module TrainSpikingNet

using LinearAlgebra, LinearAlgebra.BLAS, Random, JLD2, StatsBase, Statistics, SymmetricFormats, UnPack, CodecZlib
using CUDA, NNlib, NNlibCUDA, BatchedBLAS

using NLsolve

using Gadfly, Compose, DataFrames
import Cairo, Fontconfig

export param, config, init, train, test, plot

macro maybethread(scheduler, loop)
    if Threads.nthreads()>1
        quote
            if esc(:Kind)==:test
                $(esc(loop))
            else
                Threads.@threads $scheduler $(Expr(loop.head,
                                     Expr(loop.args[1].head, esc.(loop.args[1].args)...),
                                     esc(loop.args[2])))
            end
        end
    else
        quote $(esc(loop)); end 
    end
end

function vv2m(vv)
    m = zeros(eltype(eltype(vv)), maximum(length(x) for x in vv), length(vv))
    for i in eachindex(vv)
        m[1:length(vv[i]), i] .= vv[i]
    end
    return m
end

function vm2a(_P)
    if typeof(_P[1])<:SymmetricPacked
        P = zeros(eltype(_P[1]), maximum(length(x.tri) for x in _P), length(_P))
        for i in eachindex(_P)
            P[1:length(_P[i].tri), i] .= _P[i].tri
        end
    else
        P = zeros(eltype(_P[1]), maximum(size(x) for x in _P)..., length(_P))
        for i in eachindex(_P)
            P[1:size(_P[i],1), 1:size(_P[i],2), i] .= _P[i]
        end
    end
    return P
end

include("paramfn.jl")
include("initfn.jl")
include("rate2utarg.jl")
include("plotfn.jl")

using .Param

"""
    param(data_dir) -> (; Ncells, dt, etc.)

Evaluate the "param.jl" file in the directory `data_dir` and save the
required variables into "params.jld2" in the same directory.  The latter
is also returned as a NamedTuple as a convenience, for manual inspection,
but is not required by any other functions as they read directly from the
JLD2 file.

This function only needs to be called once when "param.jl" is created or
modified, or the seed of the random generator is `nothing` and a different
set of connections and weights is desired.
"""
param

"""
    config(data_dir, pu::Symbol=:cpu)

Load the code required to initialize, train, and test the model in `data_dir`
using the hardware specified by `pu`.  The latter can either be `:cpu`
or `:gpu`.

`param` must be called before `config`, and `config` must be called once
in each fresh session of Julia before `init`, `train`, and `test`.
"""
config

"""
    init(; itasks=[1], utarg_file=nothing, spikerate_file=nothing) -> (; weights, etc)

Generate the adjacency matrices and populate the weights according to the
plugins provided in "param.jl" for the model specified in the last call to
`config`.  The data are saved to JLD2 files in the `data_dir` input to `config`,
and are read by `train`.  They are also returned as a NamedTuple for easier
manual inspection.

By default, sinusoidal synaptic current targets with random phases are
generated.  If `utarg_file` is the full path to a JLD2 file containing an Nsteps
x Ncells x Ntasks array, then those are taken to be the target synaptic currents
instead.  If rather `spikerate_file` is a path (this keyword argument is
mutually exclusive with `utarg_file`), then the 3-D array therein is taken to be
peri-stimulus time histograms (PSTHs), and the Ricciardi method is used to
convert them to target synaptic currents.  In all cases, the synaptic currents
actually used are saved to `utarg.jld2` in `data_dir`.  As converting PSTHs to
currents is computationally expensive, subsequent calls to `init` should specify
this saved `utarg.jld2` file directly.

`param` and `config` must be called before `init`, the former in any
julia session and the latter in the current session.  `init` only needs
to be called once when "param.jl" is created or modified or the seed of
the random generator is `nothing` and a different set of connections and
weights is desired.
"""
init

"""
    train(; nloops = 1,
            correlation_interval = 1,
            save_best_checkpoint = false,
            restore_from_checkpoint = nothing,
            monitor_resources_used = nothing,
            return_P = false) -> (; weights, P)

Update the weights using the recursive least squares algorithm `nloops`
times, measuring the similarity between the actual and target synaptic
currents every `correlation_interval` iterations.  To continue training
a previous model, specify which of the saved weights to start from with
`restore_from_checkpoint`.  The learned plastic weights and updated covariance
matrix are saved as JLD2 files in the `data_dir` input to the last call to
`config` with the checkpoint added as a suffix.  The weights and optionally
the covariance are also returned as a NamedTuple for convenience.
"""
train

"""
    test(; ntrials = 1,
           ineurons_to_test = 1:16,
           restore_from_checkpoint = nothing,
           no_plot = false) -> (; nss, timess, utotals)

Keeping the weights frozen, evaluate the model `ntrials` times and save the
spike times and presynaptic currents of the neurons indexed as
`ineurons_to_test` to "test.jld2" in the `data_dir` input to the last call to
`config`.  These data are also returned as a NamedTuple for convenience.  `plot`
is immediately called on the results unless `no_plot` is `true`.  By default the
checkpoint with the highest number is used.  Use `restore_from_checkpoint` to
specify a different one.
"""
test

"""
    plot(test_file; ineurons_to_plot = 1:16)

Plot cumulative peri-stimulus time histograms (PSTHs) and averaged
presynaptic currents for the neurons indexed as `ineurons_to_plot`
in the JLD2 file specified in `test_file`.  Output PDFs are named
"test-{psth,syninput}-task{N}.pdf" are saved in the the same directory as
`test_file`.
"""
plot

end
