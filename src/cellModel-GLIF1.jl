#=
this file implements the GLIF1 model as defined in:

Supplementary Material for Generalized Leaky Integrate-And-Fire Models Classify Multiple Neuron Types
Teeter, Iyer, Menon, et al, Mihalas (2017)
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02717-4/MediaObjects/41467_2017_2717_MOESM1_ESM.pdf

to use it, put the following definitions in your params.jl:

thresh = ???
invR_mem = ???
invC_mem = ???
E_l = ???
vre = ???
dt = ???

cellModel_file = "cellModel-GLIF1.jl"
cellModel_args = (; thresh,
                    invR_mem, invC_mem, E_l, vre, dt)
=#

function cellModel_init!(v::AbstractVector{<:Real}, rng, args)
    randn!(rng, v)
    @. v = v * (args.thresh - args.vre) + args.vre
end
function cellModel_init!(v, rng, args)
    randn!(rng, ustrip(v))
    @. v = v * ustrip(args.thresh - args.vre) + args.vre
end

function cellModel_timestep!(i::Number, v, X, u, args)
    v[i] += args.dt * args.invC_mem[i] * (X[i] + u[i] - args.invR_mem[i] * (v[i] - args.E_l[i]))
end
function cellModel_timestep!(bnotrefrac::AbstractVector, v, X, u, args)
    @. v += bnotrefrac * args.dt * args.invC_mem * (X + u - args.invR_mem * (v - args.E_l))
end

function cellModel_spiked(i::Number, v, args)
    v[i] > args.thresh[i]
end
function cellModel_spiked!(bspike::AbstractVector, bnotrefrac::AbstractVector, v, args)
    @. bspike = bnotrefrac & (v > args.thresh)
end

function cellModel_reset!(i::Number, v, args)
    v[i] = args.vre
end
function cellModel_reset!(bspike::AbstractVector, v, args)
    @. v = ifelse(bspike, args.vre, v)
end
