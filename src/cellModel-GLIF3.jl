#=
this file implements the GLIF3 model as defined in:

Supplementary Material for Generalized Leaky Integrate-And-Fire Models Classify Multiple Neuron Types
Teeter, Iyer, Menon, et al, Mihalas (2017)
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02717-4/MediaObjects/41467_2017_2717_MOESM1_ESM.pdf

to use it, put the following definitions in your params.jl:

thresh = ???
nIj = ???  # no. of currents
Ij = fill(0.0, Ncells, nIj)
invtau_Ij = 1 ./ [tau_I1, tau_I2, ...]
f_j = ???
delta_Ij = ???
invR_mem = ???
invC_mem = ???
E_l = ???
vre = ???
dt = ???

cellModel_file = "cellModel-GLIF3.jl"
cellModel_args = (; thresh,
                    Ij, invtau_Ij, f_j, delta_Ij,
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
    args.Ij[i,:] .-= args.dt * args.invtau_Ij .* args.Ij[i,:]
    v[i] += args.dt * args.invC_mem[i] * (X[i] + u[i] + sum(args.Ij[i,:]) - args.invR_mem[i] * (v[i] - args.E_l[i]))
end
function cellModel_timestep!(bnotrefrac::AbstractVector, v, X, u, args)
    @. args.Ij -= bnotrefrac * args.dt * args.invtau_Ij' .* args.Ij
    @. v += bnotrefrac * args.dt * args.invC_mem * (X + u + $sum(args.Ij, dims=2) - args.invR_mem * (v - args.E_l))
end

function cellModel_spiked(i::Number, v, args)
    v[i] > args.thresh[i]
end
function cellModel_spiked!(bspike::AbstractVector, bnotrefrac::AbstractVector, v, args)
    @. bspike = bnotrefrac & (v > args.thresh)
end

function cellModel_reset!(i::Number, v, args)
    args.Ij[i,:] .= args.f_j * args.Ij[i,:] .+ args.delta_Ij
    v[i] = args.vre
end
function cellModel_reset!(bspike::AbstractVector, v, args)
    @. args.Ij = ifelse(bspike, args.f_j * args.Ij + args.delta_Ij, args.Ij)
    @. v = ifelse(bspike, args.vre, v)
end
