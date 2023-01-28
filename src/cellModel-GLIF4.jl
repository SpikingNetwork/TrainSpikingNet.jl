#=
this file implements the GLIF4 model as defined in:

Supplementary Material for Generalized Leaky Integrate-And-Fire Models Classify Multiple Neuron Types
Teeter, Iyer, Menon, et al, Mihalas (2017)
https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-017-02717-4/MediaObjects/41467_2017_2717_MOESM1_ESM.pdf

to use it, put the following definitions in your params.jl:

thresh = ???
thresh_s = zeros(Ncells)
b_s = ???
delta_thresh_s = ???
f_v = ???
delta_v = ???
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

cellModel_file = "cellModel-GLIF4.jl"
cellModel_args = (; thresh, thresh_s, b_s, delta_thresh_s, f_v, delta_v,
                    Ij, invtau_Ij, f_j, delta_Ij,
                    invR_mem, invC_mem, E_l, vre, dt)
=#

function cellModel_timestep!(i::Number, v, X, u, args)
    args.Ij[i,:] .-= args.dt * args.invtau_Ij .* args.Ij[i,:]
    v[i] += args.dt * args.invC_mem[i] * (X[i] + u[i] + sum(args.Ij[i,:]) - args.invR_mem[i] * (v[i] - args.E_l[i]))
    args.thresh_s[i] -= args.dt * args.b_s * args.thresh_s[i]
end
function cellModel_timestep!(bnotrefrac::AbstractVector, v, X, u, args)
    @. args.Ij -= bnotrefrac * args.dt * args.invtau_Ij' .* args.Ij
    @. v += bnotrefrac * args.dt * args.invC_mem * (X + u + $sum(args.Ij, dims=2) - args.invR_mem * (v - args.E_l))
    @. args.thresh_s -= bnotrefrac * args.dt * args.b_s * args.thresh_s
end

function cellModel_spiked(i::Number, v, args)
    v[i] > args.thresh[i] + args.thresh_s[i]
end
function cellModel_spiked!(bspike::AbstractVector, bnotrefrac::AbstractVector, v, args)
    @. bspike = bnotrefrac & (v > args.thresh + args.thresh_s)
end

function cellModel_reset!(i::Number, v, args)
    args.Ij[i,:] .= args.f_j * args.Ij[i,:] .+ args.delta_Ij
    v[i] = args.f_v * v[i] - args.delta_v
    args.thresh_s[i] += args.delta_thresh_s
end
function cellModel_reset!(bspike::AbstractVector, v, args)
    @. args.Ij = ifelse(bspike, args.f_j * args.Ij + args.delta_Ij, args.Ij)
    @. v = ifelse(bspike, args.f_v * v - args.delta_v, v)
    @. args.thresh_s = ifelse(bspike, args.thresh_s + args.delta_thresh_s, args.thresh_s)
end
