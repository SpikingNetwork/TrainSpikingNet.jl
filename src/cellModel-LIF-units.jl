# similar to cellModel-GLIF1.jl, but slightly more performant as it uses
# a single membrane time constant and no resting membrane potential

using Unitful: Ω

function cellModel_init!(v, rng, args)
    randn!(rng, ustrip(v))
    @. v = v * ustrip(args.thresh - args.vre) + args.vre
end

function cellModel_timestep!(i::Number, v, X, u, args)
    v[i] += args.dt * args.invtau_mem[i] * 1Ω * (X[i] + u[i] - v[i] / 1Ω)
end
function cellModel_timestep!(bnotrefrac::AbstractVector, v, X, u, args)
    @. v += bnotrefrac * args.dt * args.invtau_mem * 1Ω * (X + u - v / 1Ω)
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
