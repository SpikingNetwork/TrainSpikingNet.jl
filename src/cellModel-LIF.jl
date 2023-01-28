# similar to cellModel-GLIF1.jl, but more performant as it uses a single
# membrane time constant and no resting membrane potential.

function cellModel_timestep!(i::Number, v, X, u, args)
    v[i] += args.dt * args.invtau_mem[i] * (X[i] + u[i] - v[i])
end
function cellModel_timestep!(bnotrefrac::AbstractVector, v, X, u, args)
    @. v += bnotrefrac * args.dt * args.invtau_mem * (X + u - v)
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
