using ArgParse, JLD2, Random, LinearAlgebra

import ArgParse: parse_item

function ArgParse.parse_item(::Type{Vector{Int}}, x::AbstractString)
    return eval(Meta.parse(x))
end

if !(@isdefined nss)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--ineurons_to_plot", "-i"
            help = "which neurons to plot.  must be the same or a subset of ineurons_to_test used in test.jl"
            arg_type = Vector{Int}
        "test_file"
            help = "full path to the JLD file output by test.jl.  this same directory needs to contain the parameters in param.jld2, the synaptic targets in xtarg.jld2, and (optionally) the spike rate in rate.jld2"
            required = true
    end

    parsed_args = parse_args(s)

    d = load(parsed_args["test_file"])
    ineurons_to_test = d["ineurons_to_test"]

    ineurons_to_plot = something(parsed_args["ineurons_to_plot"], ineurons_to_test)

    all(in.(ineurons_to_plot,[ineurons_to_test])) || error("ineurons_to_plot must be the same or a subset of ineurons_to_test in test.jl")

    if all(in.(ineurons_to_test,[ineurons_to_plot]))
        nss = d["nss"]
        timess = d["timess"]
        xtotals = d["xtotals"]
    else
        itest = []
        for iplot in ineurons_to_plot
            push!(itest, findfirst(iplot .== ineurons_to_test))
        end
        nss = []
        timess = []
        xtotals = []
        for i in eachindex(d["nss"])
            push!(nss, d["nss"][i][itest])
            push!(timess, d["timess"][i][itest,:])
            push!(xtotals, d["xtotals"][i][:,itest])
        end
    end

    include(joinpath(@__DIR__,"struct.jl"))
    p = load(joinpath(dirname(parsed_args["test_file"]),"param.jld2"), "p")

    xtarg = load(joinpath(dirname(parsed_args["test_file"]),"xtarg.jld2"), "xtarg")
    if isfile(joinpath(dirname(parsed_args["test_file"]),"rate.jld2"))
        rate = load(joinpath(dirname(parsed_args["test_file"]),"rate.jld2"), "rate")
    else
        rate = missing
    end

    output_prefix = splitext(parsed_args["test_file"])[1]
else
    ineurons_to_plot = parsed_args["ineurons_to_test"]

    xtarg = load(joinpath(parsed_args["data_dir"],"xtarg.jld2"), "xtarg")
    if isfile(joinpath(parsed_args["data_dir"],"rate.jld2"))
        rate = load(joinpath(parsed_args["data_dir"],"rate.jld2"), "rate")
    else
        rate = missing
    end

    output_prefix = joinpath(parsed_args["data_dir"], "test")
end

using Gadfly, Compose, DataFrames, StatsBase, Statistics

ntrials = length(nss)
nneurons = length(nss[1])
nrows = isqrt(nneurons)
ncols = cld(nneurons, nrows)

ps = Union{Plot,Context}[]
for ci=1:nneurons
    df = DataFrame((t = (1:size(xtarg,1)).*p.learn_every/1000,
                    xtarg = xtarg[:,ineurons_to_plot[ci]],
                    xtotal1 = xtotals[1][:,ci]))
    xtotal_ci = hcat((x[:,ci] for x in xtotals)...)
    df[!,:xtotal_mean] = dropdims(mean(xtotal_ci, dims=2), dims=2)
    df[!,:xtotal_std] = dropdims(std(xtotal_ci, dims=2), dims=2)
    transform!(df, [:xtotal_mean, :xtotal_std] => ByRow((mu,sigma)->mu+sigma) => :xtotal_upper)
    transform!(df, [:xtotal_mean, :xtotal_std] => ByRow((mu,sigma)->mu-sigma) => :xtotal_lower)
    push!(ps, plot(df, x=:t, y=Col.value(:xtarg, :xtotal_mean, :xtotal1),
                   color=Col.index(:xtarg, :xtotal_mean, :xtotal1),
                   ymax=Col.value(:xtotal_upper), ymin=Col.value(:xtotal_lower),
                   Geom.line, Geom.ribbon,
                   Guide.colorkey(title="", labels=["data","model","model1"]),
                   Guide.title(string("neuron #", ineurons_to_plot[ci])),
                   Guide.xlabel("time (sec)", orientation=:horizontal),
                   Guide.ylabel("synaptic input", orientation=:vertical),
                   Guide.xticks(orientation=:horizontal)))
end
append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
        SVGJS(string(output_prefix, "-syninput.svg"), 8cm*ncols, 6.5cm*nrows)

timess_cat = hcat(timess...)
ps = Union{Plot,Context}[]
for ci=1:nneurons
    psth = fit(Histogram, vec(timess_cat[ci,:]), p.stim_off : p.learn_every : p.train_time)
    df = DataFrame(t=p.learn_every/1000 : p.learn_every/1000 : p.train_time/1000-1,
                   model=psth.weights./ntrials./p.learn_every*1000)
    if ismissing(rate)
        scale_color = Scale.color_discrete(n->Scale.default_discrete_colors(n+1)[2:end])
        cols = (:model, )
    else
        scale_color = Scale.default_discrete_colors
        df[!,:data] = rate[:, ineurons_to_plot[ci]]
        cols = (:data, :model)
    end
    push!(ps, plot(df, x=:t, y=Col.value(cols...), color=Col.index(cols...),
                   Geom.line,
                   scale_color,
                   Guide.colorkey(title=""),
                   Guide.title(string("neuron #", ineurons_to_plot[ci])),
                   Guide.xlabel("time (sec)", orientation=:horizontal),
                   Guide.ylabel("spike rate", orientation=:vertical),
                   Guide.xticks(orientation=:horizontal)))
end
append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
        SVGJS(string(output_prefix , "-psth.svg"), 8cm*ncols, 6.5cm*nrows)
