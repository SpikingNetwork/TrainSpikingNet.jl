using ArgParse, JLD2, Random, LinearAlgebra

if !(@isdefined nss)
    s = ArgParseSettings()

    @add_arg_table! s begin
        "test_file"
            help = "full path to the JLD file output by test.jl.  this same directory needs to contain the parameters in p.jld2, the synaptic targets in xtarg.jld2, and (optionally) the spike rate in rate.jld2"
            required = true
    end

    parsed_args = parse_args(s)

    d = load(parsed_args["test_file"])
    ineurons_to_plot = d["ineurons_to_plot"]
    nss = d["nss"]
    timess = d["timess"]
    xtotals = d["xtotals"]

    include(joinpath(@__DIR__,"struct.jl"))
    p = load(joinpath(dirname(parsed_args["test_file"]),"p.jld2"), "p")

    xtarg = load(joinpath(dirname(parsed_args["test_file"]),"xtarg.jld2"), "xtarg")
    if isfile(joinpath(dirname(parsed_args["test_file"]),"rate.jld2"))
        rate = load(joinpath(dirname(parsed_args["test_file"]),"rate.jld2"), "rate")
    else
        rate = missing
    end

    output_prefix = splitext(parsed_args["test_file"])[1]
else
    ineurons_to_plot = parsed_args["ineurons_to_plot"]

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
                   Guide.colorkey(title="", labels=["carbon","silicon","silicon1"]),
                   Guide.title(string("neuron #", ineurons_to_plot[ci])),
                   Guide.xlabel("time (sec)", orientation=:horizontal),
                   Guide.ylabel("synaptic input", orientation=:vertical),
                   Guide.xticks(orientation=:horizontal)))
end
append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
        SVGJS(string(output_prefix, "-syninput.svg"), 10cm*ncols, 7.5cm*nrows)

timess_cat = hcat(timess...)
ps = Union{Plot,Context}[]
for ci=1:nneurons
    psth = fit(Histogram, vec(timess_cat[ci,:]), p.stim_off : p.learn_every : p.train_time)
    df = DataFrame(t=p.learn_every/1000 : p.learn_every/1000 : p.train_time/1000-1,
                   silicon=psth.weights./ntrials./p.learn_every*1000)
    if ismissing(rate)
        cols = (:silicon, )
    else
        df[!,:carbon] = rate[:, ineurons_to_plot[ci]]
        cols = (:carbon, :silicon)
    end
    push!(ps, plot(df, x=:t, y=Col.value(cols...), color=Col.index(cols...),
                   Geom.line,
                   Guide.colorkey(title=""),
                   Guide.title(string("neuron #", ineurons_to_plot[ci])),
                   Guide.xlabel("time (sec)", orientation=:horizontal),
                   Guide.ylabel("spike rate", orientation=:vertical),
                   Guide.xticks(orientation=:horizontal)))
end
append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
        SVGJS(string(output_prefix , "-psth.svg"), 10cm*ncols, 7.5cm*nrows)
