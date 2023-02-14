function plot(test_file; ineurons_to_plot = 1:16)

    d = load(test_file)
    ineurons_to_test = d["ineurons_to_test"]

    ineurons_to_plot = something(ineurons_to_plot, ineurons_to_test)

    all(in.(ineurons_to_plot,[ineurons_to_test])) || error("ineurons_to_plot must be the same or a subset of ineurons_to_test in test.jl")

    if all(in.(ineurons_to_test,[ineurons_to_plot]))
        nss = d["nss"]
        timess = d["timess"]
        utotals = d["utotals"]
    else
        itest = []
        for iplot in ineurons_to_plot
            push!(itest, findfirst(iplot .== ineurons_to_test))
        end
        nss = similar(d["nss"])
        timess = similar(d["timess"])
        utotals = similar(d["utotals"])
        for ij in eachindex(d["nss"])
            nss[ij] = d["nss"][ij][itest]
            timess[ij] = d["timess"][ij][itest,:]
            utotals[ij] = d["utotals"][ij][:,itest]
        end
    end

    Param = load(joinpath(dirname(test_file),"param.jld2"), "param")

    utarg = load(joinpath(dirname(test_file),"utarg.jld2"), "utarg")
    if isfile(joinpath(dirname(test_file),"rate.jld2"))
        rate = load(joinpath(dirname(test_file),"rate.jld2"), "rate")
    else
        rate = missing
    end

    output_prefix = splitext(test_file)[1]

    ntrials = size(nss,1)
    ntasks = size(nss,2)
    nneurons = length(nss[1])
    nrows = isqrt(nneurons)
    ncols = cld(nneurons, nrows)

    for itask = 1:ntasks
        ps = Union{Plot,Context}[]
        for ci=1:nneurons
            df = DataFrame((t = (1:size(utarg,1)) .* Param.learn_every/1000,
                            utarg = utarg[:,ineurons_to_plot[ci],itask],
                            utotal1 = utotals[1,itask][:,ci]))
            utotal_ci = hcat((x[:,ci] for x in utotals[:,itask])...)
            df[!,:utotal_ave] = dropdims(median(utotal_ci, dims=2), dims=2)
            df[!,:utotal_disp] = dropdims(mapslices(mad, utotal_ci, dims=2), dims=2)
            transform!(df, [:utotal_ave, :utotal_disp] => ByRow((mu,sigma)->mu+sigma) => :utotal_upper)
            transform!(df, [:utotal_ave, :utotal_disp] => ByRow((mu,sigma)->mu-sigma) => :utotal_lower)
            push!(ps, Gadfly.plot(df, x=:t, y=Col.value(:utarg, :utotal_ave, :utotal1),
                                  color=Col.index(:utarg, :utotal_ave, :utotal1),
                                  ymax=Col.value(:utotal_upper), ymin=Col.value(:utotal_lower),
                                  Geom.line, Geom.ribbon,
                                  Guide.colorkey(title="", labels=["data","model","model1"]),
                                  Guide.title(string("neuron #", ineurons_to_plot[ci])),
                                  Guide.xlabel("time (sec)", orientation=:horizontal),
                                  Guide.ylabel("synaptic input", orientation=:vertical),
                                  Guide.xticks(orientation=:horizontal)))
        end
        append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
        gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
                PDF(string(output_prefix, "-syninput-task$itask.pdf"), 8cm*ncols, 6.5cm*nrows)

        timess_cat = hcat(timess[:,itask]...)
        ps = Union{Plot,Context}[]
        for ci=1:nneurons
            psth = fit(Histogram, vec(timess_cat[ci,:]),
                       Param.stim_off : Param.learn_every : Param.train_time)
            df = DataFrame(t=Param.learn_every/1000 : Param.learn_every/1000 : Param.train_time/1000-1,
                           model=psth.weights ./ ntrials ./ Param.learn_every * 1000)
            if ismissing(rate)
                scale_color = Scale.color_discrete(n->Scale.default_discrete_colors(n+1)[2:end])
                cols = (:model, )
            else
                scale_color = Scale.default_discrete_colors
                df[!,:data] = rate[:, ineurons_to_plot[ci]]
                cols = (:data, :model)
            end
            push!(ps, Gadfly.plot(df, x=:t, y=Col.value(cols...), color=Col.index(cols...),
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
                PDF(string(output_prefix , "-psth-task$itask.pdf"), 8cm*ncols, 6.5cm*nrows)
    end
end
