function plot(test_file; ineurons_to_plot = 1:16)

    d = load(test_file)

    ineurons_to_test = d["ineurons_to_test"]
    ineurons_to_plot = something(ineurons_to_plot, ineurons_to_test)
    all(in.(ineurons_to_plot,[ineurons_to_test])) || error("ineurons_to_plot must be the same or a subset of ineurons_to_test in test.jl")

    if all(in.(ineurons_to_test,[ineurons_to_plot]))
        times = d["times"]
        utotal = d["utotal"]
    else
        itest = []
        for iplot in ineurons_to_plot
            push!(itest, findfirst(iplot .== ineurons_to_test))
        end
        times = similar(d["times"])
        utotal = similar(d["utotal"])
        for ij in eachindex(d["times"])
            times[ij] = d["times"][ij][itest]
            utotal[ij] = d["utotal"][ij][:,itest]
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

    ntrials = size(times,1)
    ntasks = size(times,2)
    nneurons = length(times[1])
    nrows = isqrt(nneurons)
    ncols = cld(nneurons, nrows)

    if typeof(Param.learn_every)<:Real
        _learn_every = Param.learn_every/1000
        _stim_off =  Param.stim_off/1000
        _train_time =  Param.train_time/1000
        _unit = "s"
        _dt = Param.dt/1000
    else
        _learn_every = ustrip(upreferred(Param.learn_every))
        _stim_off = ustrip(upreferred(Param.stim_off))
        _train_time = ustrip(upreferred(Param.train_time))
        _unit = string(unit(upreferred(Param.learn_every)))
        _dt = ustrip(upreferred(Param.dt))
    end

    for itask = 1:ntasks
        ps = Union{Plot,Context}[]
        for ci=1:nneurons
            df = DataFrame((t = (1:size(utarg,1)) .* _learn_every,
                            utarg = utarg[:,ineurons_to_plot[ci],itask],
                            utotal1 = utotal[1,itask][:,ci]))
            utotal_ci = hcat((x[:,ci] for x in utotal[:,itask] if !ismissing(x))...)
            df[!,:utotal_ave] = dropdims(median(utotal_ci, dims=2), dims=2)
            df[!,:utotal_disp] = dropdims(mapslices(x->mad(x, normalize=true),
                                                    utotal_ci, dims=2),
                                          dims=2)
            transform!(df, [:utotal_ave, :utotal_disp] => ByRow((mu,sigma)->mu+sigma) => :utotal_upper)
            transform!(df, [:utotal_ave, :utotal_disp] => ByRow((mu,sigma)->mu-sigma) => :utotal_lower)
            push!(ps, Gadfly.plot(df, x=:t, y=Col.value(:utarg, :utotal_ave, :utotal1),
                                  color=Col.index(:utarg, :utotal_ave, :utotal1),
                                  ymax=Col.value(:utotal_upper), ymin=Col.value(:utotal_lower),
                                  Geom.line, Geom.ribbon,
                                  Guide.colorkey(title="", labels=["data","model","model1"]),
                                  Guide.title(string("neuron #", ineurons_to_plot[ci])),
                                  Guide.xlabel("time ($_unit)", orientation=:horizontal),
                                  Guide.ylabel("synaptic input", orientation=:vertical),
                                  Guide.xticks(orientation=:horizontal)))
        end
        append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
        gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
                PDF(string(output_prefix, "-syninput-task$itask.pdf"), 8cm*ncols, 6.5cm*nrows)

        ps = Union{Plot,Context}[]
        for ci=1:nneurons
            times_cat = vcat((x[ci] for x in times[:,itask] if !ismissing(x))...)
            psth = fit(Histogram, times_cat * _dt,
                       _stim_off : _learn_every : _train_time)
            df = DataFrame(t=_learn_every : _learn_every : _train_time-_stim_off,
                           model=psth.weights ./ ntrials ./ _learn_every)
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
                                  Guide.xlabel("time ($_unit)", orientation=:horizontal),
                                  Guide.ylabel("spike rate", orientation=:vertical),
                                  Guide.xticks(orientation=:horizontal)))
        end
        append!(ps, fill(Compose.context(), nrows*ncols-nneurons))
        gridstack(permutedims(reshape(ps, ncols, nrows), (2,1))) |>
                PDF(string(output_prefix , "-psth-task$itask.pdf"), 8cm*ncols, 6.5cm*nrows)
    end
end
