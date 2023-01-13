using Test, JLD2, SymmetricFormats

function compare_cpu_to_gpu(kind; ntasks=1, nloops=1)
    init_out = readlines(`$(Base.julia_cmd()) -t 2
                          $(joinpath(@__DIR__, "..", "src", "init.jl"))
                          --itasks 1:$ntasks
                          $(joinpath(@__DIR__, "scratch", "cpu-$kind"))`)
    write(joinpath(@__DIR__, "scratch", "cpu-$kind", "init.log"), join(init_out, '\n'))
    for iHz in findall(contains("Hz"), init_out)
        @test 0 < parse(Float64, match(r"([.0-9]+) Hz", init_out[iHz]).captures[1]) < 15
    end

    cp(joinpath(@__DIR__, "scratch", "cpu-$kind"), joinpath(@__DIR__, "scratch", "gpu-$kind"))
    cpu_out = readlines(`$(Base.julia_cmd()) -t 2
                         $(joinpath(@__DIR__, "..", "src", "cpu", "train.jl"))
                         --nloops $nloops
                         $(joinpath(@__DIR__, "scratch", "cpu-$kind"))`)
    write(joinpath(@__DIR__, "scratch", "cpu-$kind", "train.log"), join(cpu_out, '\n'))
    gpu_out = readlines(`$(Base.julia_cmd())
                         $(joinpath(@__DIR__, "..", "src", "gpu", "train.jl"))
                         --nloops $nloops
                         $(joinpath(@__DIR__, "scratch", "gpu-$kind"))`)
    write(joinpath(@__DIR__, "scratch", "gpu-$kind", "train.log"), join(gpu_out, '\n'))

    iHz = findlast(contains("Hz"), cpu_out)
    @test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 15
    iHz = findlast(contains("Hz"), gpu_out)
    @test 0 < parse(Float64, match(r"([.0-9]+) Hz", gpu_out[iHz]).captures[1]) < 15

    cpu_P = load(joinpath(@__DIR__, "scratch", "cpu-$kind", "P-ckpt$nloops.jld2"), "P")
    gpu_P = load(joinpath(@__DIR__, "scratch", "gpu-$kind", "P-ckpt$nloops.jld2"), "P")
    @test isapprox(kind=="SymmetricPacked" ? cat((x.tri for x in cpu_P)..., dims=2) :
                                             cat(cpu_P..., dims=3),
                   gpu_P)

    cpu_wpWeightIn = load(joinpath(@__DIR__, "scratch", "cpu-$kind", "wpWeightIn-ckpt$nloops.jld2"),
                          "wpWeightIn")
    gpu_wpWeightIn = load(joinpath(@__DIR__, "scratch", "gpu-$kind", "wpWeightIn-ckpt$nloops.jld2"),
                          "wpWeightIn")
    @test isapprox(cpu_wpWeightIn, gpu_wpWeightIn)
end

mkpath(joinpath(@__DIR__, "scratch"))

@testset "$kind" for kind in ("Array", "Symmetric", "SymmetricPacked")
    mkdir(joinpath(@__DIR__, "scratch", "cpu-$kind"))
    open(joinpath(@__DIR__, "scratch", "cpu-$kind", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "param.jl"))
            if startswith(line, "PType")
                println(fileout, "PType=$kind")
            else
                println(fileout, line)
            end
        end
    end

    compare_cpu_to_gpu(kind)

    if kind!="Array"
        cpu_wpWeightIn_Array = load(joinpath(@__DIR__, "scratch", "cpu-Array", "wpWeightIn-ckpt1.jld2"),
                                    "wpWeightIn")
        cpu_wpWeightIn_kind = load(joinpath(@__DIR__, "scratch", "cpu-$kind", "wpWeightIn-ckpt1.jld2"),
                                   "wpWeightIn")
        @test isapprox(cpu_wpWeightIn_Array, cpu_wpWeightIn_kind)
    end
end

@testset "pree=$pree, sig=$sig" for pree in [0.1, 0.0], sig in [0.65, 0.0]
    kind = string("pree", pree, "-sig", sig)
    mkdir(joinpath(@__DIR__, "scratch", "cpu-$kind"))
    open(joinpath(@__DIR__, "scratch", "cpu-$kind", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "param.jl"))
            if contains(line, ":pree => 0.1")
                println(fileout, replace(line, ":pree => 0.1" => ":pree => $pree"))
            elseif startswith(line, "sig =")
                println(fileout, "sig = $sig")
            elseif startswith(line, "L = ") && pree==0.0
                println(fileout, "L = 14")
            else
                println(fileout, line)
            end
        end
    end
    compare_cpu_to_gpu(kind)
end

@testset "voltage noise model" begin
    mkdir(joinpath(@__DIR__, "scratch", "cpu-voltagenoise"))
    open(joinpath(@__DIR__, "scratch", "cpu-voltagenoise", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "param.jl"))
            if contains(line, "noise_model=:current")
                println(fileout, replace(line, "noise_model=:current" => "noise_model=:voltage"))
            else
                println(fileout, line)
            end
        end
    end
    compare_cpu_to_gpu("voltagenoise")
end

@testset "Ricciardi" begin
    mkdir(joinpath(@__DIR__, "scratch", "ricciardi"))
    open(joinpath(@__DIR__, "scratch", "ricciardi", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "param.jl"))
            if startswith(line, "Ncells")
                println(fileout, "Ncells=4096")
            else
                println(fileout, line)
            end
        end
    end
    psth = Float64[1:100 100:-1:1 vcat(1:50,50:-1:1) (25 .+ 25*sin.(range(0,2*pi,100)))]
    save(joinpath(@__DIR__, "scratch", "ricciardi", "spikerates.jld2"), "psth", psth)
    init_out = readlines(`$(Base.julia_cmd()) -t 2
                         $(joinpath(@__DIR__, "..", "src", "init.jl"))
                         -s $(joinpath(@__DIR__, "scratch", "ricciardi", "spikerates.jld2"))
                         $(joinpath(@__DIR__, "scratch", "ricciardi"))`)
    write(joinpath(@__DIR__, "scratch", "ricciardi", "init.log"), join(init_out, '\n'))
    xtarg = load(joinpath(@__DIR__, "scratch", "ricciardi", "xtarg.jld2"))["xtarg"]
    dx = diff(xtarg, dims=1)
    @test all(dx[:,1].>0)
    @test all(dx[:,2].<0)
    @test all(dx[1:49,3].>0)
    @test all(dx[51:99,3].<0)
    @test all(dx[1:25,4].>0)
    @test all(dx[26:74,4].<0)
    @test all(dx[75:99,4].>0)
end

@testset "Int16" begin
    mkdir(joinpath(@__DIR__,  "scratch", "Int16"))
    open(joinpath(@__DIR__,  "scratch", "Int16", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "param.jl"))
            if startswith(line, "PPrecision")
                println(fileout, "PPrecision=Int16")
            elseif startswith(line, "PScale")
                println(fileout, "PScale=2^14")
            else
                println(fileout, line)
            end
        end
    end

    init_out = readlines(`$(Base.julia_cmd()) -t 2
                          $(joinpath(@__DIR__, "..", "src", "init.jl"))
                          $(joinpath(@__DIR__, "scratch", "Int16"))`)
    write(joinpath(@__DIR__, "scratch", "Int16", "init.log"), join(init_out, '\n'))
    for iHz in findall(contains("Hz"), init_out)
      @test 0 < parse(Float64, match(r"([.0-9]+) Hz", init_out[iHz]).captures[1]) < 10
    end

    gpu_out = readlines(`$(Base.julia_cmd())
                         $(joinpath(@__DIR__, "..", "src", "gpu", "train.jl"))
                         $(joinpath(@__DIR__, "scratch", "Int16"))`)
    write(joinpath(@__DIR__, "scratch", "Int16", "train.log"), join(gpu_out, '\n'))
    iHz = findlast(contains("Hz"), gpu_out)
    @test 0 < parse(Float64, match(r"([.0-9]+) Hz", gpu_out[iHz]).captures[1]) < 10
end

@testset "feed forward" begin
    mkdir(joinpath(@__DIR__, "scratch", "cpu-Lffwd"))
    open(joinpath(@__DIR__, "scratch", "cpu-Lffwd", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "param.jl"))
            if startswith(line, "Lffwd")
                println(fileout, "Lffwd = L>>1")
            else
                println(fileout, line)
            end
        end
    end
    compare_cpu_to_gpu("Lffwd")
end

@testset "multiple tasks" begin
    mkdir(joinpath(@__DIR__, "scratch", "cpu-twotasks"))
    cp(joinpath(@__DIR__, "param.jl"),
       joinpath(@__DIR__, "scratch", "cpu-twotasks", "param.jl"));
    compare_cpu_to_gpu("twotasks", ntasks=2, nloops=2)
end

@testset "test" begin
    run(pipeline(`$(Base.julia_cmd())
                  $(joinpath(@__DIR__, "..", "src", "cpu", "test.jl"))
                  $(joinpath(@__DIR__, "scratch", "cpu-twotasks"))`, stdout=devnull))

    run(pipeline(`$(Base.julia_cmd()) -t 1
                  $(joinpath(@__DIR__, "..", "src", "gpu", "test.jl"))
                  $(joinpath(@__DIR__, "scratch", "gpu-twotasks"))`, stdout=devnull))

    dcpu = load(joinpath(@__DIR__, "scratch", "cpu-twotasks", "test.jld2"))
    dgpu = load(joinpath(@__DIR__, "scratch", "gpu-twotasks", "test.jld2"))
    @test dcpu["nss"] == dgpu["nss"]
    @test dcpu["ineurons_to_test"] == dgpu["ineurons_to_test"]
    @test isapprox(dcpu["xtotals"], dgpu["xtotals"])
    @test isapprox(dcpu["timess"], dgpu["timess"])
end

@testset "learns" begin
    mkdir(joinpath(@__DIR__,  "scratch", "cpu-learns"))
    open(joinpath(@__DIR__,  "scratch", "cpu-learns", "param.jl"), "w") do fileout 
        for line in readlines(joinpath(@__DIR__, "..", "src", "param.jl"))
            if startswith(line, "seed=")
                println(fileout, "seed=1")
            else
                println(fileout, line)
            end
        end
    end

    init_out = readlines(`$(Base.julia_cmd()) -t 2
                          $(joinpath(@__DIR__, "..", "src", "init.jl"))
                          $(joinpath(@__DIR__, "scratch", "cpu-learns"))`)
    write(joinpath(@__DIR__, "scratch", "cpu-learns", "init.log"), join(init_out, '\n'))

    cp(joinpath(@__DIR__, "scratch", "cpu-learns"), joinpath(@__DIR__, "scratch", "gpu-learns"))
    cpu_out = readlines(`$(Base.julia_cmd()) -t 4
                         $(joinpath(@__DIR__, "..", "src", "cpu", "train.jl"))
                         --nloops 100 --correlation_interval 100
                         $(joinpath(@__DIR__, "scratch", "cpu-learns"))`)
    write(joinpath(@__DIR__, "scratch", "cpu-learns", "train.log"), join(cpu_out, '\n'))
    gpu_out = readlines(`$(Base.julia_cmd())
                         $(joinpath(@__DIR__, "..", "src", "gpu", "train.jl"))
                         --nloops 100 --correlation_interval 100
                         $(joinpath(@__DIR__, "scratch", "gpu-learns"))`)
    write(joinpath(@__DIR__, "scratch", "gpu-learns", "train.log"), join(gpu_out, '\n'))

    icor = findlast(contains("correlation"), cpu_out)
    @test parse(Float64, match(r"correlation: ([.0-9]+)", cpu_out[icor]).captures[1]) > 0.6
    icor = findlast(contains("correlation"), gpu_out)
    @test parse(Float64, match(r"correlation: ([.0-9]+)", gpu_out[icor]).captures[1]) > 0.6
end
