using Test, JLD2

for kind in ("Array", "Symmetric", "SymmetricPacked")
    @testset "$kind" begin
        mkdir(joinpath(@__DIR__, "cpu-$kind"))
        open(joinpath(@__DIR__, "cpu-$kind", "param.jl"), "w") do fileout 
            for line in readlines(joinpath(@__DIR__,"param.jl"))
                if startswith(line, "PType")
                    println(fileout, "PType=$kind")
                else
                    println(fileout, line)
                end
            end
        end

        init_out = readlines(pipeline(`$(Base.julia_cmd())
                                       $(joinpath(@__DIR__,"..","src","init.jl"))
                                       $(joinpath(@__DIR__,"cpu-$kind"))`))
        for iHz in findall(contains("Hz"), init_out)
          @test 0 < parse(Float64, match(r"([.0-9]+) Hz", init_out[iHz]).captures[1]) < 10
        end

        cp(joinpath(@__DIR__,"cpu-$kind"), joinpath(@__DIR__,"gpu-$kind"))
        cpu_out = readlines(pipeline(`$(Base.julia_cmd())
                                      $(joinpath(@__DIR__,"..","src","cpu","train.jl"))
                                      $(joinpath(@__DIR__,"cpu-$kind"))`))
        gpu_out = readlines(pipeline(`$(Base.julia_cmd())
                                      $(joinpath(@__DIR__,"..","src","gpu","train.jl"))
                                      $(joinpath(@__DIR__,"gpu-$kind"))`))

        iHz = findlast(contains("Hz"), cpu_out)
        @test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 10
        iHz = findlast(contains("Hz"), gpu_out)
        @test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 10

        cpu_wpWeightIn = load(joinpath(@__DIR__, "cpu-$kind", "wpWeightIn-ckpt1.jld2"),
                              "wpWeightIn")
        gpu_wpWeightIn = load(joinpath(@__DIR__, "gpu-$kind", "wpWeightIn-ckpt1.jld2"),
                              "wpWeightIn")
        @test isapprox(cpu_wpWeightIn, gpu_wpWeightIn)

        if kind!="Array"
            cpu_wpWeightIn_Array = load(joinpath(@__DIR__, "cpu-Array", "wpWeightIn-ckpt1.jld2"),
                                        "wpWeightIn")
            cpu_wpWeightIn_kind = load(joinpath(@__DIR__, "cpu-$kind", "wpWeightIn-ckpt1.jld2"),
                                       "wpWeightIn")
            @test isapprox(cpu_wpWeightIn_Array, cpu_wpWeightIn_kind)
        end
    end
end

run(pipeline(`$(Base.julia_cmd())
              $(joinpath(@__DIR__,"..","src","cpu","test.jl"))
              $(joinpath(@__DIR__,"cpu-Array"))`))

run(pipeline(`$(Base.julia_cmd())
              $(joinpath(@__DIR__,"..","src","gpu","test.jl"))
              $(joinpath(@__DIR__,"gpu-Array"))`))

dcpu = load(joinpath(@__DIR__,"cpu-Array/test.jld2"))
dgpu = load(joinpath(@__DIR__,"gpu-Array/test.jld2"))
@test dcpu["nss"] == dgpu["nss"]
@test dcpu["ineurons_to_plot"] == dgpu["ineurons_to_plot"]
@test isapprox(dcpu["xtotals"], dgpu["xtotals"])
@test isapprox(dcpu["timess"], dgpu["timess"])
