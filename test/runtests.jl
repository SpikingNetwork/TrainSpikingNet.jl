using Test, JLD2

mkdir(joinpath(@__DIR__,"cpu-data"))
cp(joinpath(@__DIR__,"param.jl"), joinpath(@__DIR__,"cpu-data","param.jl"))
init_out = readlines(pipeline(`$(Base.julia_cmd())
                               $(joinpath(@__DIR__,"..","src","init.jl"))
                               $(joinpath(@__DIR__,"cpu-data"))`))
for iHz in findall(contains("Hz"), init_out)
  @test 0 < parse(Float64, match(r"([.0-9]+) Hz", init_out[iHz]).captures[1]) < 10
end

cp(joinpath(@__DIR__,"cpu-data"), joinpath(@__DIR__,"gpu-data"))
cpu_out = readlines(pipeline(`$(Base.julia_cmd())
                              $(joinpath(@__DIR__,"..","src","cpu","train.jl"))
                              $(joinpath(@__DIR__,"cpu-data"))`))
gpu_out = readlines(pipeline(`$(Base.julia_cmd())
                              $(joinpath(@__DIR__,"..","src","gpu","train.jl"))
                              $(joinpath(@__DIR__,"gpu-data"))`))

iHz = findlast(contains("Hz"), cpu_out)
@test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 10
iHz = findlast(contains("Hz"), gpu_out)
@test 0 < parse(Float64, match(r"([.0-9]+) Hz", cpu_out[iHz]).captures[1]) < 10

cpu_wpWeightIn = load(joinpath(@__DIR__,"cpu-data","wpWeightIn-ckpt1.jld2"), "wpWeightIn")
gpu_wpWeightIn = load(joinpath(@__DIR__,"gpu-data","wpWeightIn-ckpt1.jld2"), "wpWeightIn")
@test isapprox(reshape(transpose(cpu_wpWeightIn), 40, 1, 2000), gpu_wpWeightIn)

readlines(pipeline(`$(Base.julia_cmd())
                    $(joinpath(@__DIR__,"..","src","cpu","test.jl"))
                    $(joinpath(@__DIR__,"cpu-data"))`))

readlines(pipeline(`$(Base.julia_cmd())
                    $(joinpath(@__DIR__,"..","src","gpu","test.jl"))
                    $(joinpath(@__DIR__,"gpu-data"))`))

dcpu = load(joinpath(@__DIR__,"cpu-data/test.jld2"))
dgpu = load(joinpath(@__DIR__,"gpu-data/test.jld2"))
@test dcpu["nss"] == dgpu["nss"]
@test dcpu["ineurons_to_plot"] == dgpu["ineurons_to_plot"]
@test isapprox(dcpu["xtotals"], dgpu["xtotals"])
@test isapprox(dcpu["timess"], dgpu["timess"])
