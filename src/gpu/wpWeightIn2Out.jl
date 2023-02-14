function wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
    wpWeightOut[CartesianIndex.(0x1 .+ wpIndexConvert, 0x1 .+ wpIndexIn)] = wpWeightIn
end
