function convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    wpWeightOut[CartesianIndex.(wpIndexConvert,wpIndexIn)] = permutedims(wpWeightIn, (2,1))
    return wpWeightOut
end
