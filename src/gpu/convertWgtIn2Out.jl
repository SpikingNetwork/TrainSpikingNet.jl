function convertWgtIn2Out(wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)
    wpWeightOut[CartesianIndex.(wpIndexConvert,wpIndexIn)] = wpWeightIn
    return wpWeightOut
end
