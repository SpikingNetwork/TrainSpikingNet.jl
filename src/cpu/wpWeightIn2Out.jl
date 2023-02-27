function wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
    for postCell in eachindex(wpIndexIn)
        for i in eachindex(wpIndexIn[postCell])
            preCell = wpIndexIn[postCell][i]
            postCellConvert = wpIndexConvert[postCell][i]
            wpWeightOut[preCell][postCellConvert] = wpWeightIn[postCell][i]
        end
    end
end
