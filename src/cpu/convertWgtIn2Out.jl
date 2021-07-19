function convertWgtIn2Out(Ncells,wpIndexIn,wpIndexConvert,wpWeightIn,wpWeightOut)

    for postCell = 1:Ncells
        for i = 1:ncpIn[postCell]
            preCell = Int(wpIndexIn[postCell,i])
            postCellConvert = Int(wpIndexConvert[postCell,i])
            wpWeightOut[postCellConvert,preCell] = wpWeightIn[postCell,i]
        end
    end

    return wpWeightOut
    
end
