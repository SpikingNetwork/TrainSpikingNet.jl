function funRollingAvg(stim_off,learn_every,t,wid,widInc,learn_nsteps,movavg,cnt,x,ci)

    startInd = Int(floor((t - stim_off - wid)/learn_every) + 1)
    endInd = Int(minimum([startInd + widInc, learn_nsteps]))
    if startInd > 0
        movavg[startInd:endInd] .+= x
        if ci == 1
            cnt[startInd:endInd] .+= 1
        end
    else
        movavg[1:endInd] .+= x
        if ci == 1
            cnt[1:endInd] .+= 1
        end
    end

    return movavg, cnt

end
