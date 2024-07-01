function wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    function kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        @inbounds for i=i0:istride:size(wpWeightIn,1), j=j0:jstride:size(wpWeightIn,2)
            wpWeightOut[wpIndexConvert[i,j],wpIndexIn[i,j]] = wpWeightIn[i,j]
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
    threads, blocks = configurator(kernel, size(wpWeightIn,1), size(wpWeightIn,2))
    kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn; threads=threads, blocks=blocks)
end
