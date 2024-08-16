function wpWeightIn2Out!(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)

    function kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
        i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
        j0 = threadIdx().y + (blockIdx().y - 1) * blockDim().y
        istride = blockDim().x * gridDim().x
        jstride = blockDim().y * gridDim().y

        i = i0
        @inbounds while i <= size(wpWeightIn,1)
            j = j0
            while j <= size(wpWeightIn,2)
                wpWeightOut[wpIndexConvert[i,j],wpIndexIn[i,j]] = wpWeightIn[i,j]
                j += jstride
            end
            i += istride
        end
        return nothing
    end

    kernel = @cuda launch=false kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn)
    threads, blocks = configurator(kernel, size(wpWeightIn,1), size(wpWeightIn,2))
    kernel(wpWeightOut, wpIndexIn, wpIndexConvert, wpWeightIn; threads=threads, blocks=blocks)
end
