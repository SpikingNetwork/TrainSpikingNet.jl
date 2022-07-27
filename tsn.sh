#!/usr/bin/env bash

nthreads=1
for ((i=0; i<=$#; i++)) ; do
 if [[ "${!i}" == "-t" ]] ; then
   j=$(( i+1 ))
   nthreads=${!j}
   set -- "${@:1:$((i-1))}" "${@:$((i+2))}"
   break
 fi
done

if [[ ( "$1" == "train" || "$1" == "test" ) && "$2" != "cpu" && "$2" != "gpu" ]] ; then
  echo ERROR: second argument must be either 'cpu' or 'gpu' when training or testing
  exit
fi

DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [[ "$1" == "install" ]] ; then
  cd $DIR
  julia -e 'using Pkg; \
            Pkg.activate("."); \
            Pkg.instantiate(); \
            Pkg.activate("test"); \
            Pkg.instantiate()'
  cd - >/dev/null

elif [[ "$1" == "unittest" ]] ; then
  julia --project=${DIR}/test ${DIR}/test/runtests.jl

elif [[ "$1" == "init" ]] ; then
  julia --project=$DIR -t $nthreads ${DIR}/src/init.jl "${@:2}"

elif [[ "$1" == "train" ]] ; then
  julia --project=$DIR -t $nthreads ${DIR}/src/$2/train.jl "${@:3}"

elif [[ "$1" == "test" ]] ; then
  julia --project=$DIR -t $nthreads ${DIR}/src/$2/test.jl "${@:3}"

elif [[ "$1" == "plot" ]] ; then
  julia --project=$DIR ${DIR}/src/plot.jl "${@:2}"

else
  echo tsn.sh: first argument must be either install, unittest, init, train, test, or plot
fi
