##
# @file   run.sh
# @author Yibo Lin
# @date   Mar 2018
#
#!/bin/bash

# generate random numbers 
MAXCOUNT=10 # number of random numbers to generate
SEED=
declare -a generated_random_array

# function to generate random numbers 
random_numbers ()
{
local count=0
local number

echo -n "Generated random number array = "
while [ "$count" -lt "$MAXCOUNT" ]
do
  number=$RANDOM
  echo -n "$number "
  generated_random_array[$count]=$number
  let "count++"
done  
}

# initial random seed 
SEED=1000
RANDOM=$SEED

echo "Random seed = $SEED"
random_numbers
echo 

# environments
#env_array=("Ant-v2" "HalfCheetah-v2" "Hopper-v2" "Humanoid-v2" "HumanoidStandup-v2" "InvertedDoublePendulum-v2" "InvertedPendulum-v2" "Reacher-v2" "Swimmer-v2" "Walker2d-v2")
env_array=("LQG")
# algorithm 
alg_array=("Plain_GD" "Natural_GD" "Newton_Natural_GD")

# make log directory 
mkdir -p log 

# run experiments 
for env in "${env_array[@]}"; do 
for alg in "${alg_array[@]}"; do 
for i in "${!generated_random_array[@]}"; do 
    generated_random_number="${generated_random_array[$i]}"
    rpt="log/${env}.${alg}.seed${generated_random_number}.log"
    echo "env = $env, generated random number = $generated_random_number, algorithm = $alg, rpt = $rpt"

    # call python script 
    python LQG_model.py --seed=$generated_random_number --alg=$alg > $rpt
done 
done 
done
