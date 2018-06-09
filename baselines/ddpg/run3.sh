##
# @file   run3.sh
# @author Yibo Lin
# @date   Apr 2018
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
env_array=("Ant-v2")
# algorithm 
alg_array=("DDPG" "DDPGRM")
#alg_array=("DDPGRM")

# make log directory 
mkdir -p log 

# run experiments 
for env in "${env_array[@]}"; do 
for alg in "${alg_array[@]}"; do
for i in "${!generated_random_array[@]}"; do 
    if [[ $i < 9 ]]; then 
        continue 
    fi 
    generated_random_number="${generated_random_array[$i]}"
    rpt="log/${env}.${alg}.seed${generated_random_number}.log"
    echo "env = $env, generated random number = $generated_random_number, algorithm = $alg, rpt = $rpt"

    # call python script 
    python main.py --env-id=$env --seed=$generated_random_number --alg=$alg > $rpt
done 
done 
done 
