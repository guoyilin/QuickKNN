for ((i=1;i<=16;++i));do
  awk 'BEGIN{srand()}{str=1;j=1;for(i=(ind-1)*8+1;i<=ind*8;++i){str=str" "j":"$i;j++} if(rand() <=1)print str}' ind="$i" $1 >train
  wc -l train
  ./cluster-src/sofia-kmeans --k $2 --init_type random --opt_type mini_batch_kmeans --mini_batch_size 100 --dimensionality 9 --iterations 1000 --training_file train  --model_out cluster/cluster.$i
done
