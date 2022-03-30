

if [ $NERSC_HOST == "perlmutter" ]
then
    module load python
    module load nersc-easybuild/.21.12
    module load CUDA/11.1.1-GCC-10.2.0
    module load cudnn/8.2.0
    conda activate tf2.7
else
    module load python
    module load cgpu
    module load cuda/11.1.1
    module load cudnn/8.1.0
    module load tensorrt/7.0.0.11-cuda-10.2
    conda activate tf2.7
fi