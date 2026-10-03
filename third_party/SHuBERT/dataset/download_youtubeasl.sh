#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=download_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

CONDA_ROOT=
env_name=

# loading conda environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $env_name

python download_youtubeasl.py --index "$i" \ 
                              --batch_size 3000 \ 
                              --time_limit 25200 \
                              --files_list /path_to_youtube_asl_txt_file_directory/youtube_asl_video_ids.txt \
                              --output_clips_directory /path_to_where_videos_are_saved \
                              --problem_file_path /path_to_problem_file_directory/problem_file.txt

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done