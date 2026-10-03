#!/bin/bash

for i in {0..255}; do
    # adjust to the number of cpus/gpus you have. the goal is to download in parallel.
    cat >temp_slurm_script_$i.sh <<EOL
#!/bin/bash
#SBATCH --job-name=clip_$i
#SBATCH --dependency=singleton
#SBATCH --partition=
#SBATCH --cpus-per-task=1
#SBATCH --output=/path_to_output_logs_dir/slurm_%x.out

CONDA_ROOT=
env_name=

# loading conda environment
source \${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate \$env_name

python clips_youtubeasl.py --index "$i" \
                          --batch_size 50 \
                          --manifest_path /path_to_manifest_file/manifest.tsv \
                          --videos_dir /path_to_input_videos_directory \
                          --output_dir /path_to_output_clips_directory \
                          --done_file_path /path_to_done_file_directory/done.txt \
                          --problem_file_path /path_to_problem_file_directory/problem.txt

EOL

    # submit the temporary slurm script
    sbatch temp_slurm_script_$i.sh

    # remove the temporary slurm script
    rm temp_slurm_script_$i.sh
done