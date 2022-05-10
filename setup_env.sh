source ~/miniconda3/etc/profile.d/conda.sh

conda create -n env_curvature python=3.6 numpy pandas networkx matplotlib seaborn jupyterlab --yes
conda activate env_curvature

conda install pytorch=1.9 torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia --yes
conda install pyg -c pyg -c conda-forge --yes

pip install wandb black numba jupyter[lab]

cd gdl
python setup.py develop
cd ..
