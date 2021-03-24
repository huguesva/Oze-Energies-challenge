srun hostname
hostname
ssh dbc.pasteur.fr
ssh g5.pasteur.fr
pwd
srun hostname
ssh myriad.pasteur.fr
hostname
srun hostname
ssh tars.pasteur.fr
module load singularity
singularity run --help
sinfo -e -o %c,%m,%G,%D -p gpu
srun  -p gpu  --qos=gpu  --gres=gpu:teslaP100:4,2   nvidia-smi
srun  -p gpu  --qos=gpu  --gres=gpu:teslaM40:1   nvidia-smi
module av
ssh tars.pasteur.fr
ls
/pasteur/projets/policy02/Larva-Screen
pwd
cd ..
pwd
cd ..
pwd
cd projets
cd policy02
cd Larva-Screen
ls
cd Leonardo/
ls
cd ..
ssh tars.pasteur.fr
ls
pwd
cd ..
pwd
cd ..
cd projets/
cd policy0
cd ..
cd policy02
ls
pwd
cd projets/
ls
cd policy02
cd Larva-Screen/
cd hugues/
ls
scp hvanhass@tars-submit0:t5 /Users/hugues/Documents/pasteur/structured-temporal-convolution
pwd
scp hvanhass@tars-submit0:/pasteur/projets/policy02/Larva-Screen/hugues/t5 /Users/hugues/Documents/pasteur/structured-temporal-convolution/data
scp hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5 /Users/hugues/Documents/pasteur/structured-temporal-convolution/data
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5 /Users/hugues/Documents/pasteur/structured-temporal-convolution/data
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5 /Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5 ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data
ls
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp hvanhass@tars-submit0:/pasteur/projets/policy02/Larva-Screen/hugues/t5 /Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5 ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/t5/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/  ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ /Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ /Users/hugues/Documents/pasteur/structured-temporal-convolution/
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ ~/Users/hugues/Documents/pasteur/structured-temporal-convolution/
ls
cd ..
ls
cd cbarre
cd hvanhass
ls
cd pasteur
cd ..
ls
cd ..
ls
cd pasteur
cd projets
ls
cd policy02
ls
cd Larva
cd Larva-Screen
ls
cd Chloe
ls
cd ..
cd hugues
ls
cd t5
ls
cd ..
ls
cd ..
ls
cd ..
ls
cd home
ls
cd ..
cd pasteur
ls
cd homes
cd hvanhass/
ls
cd ..
ls
cd..
ls
cd ..
ls
cd homes
cd hvanhass/
ls
rmdir structured-temporal-convolution/
rm -r structured-temporal-convolution/
mkdir structured-temporal-convolution
cd structured-temporal-convolution/
ls
cd structured-temporal-convolution/
ls
cd ..
rm -r structured-temporal-convolution/
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc python3 main.py
ls
srun -p dbc_pmo -q dbc main.py
python
pip
sudo dnf install python3
sudo apt-get update
ls
cd structured-temporal-convolution/
srun python3 main.py
module av Python
module load Python/3.7.2
pip3.7.2 install tqdm
pip3 install tqdm
pip install --upgrade pip
scp -r hvanhass@tars.pasteur.fr:/pasteur/projets/policy02/Larva-Screen/hugues/ /Users/hugues/Documents/pasteur/structured-temporal-convolution/data/
pip3 install --user tqdm
pip3 install --user attrdict
pip3 install --user datetime
pip3 install --user yaml
pip3 install --user argparse
pip3 install --user pyyaml
pip3 install --user numpy
pip3 install --user matplotlib
pip3 install --user sklearn
pip3 install --user celluloid
hvanhass@tars ~ $ squeue -u <hvanhass> --Format=jobid -t PD --noheader -p common -q fast
hvanhass@tars ~ $ squeue -u hvanhass --Format=jobid -t PD --noheader -p common -q fast
squeue -u hvanhass --Format=jobid -t PD --noheader -p common -q fast
sinfo -e -o %c,%m,%G,%D -p gpu
pip3 install --user torch torchvision
sudo singularity build c7-conda-lgarciac-cuda92.sif Singularity
©
~tru/singularity.d/containers/
sudo singularity build c7-conda-lgarciac-cuda92.sif Singularity
which singularity
ls
cd structured-temporal-convolution/
ls
cd configs
ls
cd ..
which singularity
ls
cd ..
ls
cd projets/
ls
cd policy02
ls
cd Larva-Screen
ls
cd hugues
pwd
ls
cd ..
ls
cd ..
ls
cd ..
ls
cd ..
ls
cd homes
cd hvanhass
ls
cd structured-temporal-convolution/
ls
cd runs
pwd
cd ..
ls
cd configs
pwd
ls
cd ..
ls
srun -p dbc_pmo -q dbc -c 4 main.py -J first 
module load SPAdes/3.7.0
srun -p dbc_pmo -q dbc -c 4 main.py -J first 
srun -p dbc_pmo -q dbc main.py -J first
srun -p dbc_pmo -q dbc python main.py -J first
srun -p dbc_pmo -q dbc python3 main.py -J first
module av Python
srun -p dbc_pmo -q dbc python3.6.0 main.py -J first
module load singularity
singularity --version
srun -p dbc_pmo -q dbc main.py
singularity --version
module load Python/3.6.0
srun -p dbc_pmo -q dbc python main.py -J first
module load Python/3.7.2
pip3.7.2 install PyYAML
pip3.7.2 install --user PyYAML
pip3 install --user PyYAML
srun -p dbc_pmo -q dbc python main.py -J first
pip3 install --user PyYAML==3.10
srun -p dbc_pmo -q dbc python main.py -J first
pip3 install --user PyYAML==3.11
srun -p dbc_pmo -q dbc python main.py -J first
pip3 install --user PyYAML==3.5.1
pip3 install --user PyYAML==5.3.1
srun -p dbc_pmo -q dbc python main.py -J first
pip3.7 install --user PyYAML==5.3.1
pip3 uninstall pyyaml
pip3 install pyyaml
pip install --upgrade pip
pip3 install --upgrade pip
srun -p dbc_pmo -q dbc python main.py -J first
pip3 install torch torchvision
source activate pytorch
torch
pip show
pip3 show
pip3 show torhc
pip3 show torch
srun -p dbc_pmo -q dbc python3 main.py -J first
singularity run docker://https://hub.docker.com/r/pytorch/pytorch
singularity run docker://uvarc/pytorch
singularity run docker://pytorch/pytorch
ls
cd ..
ls
mkdir singularity && cd singularity
pwd
ls
singularity run --home $HOME:/home docker://pytorch/pytorch
singularity -s exec run --home $HOME:/home docker://pytorch/pytorch
ls
cd ..
ls
cd ..
ls
cd hvanhass
singularity run --home $HOME:/home docker://pytorch/pytorch
singularity run --home $HOME:/ docker://pytorch/pytorch
ls
cd singularity/
cd ..
ls
singularity -s run --home $HOME:/home docker://pytorch/pytorch
ls
singularity -s run --home $HOME:/home docker://alkpark/pytorch
singularity -s run --home $HOME:/home docker://alfpark/pytorch
singularity -s run --home $HOME:/home docker://floydhub/pytorch
~tru/singularity.d/containers/c7-conda-lgarciac-cuda92-2020-04-15-1840.sif
modula.load singularity
module load singularity
which singularity
module load python/3.7.6
pip3 install virtualenv
module load Python/3.7.2
pip3 install virtualenv
virtualenv convenv
source convenv/bin/activate
pip3 install -r requirements.txt
ls
cd structured-temporal-convolution/
ls
srun -p dbc_pmo -q dbc -J first python main.py
cd ..
ls
cd singularity/
ls
cd ..
scp hvanhass@tars.pasteur.fr:/pasteur/homes/tru/singularity.d/containers/c7-conda-lgarciac-cuda92-2020-04-15-1840.sif singularity
ls
cd structured-temporal-convolution/
$ srun --qos gpu --gres=gpu:1 -p gpu -n1 -N1 --pty --preserve-env singularity exec -B /pasteur --nv ~tru/singularity.d/containers/c7-conda-lgarciac-cuda92-2020-04-15-1840.sif  python3 ~tru/python/torch-cuda.py 
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/c7-conda-lgarciac-cuda92-2020-04-15-1840.sif python3 main.py
module load sigularity
module load singularity
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
pip install attrdict
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
module load singularity
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
pip install tensorboard
pip3 install tensorboard
module load python/3.7.2
module load Python/3.7.2
pip3 install tensorboard
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
source convenv/bin/activate
ls
cd ..
source convenv/bin/activate
pip3 install matplotlib
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
git clone https://github.com/matplotlib/matplotlib
ls
cd matplotlib
pip install e .
cd ..
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
pip3 install matplotlib.pyplot
cd matplotlib/
cd pytest.ini 
pip install e .
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
cd ..
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv /singularity/torch.sif python3 main.py
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
cd structured-temporal-convolution/
cd singularity/
pwd
squeue -u hvanhass
sattach jobid.stepid
sattach 29152668.stepid
ls
cd structured-temporal-convolution/
ls
srun -p dbc_pmo -q dbc -J second -o runs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
ls
squeue -u hvanhass
sinfo -e -o %c,%m,%G,%D -p gpu
srun  -p gpu  --qos=gpu  --gres=gpu:teslaM40:1   nvidia-smi
squeue -u hvanhass
sinfo -e -o %c,%m,%G,%D -p dbc
sinfo -e -o %c,%m,%G,%D -p dbc_pmo
squeue -u hvanhass
ls
cd structured-temporal-convolution/
ls
cd singularity/
ls
cd ..
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
module load singularity
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
ls
cd runs
pwd
ls
cd ..
ls
cd configs
pwd
cd ..
ls
cd ..
ls
cd ..
ls
cd projets/
ls
cd policy02
ls
cd Larva-Screen
ls
cd hugues
pwd
cd ..
ls
cd ..
ls
cd ..
ls
cd homes/
ls
cd hvanhass/
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -J first --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
cd hvanhass/
ls
srun -p dbc_pmo -q dbc -J first -c 4 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -J first -c 8 --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -J label_stats -c 8 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
ls
cd structured-temporal-convolution/

sbatch run.sh
squeue -u hvanhass
ls
srun -p gpu --qos gpu -c 8 --gres=gpu:1 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
srun --qos gpu --gres=gpu:1 -p gpu -c 8 --gres=gpu:1 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
module load singularity
srun --qos gpu --gres=gpu:1 -p gpu -c 8 --gres=gpu:1 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/torch.sif python3 main.py
ls
cd singularity/
ls
cd ..
srun --qos gpu --gres=gpu:1 -p gpu -c 8 --gres=gpu:1 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun --qos gpu --gres=gpu:1 -p gpu -c 8 --gres=gp-J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 --gres=gp-J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
ls
srun -p dbc_pmo -q dbc -c 8 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun --jobid=29157332 --pty   top -u hvanhass
ssh tars-300
ssh tars-642
ssh tars-300
tmux new -s labelstat
tmux kill-session -a
tmux new -s labelstat
tmux new -s label
detach
squeue -u hvanhass
srun -p dbc_pmo -q dbc -c 8 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
module load singularity
srun -p dbc_pmo -q dbc -c 8 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -c 8 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
tmux new -s config
sudo apt install tmux
sudo yum install tmux
module load tmux
tmux
tmux new -s screen
tmux list-sessions
tmux attach -t config
tmux list-sessions
tmux a -t config
tmux list-sessions
tmux a -t screen
tmux list-sessions
tmux a -t config
tmux list-sessions
tmux a -t screen
tmux list-sessions
tmux a -t screen
tmux a -t config
ls
cd structured-temporal-convolution/
ls
tmux a -t screen
jupyter notebook
squeue -u hvanhass
tmux a -t screen
tmux new -s test
tmux a -t test
tmux new -s config1
squeue -u hvnahass
squeue -u hvanhass
tmux list-sessions
tmux a -t screen
tmux list-sessions
tmux a -t screen
cd structured-temporal-convolution/
tmux list_sessions
tmux list-sessions
tmux a -t config
tmux a -t config1
squeue -u hvanhass
tmux a -t config1
tmux list-sessions
tmux a -t screen
squeue =u hvanhass
squeue -u hvanhass
tmux list=sessions
tmux list-sessions
tmux a -t test
tmux a -t config
tmux a -t config2
tmux a -t config1
tmux a -t test
tmux a -t screen
tmux a -t test
tmux a -t config
tmux new -s config2
tmux a -t config1

srun -p dbc_pmo -q dbc -c 8 -J configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.pyb
srun -p gpu -q gpu --gres=gpu:1 -c 8 -J embed_gpu --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.p --seed 1
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
a + 2 = 3
2+4
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -c 8 -J label_stats --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J embed_size --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J various_configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J large_batch --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J config3 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config3
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 50
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
module load singularity
ls
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -c 8 -J various_configs --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2
module load singularity
srun -p dbc_pmo -q dbc -c 8 -J config1 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
cd structured-temporal-convolution/
srun -p dbc_pmo -q dbc -c 8 -J config1 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
srun -p dbc_pmo -q dbc -c 8 -J config2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py
tmux kill-sessions -a
pkill -f tmux
tmux kill-server
tmux kill-session -a
tmux kill-session config
tmux kill-session -t config
tmux list-sessions
ls
tmux a -t config
tmux list-sessions
tmux new -s config
tmux a -t config
cd structured-temporal-convolution/
 srun -p dbc_pmo -q dbc -c 8 -J config_2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
module load singularity
 srun -p dbc_pmo -q dbc -c 8 -J config_2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config2 --seed 30
 srun -p dbc_pmo -q dbc -c 8 -J config_2 --preserve-env singularity exec -B /pasteur --nv singularity/convenv.sif python3 main.py --config config1 --seed 30
tmux a -t config
tmux list-sessions
tmux new -s screen
tmux new -s embed
tmux a -t config
squeue -u hvanhass
tmux a -t embed
tmux new -s batch5
tmux list-sessions
tmux a -t screen
tmux list-sessions
tmux a -t embed
tmux list-sessions
tmux a -t config
tmux a -t screen
tmux new -s len_pred
squeue -u hvanhass
tmux a -t screen
tmux a -t config
squeue -u hvanhass
tmux list-sessions
tmux a -t embed
tmux a -t batch5
squeue -u hvanhass
tmux a -t config
squeue -u hvanhass
tmux list-sessions
tmux a -t len_pred
ls
cd structured-temporal-convolution/
cd batches/
pwd
tmux list-sessions
tmux a -t screen
tmux list-sessions
tmux a -t len_pred
tmux a -t batch5
tmux a -t screen
tmux list-sessions
tmux a -t config
ls
cd ..
ls
cd ..
ls
cd projets
ls
cd policy02
ls
cd ..
cd policy02
cd Larva-Screen
ls
cd hugues
pwd
tmux a -t Screen
tmux a -t screen
tmux list-sessions
tmux a-t config
tmux a -t config
tmux a -t screen
tmux a -t config
tmux list-sessions
tmux a -t embed
tmux a -t config
tmux list-sessions
squeue -u hvanhass
tmux a -t config
tmux a -t embed
tmux new -s embedding
tmux a -t embedding
tmux a -t embed
tmux a -t screen
tmux a -t embed
tmux a -t screen
tmux a -t embed
tmux a -t screen
tmux a -t embed
tmux a -t config
tmux a -t embedding
tmux new -s tanh
tmux a -t embedding
tmux a -t config
tmux new -s present
tmux new -s batchhunch
tmux a -t screen
tmux a -t embed
tmux new -s rotat
tmux new -s latent2
tmux a -t latent2
tmux a -t embed
tmux a -t batchhunch
tmux a -t latent2
tmux a -t batchhunch
tmux a -t embed
tmux a -t screen
tmux a -t embed
tmux a -t screen
tmux a -t embed
tmux a -t screen
tmux a -t latent2
tmux a -t latent2
tmux a -t screen
tmux a -t embed
tmux a -t screen
tmux a -t latent2
tmux a -t batchhunch
tmux a -t embedding
tmux a -t batchhunch
tmux a -t config
tmux a -t tanh
tmux a -t batchhunch
tmux a -t embed
tmux a -t embed
tmux a -t screen
tmux a -t latent2
tmux a -t screen
tmux a -t embed
tmux a -t batchhunch
tmux a -t embed
tmux a -t batchhunch
tmux a -t screen
tmux a -t batchhunch
tmux a -t batchhunc
tmux a -t tanh
tmux a -t latent2
tmux a -t config
tmux a -t embedding
tmux a -t rotat
tmux a -t latent2
tmux a -t tanh
tmux a -t rotat
tmux a -t config
tmux a -t batchhunch
tmux a -t screen
tmux a -t tanh
tmux a -t batchhunch
tmux a -t screen
tmux new -s manyparams
tmux a -t manyparams
tmux a -t rotat
tmux a -t config
tmux a -t latent2
tmux a -t Screen
tmux a -t present
tmux a -t latent2
tmux a -t present
tmux a -t latent2
tmux a -t present
tmux a -t latent2
tmux a -t present
tmux a -t Screen
tmux a -t screen
tmux a -t present
tmux a -t screen
tmux a -t latent2
tmux a -t screen
tmux a -t present
tmux a -t latent2
tmux a -t present
tmux a -t latent2
tmux a -t screen
tmux a -t present
tmux a -t screen
tmux a -t present
tmux a -t latent2
tmux a -t config
tmux a -t rotat
tmux a -t Screen
tmux a -t manyparams
tmux a -t embed2
tmux a -t batchhunch
tmux a -t present
tmux a -t screen
tmux a -t batchhunch
tmux a -t embed2
tmux a -t latent2
tmux a -t rotat
tmux a -t config
tmux a -t present
tmux a -t screen
tmux a -t present
tmux a -t screen
tmux a -t config
tmux a -t screen
tmux a -t config
tmux a -t screen
tmux a -t present
tmux a -t config
tmux a -t present
tmux a -t screen
tmux a -t present
tmux a -t screen
tmux a -t present
tmux a -t config
tmux a -t present
tmux a -t config
tmux a -t present
tmux a -t config
tmux a -t rotat
tmux a -t present
tmux a -t latent2
tmux a -t present
tmux a -t rotat
tmux a -t config
tmux a -t rotat
tmux a -t present
tmux a -t latent2
tmux a -t config
tmux a -t present
tmux a -t config
tmux a -t latent2
tmux a -t config
tmux a -t latent2
tmux a -t rotat
tmux a -t config
tmux a -t rotat
tmux a -t latent2
tmux a -t rotat
tmux a -t config
tmux a -t present
tmux a -t config
tmux a -t embed2
tmux a -t latent5
tmux a -t batchhunch
tmux a -t latent5
tmux a -t config
tmux a -t rotat
tmux a -t config
tmux a -t latent2
tmux a -t batchhunch
tmux a -t config
tmux a -t rotat
tmux a -t latent5
tmux a -t embed2
tmux a -t config
tmux a -t batchhunch
tmux a -t screen
tmux a -t batchhunch
tmux a -t screen
tmux a -t batchhunch
tmux a -t config
tmux a -t screen
tmux a -t embed2
tmux a -t rotat
tmux a -t present
tmux a -t latent5
tmux a -t latent2
tmux a -t present
tmux a -t rotat
tmux a -t screen
tmux a -t batchhunch
tmux a -t rotat
tmux a -t batchhunch
tmux a -t rotat
