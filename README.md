# gan4hep
Use GAN to generate HEP events

# Installation
```
pip install -e . 
```

# Instructions
Following command trains a GNN-based GAN:
```bash
train_gan4hep.py --config gan4hep/configs/config_gnn_gnn_v1.yml
```
The configuration can be found in the folder `gan4help/configs` and looks like the following:
One would change the `input_dir` and `output_dir` at least.
```yaml
batch_size: 512
debug: false
decay_base: 0.96
decay_epochs: 2
disable_tqdm: true
disc_batches: 10
disc_lr: 0.0003
evts_per_file: 5000
gamma_reg: 1.0
gan_type: gnn_gnn_gan
gen_lr: 0.0001
hadronic: true
input_dir: inputs
input_frac: 1.0
layer_size: 512
log_freq: 1000
loss_type: logloss
max_epochs: 100
noise_dim: 8
num_epochs: 100
num_layers: 5
num_processing_steps: null
output_dir: gnn_gnn/v1
patterns: '*'
shuffle_size: -1
val_batches: 5
verbose: INFO
warm_up: true
with_disc_reg: true
```