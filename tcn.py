import torch
import torch.nn as nn
from tqdm import tqdm
from args import LOSS, OPTIM, ACTIVATION
from visu import Visualizer

class TCNTrainer(nn.Module):
    def __init__(self,
                activation: str,
                data_dir: str,
                bias: bool,
                depth: int,
                kernel: int,
                layers: int,
                net_type: str,
                grad_clip: float, 
                init: str,
                len_traj: int,
                log_dir: str,
                loss: str,
                lr: int,
                num_workers: int, 
                optim_iter: int,
                optimizer: str,
                scheduler: str,
                validation: bool, 
                **kwargs,
                ):
        super().__init__()
        self.TCN = TCN(layers=layers, net_type=net_type, depth=depth, activation=activation, kernel=kernel, bias=bias,
                len_traj=len_traj)

        self.stop_training = False
        self.len_traj = len_traj
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.visu = Visualizer(log_dir=self.log_dir)

        X = torch.load(self.data_dir+'/X.pt')
        T = X.shape[-1]
        l = int(0.75*T)
        Y = torch.load(self.data_dir+'/Y.pt').T

        self.X_train = torch.stack([torch.Tensor(X[:,t-self.len_traj:t]) for t in range(self.len_traj, l)], 0)
        self.Y_train = torch.stack([torch.Tensor(Y[:,t]) for t in range(self.len_traj, l)], 0)

        self.X_eval = torch.stack([torch.Tensor(X[:,t-self.len_traj:t]) for t in range(l, T)], 0)
        self.Y_eval = torch.stack([torch.Tensor(Y[:,t]) for t in range(l, T)], 0)

        self.init = init
        self.apply(self._init_weights)
        self._iter = 1
        self.loss = LOSS[loss]
        self.lr = lr
        self.optim_iter = optim_iter
        self.optimizer = optimizer
        self.optim = OPTIM[self.optimizer](self.parameters(), self.lr)
        self.scheduler = self._make_scheduler(scheduler)
        self.grad_clip = grad_clip
        self.eval_loss = 10e10
        self.validation = validation

    def fit(self):
        self.fitted = True
        pbar = tqdm(range(self.optim_iter))
        for i in pbar:
            self._single_step(pbar)
            if self.stop_training:
                self._iter += 1
                self._evaluate()
                break
        self.visu.plot_losses()
        torch.save(self.saved_model, self.log_dir+'/saved_model.pt')

    def _init_weights(self, m):
        if isinstance(m,(nn.Linear,nn.Conv2d,nn.Conv1d,nn.ConvTranspose2d,nn.ConvTranspose1d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if self.init == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            if self.init == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            if self.init == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    def _make_scheduler(self, title):
        scheduler = None
        if title == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.optim_iter)
        elif title == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, patience=1000)
        return scheduler

    def _single_step(self, pbar=None):
        self.requires_grad_(True)
        self.train()

        pred = self.TCN(self.X_train)
        pred_loss = self.loss()(pred, self.Y_train)
        
        self.optim.zero_grad()
        pred_loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
        self.optim.step()

        if self.scheduler is not None:
            self.scheduler.step()

        with torch.no_grad():
            gn = sum([p.grad.pow(2).sum() for p in self.parameters() if p.grad is not None]).sqrt().item()

        self.visu.train_losses.append(pred_loss.item())
        self.eval()
        
        if not self._iter % 20 or self._iter == self.optim_iter:
            with torch.no_grad():
                if self.validation:
                    eval_loss = self._evaluate()
                    self.visu.eval_losses.append(eval_loss.item())
                    if self.eval_loss > eval_loss:
                        self.eval_loss = eval_loss
                        self.saved_iter = self._iter
                        self.saved_model = self._clone_state_dict()
                    elif self._iter - self.saved_iter >= 60:
                        self.stop_training = True
                else:
                    self._save_figs(batch, pred, embed)
                    self.saved_model = self._clone_state_dict()

        if pbar is not None:
            pbar.set_description(f'pred_loss: {float(pred_loss.item()): 4.2f}, '
                f'grad_norm: {float(gn): 4.2f}, '
                f'validation_loss: {float(self.eval_loss): 4.2f}')
        
        self._iter += 1
        self.visu._iter += 1

    def _evaluate(self):
        pred = self.TCN(self.X_eval)
        return self.loss()(pred, self.Y_eval)

    def _clone_state_dict(self):
        state_dict = self.state_dict()
        return {k: state_dict[k].clone() for k in state_dict}


class TCN(nn.Module):
    def __init__(self, layers=6, depth=256, activation='relu', kernel=5, bias=False, net_type='1d',
                 n_features=343, len_traj=50, dim_pred=4):
        super().__init__()
        self.n_features = n_features
        self.len_traj = len_traj
        self.activation = ACTIVATION[activation]
        self.bias = bias
        self.dim_pred = dim_pred
        self.depths = [int(depth)]*(layers-1) if not isinstance(depth, (list, tuple)) else depth
        self.kernels = [int(kernel)]*(layers-1) if not isinstance(kernel, (list, tuple)) else kernel
        self.type = net_type
        if self.type == 'conv2d-1d':
            self.conv2d_layers = nn.Sequential(*self._build_conv_block(1, self.depths[0], self.kernels[0], d=2))
        self.main = nn.Sequential(*(self._build_conv1d_layers() + (self._build_final_layers())))

    def forward(self, x):
        if self.type == 'conv2d-1d':
            x = self.conv2d_layers(x.unsqueeze(1))
            x = x.contiguous().view(x.shape[0], -1, x.shape[-1])
        return self.main(x)

    def _build_conv1d_layers(self):
        layers = self._build_conv_block(self.n_features * self.depths[0] if self.type == 'conv2d-1d' else self.n_features, self.depths[0], self.kernels[0])
        for i in range(len(self.depths)-1):
            layers += self._build_conv_block(self.depths[i], self.depths[i+1], self.kernels[i+1])
        layers += [nn.Flatten()]
        return layers

    def _build_conv_block(self, _in, _out, kernel_size, d=1):
        conv = nn.Conv1d if d==1 else nn.Conv2d
        layers = [conv(_in, _out, kernel_size=kernel_size, padding=int((kernel_size-1)/2), bias=self.bias)]
        layers.append(self.activation)
        bn = nn.BatchNorm1d if d==1 else nn.BatchNorm2d
        layers.append(bn(_out, affine=True, track_running_stats=True))
        return layers

    def _build_final_layers(self):
        layers = [nn.Linear(self.len_traj*self.depths[-1], self.dim_pred)]
        layers.append(nn.BatchNorm1d(self.dim_pred, affine=True, track_running_stats=True))
        return layers