import torch
import torch.nn as nn
import numpy as np
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
                confidence_interval: float,
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
                output_pi: bool,
                dropout: float,
                dilation: int,
                n_models: int,
                pearce: bool, 
                **kwargs,
                ):
        super().__init__()

        self.pearce = pearce
        self.n_models = n_models
        self.output_pi = output_pi
        self.models = [TCN(layers=layers, net_type=net_type, depth=depth, dilation=dilation, activation=activation, kernel=kernel, bias=bias,
                len_traj=len_traj, dim_pred=8 if self.output_pi else 4, dropout=0 if self.output_pi else dropout) for _ in range(self.n_models)]

        self.stop_training = False
        self.len_traj = len_traj
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.visu = Visualizer(log_dir=self.log_dir)
        self.confidence_interval = confidence_interval

        X = torch.load(self.data_dir+'/X.pt')
        X_test = torch.load(self.data_dir+'/X_test.pt')
        T = X.shape[-1]
        l = T
        Y = torch.load(self.data_dir+'/Y.pt').T

        self.X_mean = X.mean(-1)
        self.X_std = X.std(-1)

        self.Y_mean = Y.mean(-1)
        self.Y_std = Y.std(-1)

        self.X_train = torch.stack([torch.Tensor(X[:,t-self.len_traj:t]) for t in range(self.len_traj, l)], 0)
        self.Y_train = torch.stack([torch.Tensor(Y[:,t]) for t in range(self.len_traj, l)], 0)

        #self.X_eval = torch.stack([torch.Tensor(X[:,t-self.len_traj:t]) for t in range(l, T)], 0)
        #self.Y_eval = torch.stack([torch.Tensor(Y[:,t]) for t in range(l, T)], 0)

        concat_X = torch.cat((X, X_test), -1)
        self.X_test = torch.stack([torch.Tensor(concat_X[:,-self.len_traj+t:t]) for t in range(-X_test.shape[-1], 0)], 0)

        self.X_train = (self.X_train - self.X_mean.unsqueeze(-1)) / self.X_std.unsqueeze(-1)
        self.X_test = (self.X_test - self.X_mean.unsqueeze(-1)) / self.X_std.unsqueeze(-1)

        self.init = init
        self.apply(self._init_weights)
        self._iter = 1
        self.loss = LOSS[loss]
        self.lr = lr
        self.optim_iter = optim_iter
        self.optimizer = optimizer
        self.optims = [OPTIM[self.optimizer](model.parameters(), self.lr) for model in self.models]
        self.scheduler = self._make_scheduler(scheduler)
        self.grad_clip = grad_clip
        self.eval_loss = 10e10
        self.validation = validation
        self.counts = 0

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
        lowers, uppers = self._generate_bounds(self.models[0], self.X_test)
        Y_test = (torch.cat([torch.stack((lowers[:,i],uppers[:,i]),-1) for i in range(4)], -1)).detach().cpu().numpy()
        with open(self.log_dir+'Y_test.npy', 'wb') as f:
            np.save(f, Y_test)

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
        
        pred_losses = []
        for i in range(self.n_models):
            if self.pearce:
                pred_loss = self._pearce_loss(self.X_train, self.Y_train, self.models[i])
            else:
                pred_loss = self.loss()(self.models[i](self.X_train), (self.Y_train-self.Y_mean.unsqueeze(0))/self.Y_std.unsqueeze(0))
            pred_losses.append(pred_loss)

            self.optims[i].zero_grad()
            pred_loss.backward()
            nn.utils.clip_grad_norm_(self.models[i].parameters(), self.grad_clip)
            self.optims[i].step()
        pred_loss = sum(pred_losses)

        if self.scheduler is not None:
            self.scheduler.step()

        #with torch.no_grad():
            #gn = sum([p.grad.pow(2).sum() for p in self.TCN.parameters() if p.grad is not None]).sqrt().item()

        self.visu.train_losses.append(pred_loss.item())
        self.eval()
        for i in range(self.n_models):
            for m in self.models[i].modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        
        if not self._iter % 10 or self._iter == self.optim_iter:
                if self.validation:
                    eval_loss = self._evaluate()
                    self.visu.eval_losses.append(eval_loss.item())
                    self.visu.plot_losses()
                    if self.eval_loss > eval_loss:
                        self.eval_loss = eval_loss
                        self.saved_iter = self._iter
                        self.saved_model = self._clone_state_dict()
                    elif self._iter - self.saved_iter >= 100:
                        self.stop_training = True
                else:
                    self._save_figs(batch, pred, embed)
                    self.saved_model = self._clone_state_dict()

        if pbar is not None:
            pbar.set_description(f'pred_loss: {float(pred_loss.item()): 4.2f}, '
                f'counts in bounds: {float(self.counts): 4.2f}, '
                f'validation_loss: {float(self.eval_loss): 4.2f}')
        
        self._iter += 1
        self.visu._iter += 1

    def _evaluate(self):
        pred_losses = []
        for i in range(self.n_models):
            pred_loss = self._pearce_loss(self.X_train, self.Y_train, self.models[i])
            pred_losses.append(pred_loss)
        return sum(pred_losses)

    def _clone_state_dict(self):
        state_dict = self.state_dict()
        return {k: state_dict[k].clone() for k in state_dict}

    def _pearce_loss(self, X, Y, model):
        self.counts = 0
        l = Y.shape[0]//168
        total_loss = 0
        for i in range(l): 
            lowers, uppers = self._generate_bounds(model, X[i*168:(i+1)*168])
            bools = torch.logical_and(uppers>Y[168*i:(i+1)*168],Y[168*i:(i+1)*168]>lowers)
            diffs = ((uppers - lowers)*bools.float()).sum()
            counts = bools.all(-1).sum()
            self.counts += counts/Y.shape[0]
            total_loss += diffs/(counts+1e-6) + (168/(0.05*0.95))*(torch.max(torch.Tensor([0, 0.95 - counts/168]))**2)
        return total_loss/Y.shape[0]

    def _generate_bounds(self, model, X, N=20):
        if self.pearce:
            if self.output_pi:
                pred = model(X)*torch.cat((self.Y_std,self.Y_std),-1).unsqueeze(0) + torch.cat((self.Y_mean,self.Y_mean),-1).unsqueeze(0)
                return pred[:,[0,2,4,6]], pred[:,[1,3,5,7]]
            else:
                #Generate bounds with dropout
                preds = torch.empty(N, X.shape[0], 4)
                for i in range(N):
                    preds[i] = model(X)*self.Y_std.unsqueeze(0) + self.Y_mean.unsqueeze(0)
                return torch.quantile(preds, 0.5 - self.confidence_interval/2, dim=0), torch.quantile(preds, 0.5 + self.confidence_interval/2, dim=0)
        else:
            return model(X)*self.Y_std.unsqueeze(0) + self.Y_mean.unsqueeze(0) - 0.98*self.Y_std.unsqueeze(0), model(X)*self.Y_std + self.Y_mean + 0.98*self.Y_std.unsqueeze(0)


class TCN(nn.Module):
    def __init__(self, layers=6, depth=256, activation='relu', kernel=5, bias=False, net_type='1d',
                 n_features=343, len_traj=50, dim_pred=4, dropout=0.2, dilation=2):
        super().__init__()
        self.n_features = n_features
        self.len_traj = len_traj
        self.activation = ACTIVATION[activation]
        self.bias = bias
        self.dim_pred = dim_pred
        self.depths = [int(depth)]*(layers-1) if not isinstance(depth, (list, tuple)) else depth
        self.kernels = [int(kernel)]*(layers-1) if not isinstance(kernel, (list, tuple)) else kernel
        self.type = net_type
        self.dropout = dropout
        self.dilation = dilation
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
        layers = [conv(_in, _out, kernel_size=kernel_size, padding=int((kernel_size-1)/2), bias=self.bias, dilation=self.dilation)]
        layers.append(self.activation)
        bn = nn.BatchNorm1d if d==1 else nn.BatchNorm2d
        layers.append(bn(_out, affine=True, track_running_stats=True))
        layers.append(nn.Dropout(p=self.dropout, inplace=True))
        return layers

    def _build_final_layers(self):
        layers = [nn.Linear(self.len_traj*self.depths[-1], self.dim_pred)]
        layers.append(nn.BatchNorm1d(self.dim_pred, affine=True, track_running_stats=True))
        return layers