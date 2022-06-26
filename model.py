import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from cl_cv_classifier import Net 

class D_Cos(nn.Module):
    def __init__(self):
        super(D_Cos, self).__init__()
        self.margin = False
        
        if torch.cuda.is_available():
            self.device ='cuda'
        else:
            self.device ='cpu'

    def forward(self, p, z):
        """detach() remove the tensor from a computation graph and constructs a new view
        on a tensor which is declared not to need gradients"""       
        z = z.detach()
    
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        
        output_cos = -((p * z).sum(dim=1).mean())
        if self.margin:
            margin = 0.8          
            ones = torch.ones(size = output_cos.shape).to(self.device)
            output_cos = (ones + torch.add(output_cos, - margin)).double()
            null = torch.tensor([0], dtype=torch.double).to(self.device)
            output_cos = torch.where(output_cos < null, null, output_cos)
            return output_cos
        else:
            return output_cos

    
class D_MSE(nn.Module):
    def __init__(self):
        super(D_MSE, self).__init__()
        self.margin = False
        
        if torch.cuda.is_available():
            self.device ='cuda'
        else:
            self.device ='cpu'

    def forward(self, p, z):     
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        MSE = torch.square(p - z).sum(dim=1).mean() 
        
        if self.margin:
            margin = 0.8       
            output_mse = torch.add(MSE, - margin).double()
            null = torch.tensor([0], dtype=torch.double).to(self.device)
            output_mse = torch.where(output_mse < null, null, output_mse)
            return output_mse
        else:
            return MSE
    
class D_CE(nn.Module):
    def __init__(self):
        super(D_CE, self).__init__()
        self.margin = False
        
        if torch.cuda.is_available():
            self.device ='cuda'
        else:
            self.device ='cpu'

    def forward(self, p, z):     
        z = z.detach()  
        softmax_p = nn.Softmax(dim=1)(p)
        softmax_z = nn.Softmax(dim=1)(z)
        CE = -(softmax_z*torch.log(softmax_p)).sum(dim=1).mean() 
        
        if self.margin:
            margin = 0.8          
            output_ce = torch.add(CE, - margin).double()
            null = torch.tensor([0],dtype=torch.double).to(self.device)
            output_ce = torch.where(output_ce < null, null, output_ce)
            return output_ce
        else:
            return CE


class Model(nn.Module):
    def __init__(self, args, downstream=False):
        super(Model, self).__init__()
        
#        resnet18 = models.resnet18(pretrained=False)       
#        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
#        backbone_in_channels = resnet18.fc.in_features

        net = Net()       
        self.backbone = nn.Sequential(*list(net.children())[:-1])
        backbone_in_channels = net.fc.in_features     
        
        proj_hid, proj_out = args.proj_hidden, args.proj_out
        pred_hid, pred_out = args.pred_hidden, args.pred_out

        self.projection = nn.Sequential(
            nn.Linear(backbone_in_channels, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_hid),
            nn.BatchNorm1d(proj_hid),
            nn.ReLU(),
            nn.Linear(proj_hid, proj_out),
            nn.BatchNorm1d(proj_out)
        )

        self.prediction = nn.Sequential(
            nn.Linear(proj_out, pred_hid),
            nn.BatchNorm1d(pred_hid),
            nn.ReLU(),
            nn.Linear(pred_hid, pred_out),
        )

        if args.similarity =='cosine':
            self.d = D_Cos()
        elif args.similarity =='mean_square_error':
            self.d = D_MSE()
        elif args.similarity =='cross_entropy': 
            self.d = D_CE()
    
        if not args.supervised:
            if args.checkpoints is not None and downstream:
                print('Model Loaded')
                self.load_state_dict(torch.load(args.checkpoints)['model_state_dict'])
        else:
            print('Supervised Training')

    def forward(self, x1, x2):
        out1 = self.backbone(x1).squeeze()
        z1 = self.projection(out1)
        p1 = self.prediction(z1)

        out2 = self.backbone(x2).squeeze()
        z2 = self.projection(out2)
        p2 = self.prediction(z2)

        d1 = self.d(p1, z2) / 2.
        d2 = self.d(p2, z1) / 2.

        return d1, d2, z1


class DownStreamModel(nn.Module):
    def __init__(self, args):
        super(DownStreamModel, self).__init__()
        self.simsiam = Model(args, downstream=True)
        hidden = 512

        self.net_backbone = nn.Sequential(
            self.simsiam.backbone,
        )

        for name, param in self.net_backbone.named_parameters():
            """Returns an iterator over module parameters and names, yielding both the
            name of the parameter as well as the parameter itself."""
            if args.supervised:
                param.requires_grad = True
            else:    
                param.requires_grad = False

        self.net_projection = nn.Sequential(
            self.simsiam.projection,
        )

        for name, param in self.net_projection.named_parameters():
            if args.supervised:
                param.requires_grad = True
            else:    
                param.requires_grad = False

        self.out = nn.Sequential(
            nn.Linear(args.proj_out, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, args.n_classes),
        )

    def forward(self, x):
        out = self.net_backbone(x).squeeze()
        proj = self.net_projection(out)
        out = self.out(proj)

        return out
