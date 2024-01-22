import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp
from torchvision.datasets import CIFAR10
from torchvision import transforms
import numpy as np 
import time
  
class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x, out_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if out_feature == False:
            return out
        else:
            return out,feature
 
 
def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)
def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)



def measure_inference_time(net, input, repeat=100):
   # torch.cuda.synchronize()   # if use cuda uncomment it
    start = time.perf_counter()
    for _ in range(repeat):
        model(input)
        #torch.cuda.synchronize() # if use cuda uncomment it
    end = time.perf_counter()
    return (end-start) / repeat

def prune_model(model, round_to=1):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    def prune_conv(conv, amount=0.2, round_to=1):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount, round_to=round_to)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()
    
    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    for m in model.modules():
        if isinstance( m, BasicBlock ):
            prune_conv( m.conv1, block_prune_probs[blk_id], round_to )
            prune_conv( m.conv2, block_prune_probs[blk_id], round_to )
            blk_id+=1
    return model 
 
device = torch.device('cpu')  #torch.device('cuda') # or torch.device('cpu')
repeat = 100

# before pruning
model = ResNet18().eval()
fake_input = torch.randn(16,3,32,32)
model = model.to(device)
fake_input = fake_input.to(device)
inference_time_before_pruning = measure_inference_time(model, fake_input, repeat)
print("before pruning: inference time=%f s, parameters=%d"%(inference_time_before_pruning, tp.utils.count_params(model)))

# w/o rounding
model = ResNet18().eval()
prune_model(model)
print(model)
model = model.to(device)
fake_input = fake_input.to(device)
inference_time_without_rounding = measure_inference_time(model, fake_input, repeat)
print("w/o rounding: inference time=%f s, parameters=%d"%(inference_time_without_rounding, tp.utils.count_params(model)))
    
# w/ rounding
model = ResNet18().eval()
prune_model(model, round_to=16)
print(model)
model = model.to(device)
fake_input = fake_input.to(device)
inference_time_with_rounding = measure_inference_time(model, fake_input, repeat)
print("w/ rounding: inference time=%f s, parameters=%d"%(inference_time_with_rounding, tp.utils.count_params(model)))
