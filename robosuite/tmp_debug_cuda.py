import os, sys, torch

def p(tag):
    print(tag, 'CUDA_VISIBLE_DEVICES=', repr(os.environ.get('CUDA_VISIBLE_DEVICES')), 'cuda?', torch.cuda.is_available(), torch.cuda.device_count())

p('start')
import benchmark.core
p('after benchmark.core')
import robosuite.discriminator.dyn_disc.adapters.pu_bce
p('after adapters.pu_bce')
import robosuite.discriminator.dyn_disc.detectors.pu_bce
p('after detectors.pu_bce')
import robosuite.discriminator.dyn_disc.visualization.visualize_pu_bce as v
p('after viz module')
args=v._parse_args()
print('args', args.device)
p('before require')
try:
    d=v._require_cuda(args.device)
    print('device', d)
except Exception as e:
    print('err', e)
