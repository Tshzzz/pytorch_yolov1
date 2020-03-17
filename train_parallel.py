from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data.distributed
import torch.distributed as dist

import numpy as np
import random
from datetime import datetime
from data.build import make_dist_voc_loader,make_mutilscale_voc_loader

from yolo import create_yolov1

torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)
np.random.seed(666)
random.seed(666)
torch.set_printoptions(precision=10)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

from tensorboardX import SummaryWriter
from metric import training_eval

import shutil
import tqdm

from utils import *

try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

import argparse
def parse():
    parser = argparse.ArgumentParser(description='PyTorch Detector Training')
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--training_cfg", type=str,default=None)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--opt-level', type=str,default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--sync_bn', action='store_true',help='enabling apex sync BN.')
    args = parser.parse_args()
    return args


args = parse()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt



def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = args.world_size
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_training():
    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))
    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    args.world_size = 1
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend='nccl',init_method='env://')
    args.world_size = torch.distributed.get_world_size()

class train_engine(object):
    def __init__(self,epochs,
                 model,optim,
                 trnload,valload,train_img_size,
                 save_dir,
                 logger,classes,
                 resume,
                 save_iter=3
                 ):
        self.epochs = epochs
        self.optim = optim
        self.Model = model
        self.trainload = trnload
        self.valload = valload
        self.device = 'cuda'
        self.save_dir = save_dir
        self.logger = logger
        self.save_iter = save_iter-1
        self.classes = classes
        self.resume = resume
        self.start_epoch = 0
        self.train_img_size = train_img_size


    def setup(self,train_mile,gamma):
        print('setup training')

        self.Model, self.optim = amp.initialize(self.Model, self.optim,
                                        opt_level=args.opt_level,
                                        keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                        loss_scale=args.loss_scale
                                        )

        self.scheduler = lr_scheduler.MultiStepLR(self.optim, train_mile, gamma=gamma)
        #self.scheduler = WarmUpLR(self.scheduler,3)
        self.Model = DDP(self.Model, delay_allreduce=True)

        if self.resume:
            checkpoint = torch.load('{}/model_checkpoint.pth'.format(self.save_dir))
            self.Model.load_state_dict(checkpoint['model'], strict=True)


            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            amp.load_state_dict(checkpoint['amp'])
            self.start_epoch = checkpoint['epoch']
            print('resume from epoch {}'.format(self.start_epoch))
            self.start_epoch += 1



    def train(self):
        #TODO full speed validation
        scheduler = self.scheduler
        best_mean_val = 0
        train_epochs = tqdm.tqdm(range(self.start_epoch,self.epochs),ncols=40)
        for epoch in train_epochs:
            losses = dict()
            for idx,(img,gt_info,img_info) in enumerate(self.trainload):
                img = img.to(self.device)
                loss_dict = self.Model(img,gt_info)

                loss = sum(l for l in loss_dict.values())
                self.optim.zero_grad()

                with amp.scale_loss(loss, self.optim) as scaled_loss:
                    scaled_loss.backward()

                self.optim.step()

                for k in loss_dict.keys():
                    if k in losses.keys():
                        reduced_loss = reduce_tensor(loss_dict[k].data)
                        losses[k].update(to_python_float(reduced_loss), img.size(0))
                    else:
                        losses[k] = AverageMeter()
                        reduced_loss = reduce_tensor(loss_dict[k].data)
                        losses[k].update(to_python_float(reduced_loss), img.size(0))

                torch.cuda.synchronize()
                if args.local_rank == 0:
                    status = '[{}] lr={:.5f} '.format(
                        epoch + 1, scheduler.get_lr()[0])

                    for k,meter in losses.items():
                        status += 'bs_{}={:.3f} ep_{}={:.3f} '.format(
                            k,meter.val, k,meter.avg)

                    train_epochs.set_description(status)

            torch.cuda.synchronize()

            if args.local_rank == 0:
                scheduler.step(epoch)
                self.logger.add_scalar('lr: ', scheduler.get_lr()[0], epoch)
                checkpoint = {
                    'model': self.Model.state_dict(),
                    'optimizer': self.optim.state_dict(),
                    'amp': amp.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'epoch': epoch
                }

                for tag, value in losses.items():
                    self.logger.add_scalar(tag, value.avg, epoch)
                torch.save(checkpoint, '{}/model_checkpoint.pth'.format(self.save_dir))

            if args.local_rank == 0 and epoch % self.save_iter == 0 and self.valload is not None:

                self.Model.eval()
                result = training_eval(self.Model, self.valload, self.classes, self.device)
                self.Model.train()

                mean_val = result['map']
                del result['map']
                for tag, value in result.items():
                    self.logger.add_scalar(tag, value, epoch)

                if best_mean_val < mean_val:
                    best_mean_val = mean_val
                    best_model = {
                        'model': self.Model.state_dict(),
                        'epoch': epoch
                    }
                    torch.save(
                        best_model, '{}/best_model.pth'.format(self.save_dir)
                    )
                self.logger.add_scalar('mAp', mean_val, epoch)
                self.logger.add_scalar('bestmAp', best_mean_val, epoch)
        print('copy best model to logger dir')
        print(best_mean_val)
        if args.local_rank == 0:
            try:
                model_copy_path = self.logger.logdir
            except:
                model_copy_path = self.logger.log_dir
            shutil.copy(os.path.join(self.save_dir,'best_model.pth'),model_copy_path)

def train_voc_demo(cfg):
    setup_training()
    train_cfg = cfg['train_cfg']
    model_cfg = cfg['model_cfg']
    model_name = model_cfg['model_type']
    epochs = train_cfg['epochs']
    classes = train_cfg['classes']
    lr = train_cfg['lr']
    bs = train_cfg['batch_size']
    device = train_cfg['device']
    out_dir = train_cfg['out_dir']
    resume = train_cfg['resume']
    use_sgd = train_cfg['use_sgd']
    scale = train_cfg['scale']
    mile = train_cfg['milestone']
    gamma = train_cfg['gamma']
    train_root = train_cfg['dataroot']
    patch_size = train_cfg['patch_size']


    out_dir = out_dir + '/' + model_name
    create_dir(out_dir)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(out_dir, current_time)
    create_dir(log_dir)
    logger = SummaryWriter(log_dir)
    ModelG = create_yolov1(model_cfg)

    if use_sgd:
        OptimG = optim.SGD(ModelG.parameters(), lr=lr[0], momentum=0.9, weight_decay=1e-6)
    else:
        OptimG = torch.optim.Adam(ModelG.parameters(),
                                        lr=lr[0],
                                        weight_decay=5e-5)

    if args.sync_bn:
        print("using apex synced BN")
        ModelG = apex.parallel.convert_syncbn_model(ModelG)

    ModelG = ModelG.to(device)

    trnloader = make_dist_voc_loader(os.path.join(train_root,'train.txt'),
                                 img_size=patch_size,
                                 batch_size=bs,
                                 train=True,
                                 rank=args.local_rank
                                 )
    valloader = make_mutilscale_voc_loader(os.path.join(train_root,'VOC2007_test.txt'),
                                 img_size=[(512,512)],
                                 batch_size=16,
                                 train=False
                                )


    engine = train_engine(epochs,ModelG,OptimG,trnloader,
                          valloader,patch_size,
                          out_dir,logger,classes,resume)
    engine.setup(mile,gamma)
    engine.train()




if __name__ == "__main__":
    from cfg.voc import cfg
    train_voc_demo(cfg)

