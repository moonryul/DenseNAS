import argparse
import ast
import importlib
import logging
import os
import pprint
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter

from configs.search_config import search_cfg
from configs.imagenet_train_cfg import cfg
from dataset import imagenet_data
from models import model_derived
from tools import utils
from tools.config_yaml import merge_cfg_from_file, update_cfg_from_cfg
from tools.lr_scheduler import get_lr_scheduler
from tools.multadds_count import comp_multadds

from .optimizer import Optimizer
from .trainer import SearchTrainer


#### for DistributedDataParallel

# Copyright (c) Facebook, Inc. and its affiliates.
# copied from detectron2/detectron2/engine/launch.py
# https://github.com/facebookresearch/detectron2/blob/9246ebc3af1c023cfbdae77e5d976edbcf9a2933/detectron2/engine/launch.py
import logging
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from detectron2.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)

#Added by Moon Jung
from torch.nn.parallel import DistributedDataParallel as DDP


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                args,
                timeout,
            ),
            daemon=False,
        )
    else:
        #main_func(*args)

        main_func(args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    #main_func(*args)
    main_func(args)
    
def main_func( args ):   
        
    update_cfg_from_cfg(search_cfg, cfg)
    if args.config is not None:
        merge_cfg_from_file(args.config, cfg)
    config = cfg

    if args.job_name != '':
        args.job_name = time.strftime("%Y%m%d-%H%M%S-") + args.job_name
        args.save = os.path.join(args.save, args.job_name)
        utils.create_exp_dir(args.save)
        os.system('cp -r ./* '+args.save)
        args.save = os.path.join(args.save, 'output')
        utils.create_exp_dir(args.save)
    else:
        args.save = os.path.join(args.save, 'output')
        utils.create_exp_dir(args.save)

    if args.tb_path == '':
        args.tb_path = args.save

    log_format = '%(asctime)s %(message)s'
    date_format = '%m/%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt=date_format)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format, date_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('No gpu device available')
        sys.exit(1)
    cudnn.benchmark = True
    cudnn.enabled = True

    if config.train_params.use_seed:    
        np.random.seed(config.train_params.seed)
        torch.manual_seed(config.train_params.seed)
        torch.cuda.manual_seed(config.train_params.seed)

    logging.info("args = %s", args)
    logging.info('Training with config:')
    logging.info(pprint.pformat(config))
    job_name = os.popen('cd %s && pwd -P && cd -' % args.save).readline().split('/')[-2]

    writer = SummaryWriter(args.tb_path)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    SearchSpace = importlib.import_module('models.search_space_'+config.net_type).Network
    ArchGenerater = importlib.import_module('.derive_arch_'+config.net_type, __package__).ArchGenerate
    derivedNetwork = getattr(model_derived, '%s_Net' % config.net_type.upper())

    super_model = SearchSpace(config.optim.init_dim, config.data.dataset, config)
    arch_gener = ArchGenerater(super_model, config)
    der_Net = lambda net_config: derivedNetwork(net_config, 
                                                config=config)
    
    # Changed by Moon Jung
    #super_model = nn.DataParallel(super_model)
    cur_rank = comm.get_local_rank()
    
    super_model = DDP( super_model.to( cur_rank ), device_ids=[cur_rank], broadcast_buffers = False)
    
    # whether to resume from a checkpoint
    if config.optim.if_resume:
        utils.load_model(super_model, config.optim.resume.load_path)
        start_epoch = config.optim.resume.load_epoch + 1
    else:
        start_epoch = 0
        
    # Changed by Moon Jung
    #super_model = super_model.cuda()

    if config.optim.sub_obj.type=='flops':
        flops_list, total_flops = super_model.module.get_cost_list(
                                config.data.input_size, cost_type='flops')
        super_model.module.sub_obj_list = flops_list
        logging.info("Super Network flops (M) list: \n")
        logging.info(str(flops_list))
        logging.info("Total flops: " + str(total_flops))
    elif config.optim.sub_obj.type=='latency':
        with open(os.path.join('latency_list', config.optim.sub_obj.latency_list_path), 'r') as f:
            latency_list = eval(f.readline())
        super_model.module.sub_obj_list = latency_list
        logging.info("Super Network latency (ms) list: \n")
        logging.info(str(latency_list))
    else:
        raise NotImplementedError
    logging.info("Num Params = %.2fMB", utils.count_parameters_in_MB(super_model))

    if config.data.dataset == 'imagenet':
        imagenet = imagenet_data.ImageNet12(trainFolder=os.path.join(args.data_path, 'train'), 
                                            testFolder=os.path.join(args.data_path, 'val'),
                                            num_workers=config.data.num_workers,
                                            type_of_data_augmentation=config.data.type_of_data_aug,
                                            data_config=config.data)
        train_queue, valid_queue = imagenet.getTrainTestLoader(config.data.batch_size, 
                                                                train_shuffle=True,
                                                                val_shuffle=True)
    else:
        raise NotImplementedError

    search_optim = Optimizer(super_model, criterion, config)
    
    scheduler = get_lr_scheduler(config, search_optim.weight_optimizer, imagenet.train_num_examples)
    scheduler.last_step = start_epoch * (imagenet.train_num_examples // config.data.batch_size + 1)

    search_trainer = SearchTrainer(train_queue, valid_queue, search_optim, criterion, scheduler, config, args)

    betas, head_alphas, stack_alphas = super_model.module.display_arch_params()    
    derived_archs = arch_gener.derive_archs(betas, head_alphas, stack_alphas)
    derived_model = der_Net('|'.join(map(str, derived_archs)))
    logging.info("Derived Model Mult-Adds = %.2fMB" % comp_multadds(derived_model, 
                                                    input_size=config.data.input_size))
    logging.info("Derived Model Num Params = %.2fMB", utils.count_parameters_in_MB(derived_model))

    best_epoch = [0, 0, 0] # [epoch, acc_top1, acc_top5]
    rec_list = []
    for epoch in range(start_epoch, config.train_params.epochs):
        # training part1: update the architecture parameters
        if epoch >= config.search_params.arch_update_epoch:
            search_stage = 1
            search_optim.set_param_grad_state('Arch')
            train_acc_top1, train_acc_top5, train_obj, sub_obj, batch_time = search_trainer.train(
                        super_model, epoch, 'Arch', search_stage)
            logging.info('EPOCH%d Arch Train_acc  top1 %.2f top5 %.2f loss %.4f %s %.2f batch_time %.3f', 
                epoch, train_acc_top1, train_acc_top5, train_obj, config.optim.sub_obj.type, sub_obj, batch_time)
            writer.add_scalar('arch_train_acc_top1', train_acc_top1, epoch)
            writer.add_scalar('arch_train_loss', train_obj, epoch)
        else:
            search_stage = 0

        # training part2: update the operator parameters
        search_optim.set_param_grad_state('Weights')
        train_acc_top1, train_acc_top5, train_obj, sub_obj, batch_time = search_trainer.train(
                        super_model, epoch, 'Weights', search_stage)
        logging.info('EPOCH%d Weights Train_acc  top1 %.2f top5 %.2f loss %.4f %s %.2f  | batch_time %.3f', 
                epoch, train_acc_top1, train_acc_top5, train_obj, config.optim.sub_obj.type, sub_obj, batch_time)
        writer.add_scalar('weight_train_acc_top1', train_acc_top1, epoch)
        writer.add_scalar('weight_train_loss', train_obj, epoch)

        # validation
        if epoch >= config.search_params.val_start_epoch:
            with torch.no_grad():
                val_acc_top1, val_acc_top5, valid_obj, sub_obj, batch_time = search_trainer.infer(super_model, epoch)
            logging.info('EPOCH%d Valid_acc  top1 %.2f top5 %.2f %s %.2f batch_time %.3f', 
                        epoch, val_acc_top1, val_acc_top5, config.optim.sub_obj.type, sub_obj, batch_time)
            writer.add_scalar('arch_val_acc', val_acc_top1, epoch)
            writer.add_scalar('arch_whole_{}'.format(config.optim.sub_obj.type), sub_obj, epoch)

            if val_acc_top1 > best_epoch[1]:
                best_epoch = [epoch, val_acc_top1, val_acc_top5]
                utils.save(super_model, os.path.join(args.save, 'weights_best.pt'))
            logging.info('BEST EPOCH %d  val_top1 %.2f val_top5 %.2f', best_epoch[0], best_epoch[1], best_epoch[2])
        else:
            utils.save(super_model, os.path.join(args.save, 'weights_best.pt'))
        
        betas, head_alphas, stack_alphas = super_model.module.display_arch_params()
        derived_arch = arch_gener.derive_archs(betas, head_alphas, stack_alphas)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = der_Net(derived_arch_str)
        derived_flops = comp_multadds(derived_model, input_size=config.data.input_size)
        derived_params = utils.count_parameters_in_MB(derived_model)
        logging.info("Derived Model Mult-Adds = %.2fMB" % derived_flops)
        logging.info("Derived Model Num Params = %.2fMB" % derived_params)
        writer.add_scalar('derived_flops', derived_flops, epoch)

        if (epoch+1)==config.search_params.arch_update_epoch:
            utils.save(super_model, os.path.join(args.save, 'weights_{}.pt'.format(epoch)))

        if epoch >= config.search_params.val_start_epoch:
            epoch_rec = {'top1_acc': val_acc_top1,
                        'epoch': epoch,
                        'multadds': derived_flops,
                        'params': derived_params,
                        'arch': derived_arch_str}
            if_update = utils.record_topk(2, rec_list, epoch_rec, 'top1_acc', 'arch')
            if if_update:
                with open(os.path.join(args.save, 'top_results'), 'w') as f:
                    f.write(str(rec_list) + '\n')
                    f.write(job_name)
                with open(os.path.join(args.save, 'excel_record'), 'w') as f:
                    for record in rec_list:
                        f.write(',,,{:.2f}MB,{:.2f}MB,,,,{},{}\n'.format(
                                record['multadds'], record['params'],
                                job_name, record['epoch']))
                        f.write(record['arch']+'\n')

    logging.info('\nTop2 arch records for Excel: ')
    for record in rec_list:
        logging.info('\n,,,{:.2f}MB,{:.2f}MB,,,,{},{}'.format(
                record['multadds'], record['params'], job_name, record['epoch']))
        logging.info('\n'+record['arch'])
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser("Search_Configs")
    parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
    parser.add_argument('--data_path', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--save', type=str, default='../', help='experiment name')
    parser.add_argument('--tb_path', type=str, default='', help='tensorboard output path')
    parser.add_argument('--job_name', type=str, default='', help='job_name')
    parser.add_argument('-c', '--config', metavar='C', default=None, help='The Configuration file')

    args = parser.parse_args()
    
    # Added by Moon Jung
    
    #def launch(
    # main_func,
    # num_gpus_per_machine,
    # num_machines=1,
    # machine_rank=0,
    # dist_url=None,
    # args=(),
    # timeout=DEFAULT_TIMEOUT,
    #):        
        
    launch( main_func,
            num_gpus_per_machine=4,
            num_machines=1,
            machine_rank=0,
            dist_url=‘auto’,
            args = args
          )
    
