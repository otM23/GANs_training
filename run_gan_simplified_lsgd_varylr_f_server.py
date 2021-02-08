# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 19:00:50 2021

@author: othmane.mounjid
"""



# import libraries
import os
import argparse
from importlib import import_module
import shutil
import json
import numpy as np


import torch
from torch.autograd import Variable
from optim.lsgd_improve import LSGD_improve
from utils.datasets_utils.data_gen_class import CUSTOM_BINARY
from utils.logger import Logger


DATASET_CUSTOMBINARY = 'CUSTOM_BINARY'

# losses
adversarial_loss = torch.nn.CrossEntropyLoss()

# get the folder path
real_path  = os.path.realpath(__file__) # to remove

################################################################################
################################################################################
########################## Training ############################################
################################################################################
################################################################################

    
def train(discriminator, generator, device, train_loader, optimizer, optimizerg, \
          scheduler, epoch, args, logger, use_cuda = False):

    # define variable type
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    
    # scheduler type
    def scheduler_type(_scheduler):
        if _scheduler is None:
            return 'none'
        return getattr(_scheduler, 'scheduler_type', 'epoch')

    discriminator.train()
    generator.train() # check !!
    gen_in_numel = generator.in_numel

    total_correct = 0
    d_loss, g_loss = None, None
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch
          
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size = data.shape[0]
        data, target = data.to(device), target.to(device)

        # Adversarial ground truths
        valid = Variable(LongTensor(batch_size).fill_(1.0), requires_grad=False)
        fake = Variable(LongTensor(batch_size).fill_(0.0), requires_grad=False)


        # Sample noise and labels as generator input
        noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, gen_in_numel))))     

        # Generate a batch of images   
        gen_imgs = generator(noise)

        
        # -----------------
        #  Train Generator
        # -----------------

        def closureg():
            optimizerg.zero_grad()
            validity = discriminator(gen_imgs) # the lenet model does not give validity
            loss = adversarial_loss(validity, valid) 
            loss.backward(create_graph=args.create_graph, retain_graph=True)
            
            return loss, validity          
        
        g_loss, validity = optimizerg.step(closure=closureg) 
     
        
        # -----------------
        #  Train discriminator
        # -----------------

        for name, param in discriminator.named_parameters(): ## copy old parameters value
            attr = 'p_pre_{}'.format(name)
            setattr(discriminator, attr, param.detach().clone())

        # data concatenation
        data_all = torch.cat((data.float(), gen_imgs.detach()), 0)
        valid_all = torch.cat((valid, fake), 0)
        
        # update params
        def closured():
            optimizer.zero_grad()
            validity = discriminator(data_all) # the lenet model does not give validity
            d_loss = adversarial_loss(validity, valid_all) 
            d_loss.backward(create_graph=args.create_graph)
            
            return d_loss, validity  
            

        d_loss, validity = optimizer.step(closure=closured)
        
        # Calculate discriminator accuracy
        correct = np.sum(np.argmax(validity.cpu().detach().numpy(), axis=1) == valid_all.cpu().detach().numpy())
        d_acc = np.mean(np.argmax(validity.cpu().detach().numpy(), axis=1) == valid_all.cpu().detach().numpy())

        
        # -----------------
        #  summary of the results
        # -----------------
        
        total_correct += correct       
        iteration = base_num_iter + batch_idx + 1
        total_data_size += len(data_all) # len(target_all)

        if scheduler_type(scheduler) == 'iter':
            scheduler.step()

        if batch_idx % args.log_interval == 0:
            d_acc = 100. * total_correct / total_data_size
            elapsed_time = logger.elapsed_time
            print('Train Epoch: {} [{}/{} ({:.0f}%)], [Batch {}/{}], d_Loss: {:.6f}, '
                  'd_Accuracy: {:.0f}/{} ({:.2f}%), '
                  'g_Loss: {:.6f}, '
                  'Elapsed Time: {:.1f}s'.format(
                epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch, 
                batch_idx, len(train_loader),d_loss, total_correct, total_data_size, d_acc, g_loss,
                elapsed_time))
          
            # save log
            lr = optimizer.param_groups[0]['lr']
            log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                   'accuracy': d_acc, 'd_loss': d_loss.cpu().detach().numpy().item(), 'g_loss':g_loss.cpu().detach().numpy().item(), 'lr': lr}

            for name, param in discriminator.named_parameters():
                attr = 'p_pre_{}'.format(name)
                p_pre = getattr(discriminator, attr)
                p_norm = param.norm().item()
                p_shape = list(param.size())
                p_pre_norm = p_pre.norm().item()
                g_norm = param.grad.norm().item()
                upd_norm = param.sub(p_pre).norm().item()
                noise_scale = getattr(param, 'noise_scale', 0)

                p_log = {'p_shape': p_shape, 'p_norm': p_norm, 'p_pre_norm': p_pre_norm,
                         'g_norm': g_norm, 'upd_norm': upd_norm, 'noise_scale': noise_scale}
                log[name] = p_log

            logger.write(log)

    if scheduler_type(scheduler) == 'epoch':
        scheduler.step(epoch - 1)

    d_acc = 100. * total_correct / total_data_size

    return d_acc, d_loss.cpu().detach().numpy().item(), g_loss.cpu().detach().numpy().item(), total_correct, total_data_size


################################################################################
################################################################################
########################## Validation ##########################################
################################################################################
################################################################################


def validate(discriminator, generator, device, val_loader, optimizer, use_cuda = False):
    
    # define variable type
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor

    
    discriminator.eval()
    generator.eval()
    gen_in_numel = generator.in_numel
    
    val_loss = 0
    val_g_loss = 0
    total_correct = 0
    total_data = 0

    with torch.no_grad():
        for data, target in val_loader:
            batch_size = data.shape[0]
            data, target = data.to(device), target.to(device)

            # Adversarial ground truths
            valid = Variable(LongTensor(batch_size).fill_(1.0), requires_grad=False)
            fake = Variable(LongTensor(batch_size).fill_(0.0), requires_grad=False)


            # Sample noise and labels as generator input
            noise = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, gen_in_numel))))
        
            # Generate a batch of images
            gen_imgs = generator(noise)

            # generator loss 
            validity = discriminator(gen_imgs) # the lenet model does not give validity
            g_loss = adversarial_loss(validity, valid)

            # discriminator loss
            data_all = torch.cat((data.float(), gen_imgs.detach()), 0)
            valid_all = torch.cat((valid, fake), 0)
            real_pred = discriminator(data_all)
            d_loss = adversarial_loss(real_pred, valid_all)  
    
            # Calculate discriminator number of correct guess
            correct = np.sum(np.argmax(real_pred.cpu().detach().numpy(), axis=1) == valid_all.cpu().detach().numpy())
            
            
            # summarize the results
            val_loss += d_loss
            val_g_loss += g_loss
            total_correct += correct
            total_data += valid_all.shape[0]
        
    val_loss /= total_data
    val_g_loss /= total_data
    val_accuracy = 100. * total_correct / total_data

    print('\nEval: Average loss: {:.4f}, Accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
        val_loss, total_correct, total_data, val_accuracy))


    return val_accuracy, val_loss.cpu().detach().numpy().item(), correct, val_g_loss.cpu().detach().numpy().item()


################################################################################
################################################################################
########################## Main function #######################################
################################################################################
################################################################################



def main(config_optim = 'configs/cifar10/lenet_adam.json'):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_CUSTOMBINARY], default=DATASET_CUSTOMBINARY,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
    parser.add_argument('--random_crop', action='store_true',
                        help='[data augmentation] random crop')
    parser.add_argument('--random_horizontal_flip', action='store_true',
                        help='[data augmentation] random horizontal flip')
    # Training Settings
    parser.add_argument('--arch_file_discriminator', type=str, default='models/DiscriminatorBase.py',
                        help='name of file which defines the discriminator architecture')
    parser.add_argument('--arch_name_discriminator', type=str, default='Discriminator_simple',
                        help='name of the discriminator architecture')
    parser.add_argument('--arch_file_generator', type=str, default='models/GeneratorLinear.py',
                        help='name of file which defines the generator architecture')
    parser.add_argument('--arch_name_generator', type=str, default='Generator_simple',
                        help='name of the generator architecture')
    parser.add_argument('--arch_args_discriminator', type=json.loads, default=None,
                        help='[JSON] arguments for the discriminator architecture')
    parser.add_argument('--arch_args_generator', type=json.loads, default=None,
                        help='[JSON] arguments for the generator architecture')
    parser.add_argument('--optim_name', type=str, default="SGD",
                        help='name of the optimizer')
    parser.add_argument('--optim_args_discriminator', type=json.loads, default=None,
                        help='[JSON] arguments for the discriminator optimizer')
    parser.add_argument('--optim_args_generator', type=json.loads, default=None,
                        help='[JSON] arguments for the generator optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the curvature')
    parser.add_argument('--fisher_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the fisher')
    parser.add_argument('--scheduler_name', type=str, default=None,
                        help='name of the learning rate scheduler')
    parser.add_argument('--scheduler_args', type=json.loads, default=None,
                        help='[JSON] arguments for the scheduler')
    # Options
    parser.add_argument('--download', action='store_true', default=False,
                        help='if True, downloads the dataset (CIFAR-10 or 100) from the internet')
    parser.add_argument('--create_graph', action='store_true', default=False,
                        help='create graph of the derivative')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path for resume training')
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save output files')
    parser.add_argument('--config', default=config_optim,
                        help='config file path')

    args = parser.parse_args()
    dict_args = vars(args)

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)
        
        
    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup data loader
    if args.dataset == DATASET_CUSTOMBINARY:
        # CUSTOM_BINARY
        num_classes = 2
        dataset_class = CUSTOM_BINARY
    else:
        assert False, f'unknown dataset {args.dataset}'
        
    
    train_dataset = dataset_class(root=args.root, train=True)
    val_dataset = dataset_class(root=args.root, train=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers)


    # Setup models: discriminator and generator
    ## Discriminator
    if args.arch_file_discriminator is None:
        assert False, f'unknown discriminator architecture'
    else:
        _, ext = os.path.splitext(args.arch_file_discriminator)
        dirname = os.path.dirname(args.arch_file_discriminator)

        if dirname == '':
            module_path = args.arch_file_discriminator.replace(ext, '')
        elif dirname == '.':
            module_path = os.path.basename(args.arch_file_discriminator).replace(ext, '')
        else:
            module_path = '.'.join(os.path.split(args.arch_file_discriminator)).replace(ext, '')

        module = import_module(module_path)
        arch_class_discriminator = getattr(module, args.arch_name_discriminator)
        
    arch_kwargs = {} if args.arch_args_discriminator is None else args.arch_args_discriminator
    arch_kwargs['num_classes'] = num_classes

    discriminator = arch_class_discriminator(**arch_kwargs)
    setattr(discriminator, 'num_classes', num_classes)
    discriminator = discriminator.to(device)

    ## Generator
    if args.arch_file_generator is None:
        assert False, f'unknown generator architecture'
    else:
        _, ext = os.path.splitext(args.arch_file_generator)
        dirname = os.path.dirname(args.arch_file_generator)

        if dirname == '':
            module_path = args.arch_file_generator.replace(ext, '')
        elif dirname == '.':
            module_path = os.path.basename(args.arch_file_generator).replace(ext, '')
        else:
            module_path = '.'.join(os.path.split(args.arch_file_generator)).replace(ext, '')

        module = import_module(module_path)
        arch_class_generator = getattr(module, args.arch_name_generator)

    arch_kwargs = {} if args.arch_args_generator is None else args.arch_args_generator
    generator = arch_class_generator(**arch_kwargs)
    generator = generator.to(device)
    
    
    # Setup optimizer
    optim_kwargs = {} if args.optim_args_discriminator is None else args.optim_args_discriminator
    optimg_kwargs = {} if args.optim_args_generator is None else args.optim_args_generator

    if args.optim_name == 'LSGD_improve':
        optim_kwargs['model'] = discriminator
        optim_kwargs['histogram'] = False
        optimg_kwargs['model'] = generator
        optimg_kwargs['histogram'] = False
        optimizer = LSGD_improve(discriminator.parameters(), **optim_kwargs) 
        optimizerg = LSGD_improve(generator.parameters(), **optimg_kwargs)
    else:
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 10)
        optim_class = getattr(torch.optim, args.optim_name)
        optimizer = optim_class(discriminator.parameters(), **optim_kwargs)
        optimizerg = optim_class(generator.parameters(), **optimg_kwargs)

    # Setup lr scheduler
    scheduler = None
        
    # initialise epoch
    start_epoch = 1

    # All config
    print('===========================')
    for key, val in vars(args).items():
        if key == 'dataset':
            print('{}: {}'.format(key, val))
            print('train data size: {}'.format(len(train_loader.dataset)))
            print('val data size: {}'.format(len(val_loader.dataset)))
        else:
            print('{}: {}'.format(key, val))
    print('===========================')

    # Copy this file & config to args.out
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
#    shutil.copy(os.path.realpath(__file__), args.out)
    shutil.copy(real_path, args.out)

    if args.config is not None:
        shutil.copy(args.config, args.out)
    if args.arch_file_discriminator is not None:
        shutil.copy(args.arch_file_discriminator, args.out)
    if args.arch_file_generator is not None:
        shutil.copy(args.arch_file_generator, args.out)
            

    # Setup logger
    logger = Logger(args.out, args.log_file_name)
    logger.start()

    # Run training
    res = {'val_accuracy_test': [],
           'val_accuracy_train':[],
           'val_d_loss_test': [],
           'val_g_loss_test': [],
           'd_correct_test':[],
           'd_correct_train':[],
           'len_dataset_train': []}

    for epoch in range(start_epoch, args.epochs + 1): # epoch = 1

        # train
        val_accuracy_train, val_d_loss_train, val_g_loss_train, total_correct_train, total_data_size_train = train(discriminator, generator, device, train_loader, optimizer, optimizerg, scheduler, epoch, args, logger, use_cuda = use_cuda)

        # val
        val_accuracy_test, val_d_loss_test, d_correct_test, val_g_loss_test = validate(discriminator, generator, device, val_loader, optimizer, use_cuda = use_cuda) ## the variables of interest are val_accuracy, val_loss 
        

        # save log
        iteration = epoch * len(train_loader)
        log = {'epoch': epoch, 'iteration': iteration,
               'val_accuracy_train': val_accuracy_train, 'loss': val_d_loss_train,
               'val_accuracy_test': val_accuracy_test, 'val_d_loss_test': val_d_loss_test,
               'lr': optimizer.param_groups[0]['lr'],
               'momentum': optimizer.param_groups[0].get('momentum', 0)}
        logger.write(log)

        # save checkpoint
        if epoch % args.checkpoint_interval == 0 or epoch == args.epochs:
            path = os.path.join(args.out, 'epoch{}.ckpt'.format(epoch))
            data = {
                'discriminator': discriminator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'generator': generator.state_dict(),
                'optimizerg': optimizerg.state_dict(),                
                'epoch': epoch
            }
            torch.save(data, path)

        # update res
        res['val_accuracy_test'].append(val_accuracy_test)
        res['val_accuracy_train'].append(val_accuracy_train)
        res['val_d_loss_test'].append(val_d_loss_test)
        res['val_g_loss_test'].append(val_g_loss_test)
        res['d_correct_test'].append(d_correct_test)
        res['d_correct_train'].append(total_correct_train)
        res['len_dataset_train'].append(total_data_size_train)
        
        
    return res, discriminator, generator

if __name__ == '__main__':

    config_files = [
                [1e-2, 'configs/Custom_binary/lenet_lsgd_lr1_reduced_CustomData.json'],
                [5e-3, 'configs/Custom_binary/lenet_lsgd_lr2_reduced_CustomData.json'],
                [1e-3, 'configs/Custom_binary/lenet_lsgd_lr3_reduced_CustomData.json'],
                [5e-4, 'configs/Custom_binary/lenet_lsgd_lr4_reduced_CustomData.json'],
                [1e-4, 'configs/Custom_binary/lenet_lsgd_lr5_reduced_CustomData.json']
                ]
    
    for lr, config_optim  in config_files:
        df_sgd_reduced_gan, discriminator, generator = main(config_optim = config_optim)  
      
        
        # save results
        import pandas as pd
        if not os.path.isdir('result'):
            os.makedirs('result')
            
        pd.DataFrame(df_sgd_reduced_gan).to_csv('result/sgd_reduced_gan_50_rungansgdsimplified_lr_' + str(lr) + '.csv')