# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 18:09:00 2020

@author: othmane.mounjid
"""

import math
import copy
import torch
from torch.optim.optimizer import Optimizer, required
from functools import reduce

#import pdb # for debug only


class LSGD_improve(Optimizer):
    r"""Implements an improvement of Adam algorithm.

    The has been proposed in ` to fill `_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Add references 
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, clip_val = [10,1], max_memory = 20, iter_max = 20,
                 model = None, histogram = True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(LSGD_improve, self).__init__(params, defaults)
        
        # initialisation current state and previous state
        self.curr_state = dict()
        self.prev_state = dict()
        
        # initialisation clip values 
        self.clip_val = clip_val 
        
        # initialisation number of elements (parameters)
        self._numel_cache = None
        
        # initialisation maximal number of saved past gradients per optimization step
        self.max_memory = max_memory
        
        # initialisation maximal number of lbfgs iterations
        self.iter_max = iter_max

        # save params : used for the initialisation of LBFGS_sub; always none unless you know why you use it
        if model:
            self.model = model

            # initialisation of the LBFGS sub optimizer 
            self.sub_optimizer = LBFGS_sub(self.model.parameters(), lr = self.defaults['lr'], 
                                      max_memory = self.max_memory)
        
        # initialise past param groups before sub optimizer action
        self.past_param_groups = None 
        
        # store the control value and print them
        self.values = None
        self.valuesf = None
        self.denominator = None
        self.numerator = None

        # print frequency
        self.print_count = 0
        self.print_freq = 50
        self.histogram = histogram

    def __setstate__(self, state):
        super(LSGD_improve, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _numel(self):
        _params = self.param_groups[0]['params']
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), _params, 0)
        return self._numel_cache
    
    def init_optimizer(self):
        # check values
        self.values = []
        self.valuesf = []
        self.denominator = []
        self.numerator = []
        # position and gradient initialitsation
        positions = [] # store position values
        grads = [] # store gradient values
        for group in self.param_groups: ## to check later 
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                ############################################################
                # sub step : compute improvement adam
                ############################################################
                
                ## compute numerator
                numerator = torch.sum(d_p*d_p).item()
                
                ## compute denominator
#                barsigma_sub = (math.sqrt(group['lr']) * grad)
#                barsigma_sub = (grad)
#                barsigma_sub = (math.sqrt(group['lr'])*d_p)
                barsigma_sub = (d_p)
                Hessigma_sub = barsigma_sub
                denominator =  torch.sum(barsigma_sub * Hessigma_sub).item()

                # compute adjustment 
                if denominator > 0:
                    u = min((numerator/denominator),self.clip_val[0]) ## self.clip_val[0] = 10 by default 
                else:
                    u = self.clip_val[1] ## self.clip_val[1] = 1: the value 1 enables us to use standard adam             
                    
#                # check with standard value
#                u = 1
                    
                # check values 
                if denominator != 0:
                    self.values.append((numerator/denominator))
                self.valuesf.append((u))
                self.denominator.append(denominator)
                self.numerator.append(numerator)
                
                # update parameters
                p.data.add_(- u*group['lr'], d_p)
                
                # save position
                positions.append(p.data.view(-1))

                # save gradient
                grads.append(d_p.view(-1)) # we do not save the raw gradient
        
        # save values
        self.values = torch.tensor(self.values)
        self.valuesf = torch.tensor(self.valuesf)
        self.denominator = torch.tensor(self.denominator)
        self.numerator = torch.tensor(self.numerator)
        
        # unsqueeze to get the shape (vector_size,1)
        positions = torch.cat(positions, 0).unsqueeze(-1)
        grads = torch.cat(grads, 0).unsqueeze(-1)
              
        return [positions, grads]

    def get_state_optimizer(self):
        # position and gradient initialitsation
        positions = [] # store position values
        grads = [] # store gradient values
        barsigmas_next = [] # store next barsigma values
        updates = [] # store update values since we need to compute the missing adjustment
        for group in self.param_groups: ## to check later
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # save position
                positions.append(p.data.view(-1))

                # save gradient
                grads.append(d_p.view(-1)) # we do not save the raw gradient
                
                # save bar_sigma
#                bar_sigma_next = (math.sqrt(group['lr'])*grad)
#                bar_sigma_next = (grad)
                bar_sigma_next = (math.sqrt(group['lr'])*d_p)
#                bar_sigma_next = (0.5*d_p)
                barsigmas_next.append(bar_sigma_next.view(-1))
                
                # save updates
                update = -group['lr']*d_p
                updates.append(update.view(-1))
                
        # unsqueeze to get the shape (vector_size,1)
        positions = torch.cat(positions, 0).unsqueeze(-1)
        grads = torch.cat(grads, 0).unsqueeze(-1)
        barsigmas_next = torch.cat(barsigmas_next, 0).unsqueeze(-1)
        updates = torch.cat(updates, 0).unsqueeze(-1)
              
        return [positions, grads, barsigmas_next, updates]
    
#    self = optimizer
#    init_positions.shape
#    init_gradients.shape
#    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
#    params = sum([np.prod(p.size()) for p in model_parameters])
#    ## number of parameters is correct 
#    ## what are the heaviest operations
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # initialisation current state and previous state
        # NOTE: we use the state of the first param to store current state and previous state
        # because this helps with casting in load_state_dict
        param_state = self.param_groups[0]['params'][0]
        state = self.state[param_state]

        # initialisation for the first call of the function step        
        if not('optimizer_state_var' in state):
            # the function init_optimizer computes initial positions, gradients, and hessian estimate
            # plus updates the parameters
            init_positions, init_gradients = self.init_optimizer()
                                              
            # we need to reduce these values : to do later
            state['optimizer_state_var'] = {'curr_state' : { 'position': init_positions,
                                                             'grad': init_gradients,
                                                             'prev_position': init_positions,
                                                             'prev_grad': init_gradients,
                                                             'barsigma': torch.empty(init_positions.shape),
                                                             'update': torch.empty(init_positions.shape),
                                                             'grad_deltas':[],
                                                             'position_deltas':[],
                                                             'normalisation_factors':[],
                                                             'hessigma_estimate':torch.empty(init_positions.shape)
                                                             }}
    
            # initial values of current state and previous state
            self.curr_state = state['optimizer_state_var']['curr_state']
        # main routine:
        else:
            # compute initial positions, gradients, and hessian estimate
            curr_pos, curr_grad, curr_barsigma, curr_update = self.get_state_optimizer()
            
            # update cur_state and prev_state; the main update is the hessian 
            self.update_state(curr_pos, curr_grad, curr_barsigma, curr_update, self.max_memory, closure)
        
            # update values of the parameters
            hessigma_estimate = self.curr_state['hessigma_estimate']
            barsigma = self.curr_state['barsigma']
            update = self.curr_state['update']
            curr_grad = self.curr_state['grad'] 
                    
            self.update_parameters(hessigma_estimate, barsigma, update,curr_grad)

        # print stat
        self.print_count += 1
        if self.print_count % self.print_freq == 0:
            self.print_stat()
        
        return loss

    def update_parameters(self, hessigma_estimate, barsigma, update, curr_gradient):
        # check values
        self.values = []
        self.valuesf = []
        self.denominator = []
        self.numerator = []
        offset = 0
        for group in self.param_groups: ## to check later
            for p in group['params']:
                if p.grad is None:
                    continue
                
                ############################################################
                # update of the parameters with delay since since we needed
                # to compute the missing adjustment
                ############################################################
                
                # compute update
                numel = p.numel()
                update_sub = update[offset:offset + numel]
                
                ############################################################
                # sub step : compute improvement adam
                ############################################################
                
                ## compute numerator
                grad = curr_gradient[offset:offset + numel]
                numerator = torch.sum(grad*grad).item()
                
                ## compute denominator
                barsigma_sub = barsigma[offset:offset + numel]
                Hessigma_sub = hessigma_estimate[offset:offset + numel]
                denominator =  torch.sum(barsigma_sub * Hessigma_sub).item()

                # compute adjustment 
                if denominator > 0:
                    u = min((numerator/denominator),self.clip_val[0]) ## self.clip_val[0] = 10 by default 
                else:
                    u = self.clip_val[1] ## self.clip_val[1] = 1: the value 1 enables us to use standard adam             
                    
#                # check with standard value
#                u = 1

                # check values 
                if denominator != 0:
                    self.values.append((numerator/denominator))
                self.valuesf.append((u))
                self.denominator.append(denominator)
                self.numerator.append(numerator)

                # update parameters                
                p.data.add_(u, update_sub.view_as(p.data)) 

                # update offset; offset is an index used to select the coefficients
                # of the hessian matrix
                offset += numel 

        # save values
        self.values = torch.tensor(self.values)
        self.valuesf = torch.tensor(self.valuesf)
        self.denominator = torch.tensor(self.denominator)
        self.numerator = torch.tensor(self.numerator)
                
        # check all parameters have been updated       
        assert offset == self._numel() 
      
    def update_state(self, curr_pos, curr_grad, curr_barsigma, curr_update, max_memory = 1, closure=None, eps=1e-8):
        # update position
        self.curr_state['prev_position'] = self.curr_state['position']
        self.curr_state['position'] = curr_pos

        # update gradient
        self.curr_state['prev_grad'] = self.curr_state['grad']
        self.curr_state['grad'] = curr_grad
        
        # update grad_deltas
        bound = len(self.curr_state['grad_deltas'])
        grad_delta = self.curr_state['grad'] - self.curr_state['prev_grad']
        if bound < max_memory:            
            self.curr_state['grad_deltas'].append(grad_delta)
        else:
            del self.curr_state['grad_deltas'][0]
            self.curr_state['grad_deltas'].append(grad_delta)            
            
        # update position_deltas
        position_delta = self.curr_state['position'] - self.curr_state['prev_position']
        if bound < max_memory:            
            self.curr_state['position_deltas'].append(position_delta)
        else:
            del self.curr_state['position_deltas'][0]
            self.curr_state['position_deltas'].append(position_delta)   
        
        # update normalisation_factors
        normalisation_factor = 1/(torch.sum(position_delta * grad_delta,0)+ eps)
        if bound < max_memory:            
            self.curr_state['normalisation_factors'].append(normalisation_factor)
        else:
            del self.curr_state['normalisation_factors'][0]
            self.curr_state['normalisation_factors'].append(normalisation_factor)
            
        # update barsigma
        self.curr_state['barsigma'] = curr_barsigma

        # update `update`
        self.curr_state['update'] = curr_update
        
        # update hessigma        
        if self.model:
            self.update_hessigma(self.curr_state, closure) 
        else:
            self.curr_state['hessigma_estimate'] = self.curr_state['barsigma']
        
        # update self.state
        param_state = self.param_groups[0]['params'][0]
        state = self.state[param_state]
        state['optimizer_state_var']['curr_state'] = self.curr_state 
    
    def update_hessigma(self, curr_state, closure):
        """Update the optimizer state by computing the next hessian estimate."""
        # save starting param_groups
        self.past_param_groups = copy.deepcopy(self.param_groups)
 
        # initialisation of the sub optimizer parameters
        self.sub_optimizer.set_values(self.state_dict(), curr_state['barsigma'])
         
        # initialisation number of iterations
        n_iter = 0
        
        while n_iter < self.iter_max:
            # update parameters
            self.sub_optimizer.step(closure=closure)
            
            # update number of iterations
            n_iter += 1

        # initial values of current state and previous state
        curr_state['hessigma_estimate'] = self.sub_optimizer.curr_state['hessigma_estimate'] 

#        grad_deltas = self.sub_optimizer.curr_state['grad_deltas']
#        position_deltas = self.sub_optimizer.curr_state['position_deltas']
#        normalisation_factors = self.sub_optimizer.curr_state['normalisation_factors']
#        
#        pdb.set_trace()
        
        # load past param group values
        self.load_past_param_group()
    
    def load_past_param_group(self):
        if self.past_param_groups:
            size_param_groups = len(self.param_groups)
            for i in range(size_param_groups): 
                group = self.param_groups[i]['params']
                for j in range(len(group)):
                    p1 = self.param_groups[i]['params'][j]
                    p2 = self.past_param_groups[i]['params'][j]
                    p1.data = p2.data  

    def vect_normalisation(self, vect):
        min_v = torch.min(vect)
        range_v = torch.max(vect) - min_v
        if range_v > 0:
            normalised = (vect - min_v) / range_v
        else:
            normalised = torch.zeros(vect.size())
        return normalised
                            
    def print_stat(self):
        self.print_vect_stat(self.values, 'adjustment before clipping')
        
        self.print_vect_stat(self.valuesf, 'final adjustment')
        
#        self.print_vect_stat(self.curr_state['hessigma_estimate'], 'hessigma_estimate')
#
#        self.print_vect_stat(self.curr_state['barsigma'], 'barsigma')
#        
#        self.print_vect_stat(self.denominator, 'denominator')
#        
#        self.print_vect_stat(self.numerator, 'numerator')
        
        
    def print_vect_stat(self, vect, vect_name = ''):
        print('--------------------------------------------------------')
        print('--------------- print ' + vect_name + '-----------------')
        print('--------------------------------------------------------')
        try:
            print(' shape ' + vect_name + ' is : ' + str(vect.shape))
            print(' max ' + vect_name + ' is : ' + str(vect.max()))
            print(' min ' + vect_name + ' is : ' + str(vect.min()))
            print(' mean ' + vect_name + ' is : ' + str(vect.mean()))
            print(' median ' + vect_name + ' is : ' + str(vect.median()))
            if self.histogram :
                print(' histogram ' + vect_name + ' is : ')
                import matplotlib.pyplot as plt 
                indeces = torch.randint(len(vect), (200,))
                values = vect.view(-1)[indeces]
                plt.hist(values)
                plt.show()
        except:
            print(' ' + vect_name + ' is : ')
            print(vect)
        print('--------------------------------------------------------')


class LBFGS_sub(Optimizer):
    r"""Implements an improvement of Adam algorithm.

    The has been proposed in ` to fill `_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Add references 
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=required, max_memory = 20, barsigma = torch.empty(0)):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(LBFGS_sub, self).__init__(params, defaults)
        
        # initialisation current state and previous state
        self.curr_state = dict()
        
        # initialisation number of elements (parameters)
        self._numel_cache = None
        
        # initialisation maximal number of saved past gradients per optimization step
        self.max_memory = max_memory
        
        # initialise barsigma 
        self.barsigma = barsigma

        # initialise state
        self.init_values_state()
            
        # store the control value and print them
        self.values = None
        self.denominator = None
        self.numerator = None

        # print frequency
        self.print_count = 0
        self.print_freq = 50
        
    def __setstate__(self, state):
        super(LBFGS_sub, self).__setstate__(state)

    def _numel(self):
        _params = self.param_groups[0]['params']
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), _params, 0)
        return self._numel_cache

    def init_values_state(self):
        # initialisation state
        # NOTE: we use the state of the first param to store current state and previous state
        # because this helps with casting in load_state_dict        
        param_state = self.param_groups[0]['params'][0]
        state = self.state[param_state]

        # initialisation for the first call of the function step        
        if 'optimizer_state_var' in state:
            del state['optimizer_state_var']
            
    def set_values(self, state_dict, barsigma = torch.empty(0)):
        # copy state_dict
        self.load_state_dict(state_dict)

        # initialise bar sigma 
        self.barsigma = barsigma
        
        # initialise state
        self.init_values_state()
        
    def init_optimizer(self):
        # position and gradient initialitsation
        positions = [] # store position values
        grads = [] # store gradient values
        for group in self.param_groups: ## to check later 
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # save position
                positions.append(p.data.view(-1))

                # save gradient
                grads.append(d_p.view(-1)) # we do not save the raw gradient
             
                # update parameters
                p.data.add_(- group['lr'], d_p)
                
        # unsqueeze to get the shape (vector_size,1)
        positions = torch.cat(positions, 0).unsqueeze(-1)
        grads = torch.cat(grads, 0).unsqueeze(-1)
              
        return [positions, grads]

    def get_update_state_optimizer(self):
        # position and gradient initialitsation
        positions = [] # store position values
        grads = [] # store gradient values
        for group in self.param_groups: ## to check later
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                
                # save position
                positions.append(p.data.view(-1))

                # save gradient
                grads.append(d_p.view(-1)) # we do not save the raw gradient

                # update parameters
                p.data.add_(- group['lr'], d_p)
                
        # unsqueeze to get the shape (vector_size,1)
        positions = torch.cat(positions, 0).unsqueeze(-1)
        grads = torch.cat(grads, 0).unsqueeze(-1)
              
        return [positions, grads]
    
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            self.zero_grad()
            loss = closure()

        # initialisation current state and previous state
        # NOTE: we use the state of the first param to store current state and previous state
        # because this helps with casting in load_state_dict
        param_state = self.param_groups[0]['params'][0]
        state = self.state[param_state]

        # initialisation for the first call of the function step        
        if not('optimizer_state_var' in state):
            # the function init_optimizer computes initial positions, gradients, and hessian estimate
            # plus updates the parameters
            init_positions, init_gradients = self.init_optimizer()
            
            # initialise barsigma 
            if self.barsigma.nelement() == 0 :
                self.barsigma = 1e-02*torch.randn(init_positions.shape)  
                
            state['optimizer_state_var'] = {'curr_state' : { 'position': init_positions,
                                                             'grad': init_gradients,
                                                             'prev_position': init_positions,
                                                             'prev_grad': init_gradients,
                                                             'barsigma': self.barsigma,
                                                             'grad_deltas':[],
                                                             'position_deltas':[],
                                                             'normalisation_factors':[],
                                                             'hessigma_estimate':torch.empty(init_positions.shape),
                                                            }}
    
            # initial values of current state and previous state
            self.curr_state = state['optimizer_state_var']['curr_state']
        # main routine:
        else:
            # compute initial positions, gradients, and hessian estimate
            curr_pos, curr_grad = self.get_update_state_optimizer()
            
            # update cur_state and prev_state; the main update is the hessian 
            self.update_state(curr_pos, curr_grad, self.max_memory)
        
#        # print stat
#        self.print_count += 1
#        if self.print_count % self.print_freq == 0:
#            self.print_stat()
        
        return loss
        
    def update_state(self, curr_pos, curr_grad, max_memory = 1, eps=1e-8):
        # update position
        self.curr_state['prev_position'] = self.curr_state['position']
        self.curr_state['position'] = curr_pos

        # update gradient
        self.curr_state['prev_grad'] = self.curr_state['grad']
        self.curr_state['grad'] = curr_grad
        
        # update grad_deltas
        bound = len(self.curr_state['grad_deltas'])
        grad_delta = self.curr_state['grad'] - self.curr_state['prev_grad']
        if bound < max_memory:            
            self.curr_state['grad_deltas'].append(grad_delta)
        else:
            del self.curr_state['grad_deltas'][0]
            self.curr_state['grad_deltas'].append(grad_delta)            
            
        # update position_deltas
        position_delta = self.curr_state['position'] - self.curr_state['prev_position']
        if bound < max_memory:            
            self.curr_state['position_deltas'].append(position_delta)
        else:
            del self.curr_state['position_deltas'][0]
            self.curr_state['position_deltas'].append(position_delta)   
        
        # update normalisation_factors
        normalisation_factor = 1/(torch.sum(position_delta * grad_delta,0)+ eps)
        if bound < max_memory:            
            self.curr_state['normalisation_factors'].append(normalisation_factor)
        else:
            del self.curr_state['normalisation_factors'][0]
            self.curr_state['normalisation_factors'].append(normalisation_factor)
        
        # update hessigma
        self.update_hessigma(self.curr_state) 
        
        # update self.state
        param_state = self.param_groups[0]['params'][0]
        state = self.state[param_state]
        state['optimizer_state_var']['curr_state'] = self.curr_state 
    
    def update_hessigma(self, curr_state):
        """Update the optimizer state by computing the next hessian estimate."""
        
        # new hessian estimate
        barsigma = curr_state['barsigma']
        grad_deltas = curr_state['grad_deltas']
        position_deltas = curr_state['position_deltas']
        normalisation_factors = curr_state['normalisation_factors'] 
        
        curr_hessigma_estimate = self.lbfgs_hessigma_update(barsigma, grad_deltas, position_deltas, normalisation_factors, eps=1e-8)
        
        # conditioning if necessary : later on
        
        # update hessigma estimate 
        curr_state['hessigma_estimate'] = curr_hessigma_estimate
        
    def lbfgs_hessigma_update(self, barsigma, grad_deltas, position_deltas, normalisation_factors, eps=1e-8):
        """Applies the two loops update rule for l-BFGS to estimate the vector 
           `H * barsigma` with H the hessian matrix and  barsigma the brownian volatility .
           
           to fill at the end
          
        """
        # compute initial q 
        q = barsigma
        # descending loop
        alpha_reverted = []
        bound = len(grad_deltas)
        for i in range(bound-1,-1,-1):
            alpha_value = normalisation_factors[i] * torch.sum(grad_deltas[i] * q)
            alpha_reverted.append(alpha_value)
            q = q - alpha_value * position_deltas[i]
        # compute the scalar h_0 with H_0 = h_0 * I
        h_0 = torch.sum(grad_deltas[-1] * position_deltas[-1]) / (torch.sum(position_deltas[-1] * position_deltas[-1])  + eps) # to check
#        h_0 = torch.sum(grad_deltas[-1] * grad_deltas[-1])  / ( torch.sum(grad_deltas[-1] * position_deltas[-1]) + eps) # to check
        # compute initial r
        r = h_0 * q 
        # ascending loop 
        for i in range(bound):
            beta_value = normalisation_factors[i] * torch.sum(position_deltas[i] * r)
            r = r + (alpha_reverted[-(i+1)] - beta_value) * grad_deltas[i]
        
        return r 
    
    def print_stat(self):
        self.print_vect_stat(self.values, 'self.values')
        
        self.print_vect_stat(self.curr_state['hessigma_estimate'], 'hessigma_estimate')
        
        self.print_vect_stat(self.denominator, 'denominator')
        
        self.print_vect_stat(self.numerator, 'numerator')
        
        
    def print_vect_stat(self, vect, vect_name = ''):
        print('--------------------------------------------------------')
        print('--------------- print ' + vect_name + '-----------------')
        print('--------------------------------------------------------')
        try:
            print(' shape ' + vect_name + ' is : ' + str(vect.shape))
            print(' max ' + vect_name + ' is : ' + str(vect.max()))
            print(' min ' + vect_name + ' is : ' + str(vect.min()))
            print(' mean ' + vect_name + ' is : ' + str(vect.mean()))
            print(' median ' + vect_name + ' is : ' + str(vect.median()))
            print(' histogram ' + vect_name + ' is : ')
            import matplotlib.pyplot as plt 
            indeces = torch.randint(len(vect), (200,))
            values = vect.view(-1)[indeces]
            plt.hist(values)
            plt.show()
        except:
            print(' ' + vect_name + ' is : ')
            print(vect)
        print('--------------------------------------------------------')