import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
# import torch.utils.tensorboard as tensorboardX
from thop import profile
from thop import clever_format
import shutil
import random

class QSGAN(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'QSGAN_light'
        else :
            self.model_name = 'QSGAN'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.identity_weight = args.identity_weight
        self.penalty_weight = args.penalty_weight
        self.contrast_weight = args.contrast_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        self.start_iter = 1

        self.fid = 1000
        self.fid_A = 1000
        self.fid_B = 1000
        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)
        print("# the size of image : ", self.img_size)
        print("# the size of image channel : ", self.img_ch)
        print("# base channel number per layer : ", self.ch)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layers : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# penalty_weight : ", self.penalty_weight)
        print("# contrast_weight : ", self.contrast_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size+30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False,pin_memory=True)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False,pin_memory=True)

        """ Define Generator, Discriminator """
        self.gen2B = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.gen2A = ResnetGenerator(input_nc=self.img_ch, output_nc=self.img_ch, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disA = Discriminator(input_nc=self.img_ch, ndf=self.ch).to(self.device)
        self.disB = Discriminator(input_nc=self.img_ch, ndf=self.ch).to(self.device)
        
        print('-----------------------------------------------')
        input = torch.randn([1, self.img_ch, self.img_size, self.img_size]).to(self.device)
        macs, params = profile(self.disA, inputs=(input, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'disA', params)
        print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
        print('-----------------------------------------------')
        x1, x2, x3, _,  _, real_A_ae = self.disA(input)
        macs, params = profile(self.gen2B, inputs=(input,real_A_ae,x1,x2,x3, ))
        macs, params = clever_format([macs*2, params*2], "%.3f")
        print('[Network %s] Total number of parameters: ' % 'gen2B', params)
        print('[Network %s] Total number of FLOPs: ' % 'gen2B', macs)
        print('-----------------------------------------------')

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)

        """ Trainer """ 
        self.G_optim = torch.optim.Adam(itertools.chain(self.gen2B.parameters(), self.gen2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disA.parameters(), self.disB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)


    def train(self):
        # writer = tensorboardX.SummaryWriter(os.path.join(self.result_dir, self.dataset, 'summaries/Allothers'))
        self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

        self.start_iter = 1
        if self.resume:
            params = torch.load(os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
            self.gen2B.load_state_dict(params['gen2B'])
            self.gen2A.load_state_dict(params['gen2A'])
            self.disA.load_state_dict(params['disA'])
            self.disB.load_state_dict(params['disB'])
            self.D_optim.load_state_dict(params['D_optimizer'])
            self.G_optim.load_state_dict(params['G_optimizer'])
            self.start_iter = params['start_iter']+1
            if self.decay_flag and self.start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (self.start_iter - self.iteration // 2)
            print("ok")
          

        # training loop
        testnum = 4
        for step in range(1, self.start_iter):
            if step % self.print_freq == 0:
                for _ in range(testnum):
                    try:
                        real_A, _ = next(testA_iter) #testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ =next(testA_iter) #testA_iter.next()

                    try:
                        real_B, _ = next(testB_iter) #testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = next(testB_iter) #testB_iter.next()

        print("self.start_iter",self.start_iter)
        print('training start !')
        start_time = time.time()
        for step in range(self.start_iter, self.iteration + 1):
            self.n_res = random.randint(1, 100)
            if step > 100:
                self.n_res = 100
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = next(trainA_iter) #trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = next(trainA_iter) #trainA_iter.next()

            try:
                real_B, _ = next(trainB_iter) #trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = next(trainB_iter) #trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            self.D_optim.zero_grad()

            real_A_x1, real_A_x2, real_A_x3,real_LA_logit,real_GA_logit, real_A_z = self.disA(real_A)
            real_B_x1, real_B_x2, real_B_x3,real_LB_logit,real_GB_logit, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A, real_A_z, real_A_x1, real_A_x2, real_A_x3)
            fake_B2A = self.gen2A(real_B, real_B_z, real_B_x1, real_B_x2, real_B_x3)

            fake_B2A = fake_B2A.detach()
            fake_A2B = fake_A2B.detach()

            fake_A2B_x1, fake_A2B_x2, fake_A2B_x3, fake_LB_logit,fake_GB_logit, fake_B_z = self.disB(fake_A2B)
            fake_B2A_x1, fake_B2A_x2, fake_B2A_x3, fake_LA_logit,fake_GA_logit, fake_A_z = self.disA(fake_B2A)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_loss_penalty_A = self.cal_gradient_penalty(real_A, w=True) + self.cal_gradient_penalty(fake_B2A, w=True)
            D_loss_penalty_B = self.cal_gradient_penalty(real_B, w=False) + self.cal_gradient_penalty(fake_A2B, w=False)
            
            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_loss_LA) + self.penalty_weight * D_loss_penalty_A
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_loss_LB) + self.penalty_weight * D_loss_penalty_B

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step() 

            # Update G
            self.G_optim.zero_grad()

            real_A_x1, real_A_x2, real_A_x3,_,_, real_A_z = self.disA(real_A)
            real_B_x1, real_B_x2, real_B_x3,_,_, real_B_z = self.disB(real_B)

            fake_A2B = self.gen2B(real_A, real_A_z, real_A_x1, real_A_x2, real_A_x3)
            fake_B2A = self.gen2A(real_B, real_B_z, real_B_x1, real_B_x2, real_B_x3)

            fake_B2A_x1, fake_B2A_x2, fake_B2A_x3, fake_LA_logit,fake_GA_logit, fake_A_z = self.disA(fake_B2A)
            fake_A2B_x1, fake_A2B_x2, fake_A2B_x3, fake_LB_logit,fake_GB_logit, fake_B_z = self.disB(fake_A2B)

            fake_B2A2B = self.gen2B(fake_A_z, fake_A_z, fake_B2A_x1, fake_B2A_x2, fake_B2A_x3)
            fake_A2B2A = self.gen2A(fake_B_z, fake_B_z, fake_A2B_x1, fake_A2B_x2, fake_A2B_x3,)
            
            fake_A2B2A_x1, fake_A2B2A_x2, fake_A2B2A_x3, _,_, fake_A2B2A_z = self.disA(fake_A2B2A)
            fake_B2A2B_x1, fake_B2A2B_x2, fake_B2A2B_x3, _,_, fake_B2A2B_z = self.disB(fake_B2A2B)

            Ireal_B_x1, Ireal_B_x2, Ireal_B_x3,_,_, Ireal_B_z = self.disA(real_B)
            Ireal_A_x1, Ireal_A_x2, Ireal_A_x3,_,_, Ireal_A_z = self.disB(real_A)

            fake_A2A = self.gen2A(real_A, Ireal_A_z, Ireal_A_x1, Ireal_A_x2, Ireal_A_x3)
            fake_B2B = self.gen2B(real_B, Ireal_B_z, Ireal_B_x1, Ireal_B_x2, Ireal_B_x3)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))

            G_Contrast_Loss_A = self.ContrastLoss([fake_A2B2A_x1, fake_A2B2A_x2, fake_A2B2A_x3], [real_A_x1, real_A_x2, real_A_x3], [fake_A2B_x1, fake_A2B_x2, fake_A2B_x3])
            G_Contrast_Loss_B = self.ContrastLoss([fake_B2A2B_x1, fake_B2A2B_x2, fake_B2A2B_x3], [real_B_x1, real_B_x2, real_B_x3], [fake_B2A_x1, fake_B2A_x2, fake_B2A_x3])

            #G_cycle_loss_A = self.L1_loss(fake_A2B2A, real_A)
            #G_cycle_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_loss_LA ) + self.contrast_weight * G_Contrast_Loss_A + self.identity_weight * G_identity_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_loss_LB ) + self.contrast_weight * G_Contrast_Loss_B + self.identity_weight * G_identity_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            #save_dr_latest = "/content/gdrive/MyDrive/checkpointTTL-GAN/" + self.dataset + "_params_latest.pt"
            #save_TTLGAN_latest = "/content/TTL-GAN/results/" + self.dataset + "_params_latest.pt"
            #save_dr_step = "/content/gdrive/MyDrive/checkpointTTL-GAN/" + self.dataset + '_params_%07d.pt' % step
            #save_TTLGAN_step = "/content/TTL-GAN/results/" + self.dataset + "/model/" + self.dataset + '_params_%07d.pt' % step
            #shutil.copy( save_TTLGAN_latest , save_dr_latest )
            #shutil.copy( save_TTLGAN_step , save_dr_step )
            self.G_optim.step()
            # writer.add_scalar('G/%s' % 'loss_A', G_loss_A.data.cpu().numpy(), global_step=step)  
            # writer.add_scalar('G/%s' % 'loss_B', G_loss_B.data.cpu().numpy(), global_step=step)  

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))

            # for name, param in self.gen2B.named_parameters():
            #     writer.add_histogram(name + "_gen2B", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.gen2A.named_parameters():
            #     writer.add_histogram(name + "_gen2A", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.disA.named_parameters():
            #     writer.add_histogram(name + "_disA", param.data.cpu().numpy(), global_step=step)

            # for name, param in self.disB.named_parameters():
            #     writer.add_histogram(name + "_disB", param.data.cpu().numpy(), global_step=step)

            
            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)
                #save_dr_latest = "/content/gdrive/MyDrive/checkpointTTL-GAN/" + self.dataset + "_params_latest.pt"
                #save_TTLGAN_latest = "/content/TTL-GAN/results/" + self.dataset + "_params_latest.pt"
                save_dr_step = "/content/gdrive/MyDrive/checkpointTTL-GAN/" + self.dataset + '_params_%07d.pt' % step
                save_TTLGAN_step = "/content/TTL-GAN/results/" + self.dataset + "/model/" + self.dataset + '_params_%07d.pt' % step
                #shutil.copy( save_TTLGAN_latest , save_dr_latest )
                shutil.copy( save_TTLGAN_step , save_dr_step )

            if step % self.print_freq == 0:
                print('current D_learning rate:{}'.format(self.D_optim.param_groups[0]['lr']))
                print('current G_learning rate:{}'.format(self.G_optim.param_groups[0]['lr']))
                self.save_path("_params_latest.pt",step)

            if step % self.print_freq == 0:
                train_sample_num = testnum
                test_sample_num = testnum
                A2B = np.zeros((self.img_size * 5, 0, 3))
                B2A = np.zeros((self.img_size * 5, 0, 3))

                self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(), self.disB.eval()

                self.gen2B,self.gen2A = self.gen2B.to('cpu'), self.gen2A.to('cpu')
                self.disA,self.disB = self.disA.to('cpu'), self.disB.to('cpu')
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = trainA_iter.next()
                    except:
                        trainA_iter = iter(self.trainA_loader)
                        real_A, _ = trainA_iter.next()

                    try:
                        real_B, _ = trainB_iter.next()
                    except:
                        trainB_iter = iter(self.trainB_loader)
                        real_B, _ = trainB_iter.next()
                    real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
                    # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    
                    real_A2B_x1, real_A2B_x2, real_A2B_x3,  _, _, real_A_z= self.disA(real_A)
                    real_B2A_x1, real_B2A_x2, real_B2A_x3,  _, _, real_B_z= self.disB(real_B)

                    fake_A2B = self.gen2B(real_A, real_A_z, real_A2B_x1, real_A2B_x2, real_A2B_x3)
                    fake_B2A = self.gen2A(real_B, real_B_z, real_B2A_x1, real_B2A_x2, real_B2A_x3)

                    fake_A2B_x1, fake_A2B_x2, fake_A2B_x3,  _,  _,  fake_A_z = self.disA(fake_B2A)
                    fake_B2A_x1, fake_B2A_x2, fake_B2A_x3,  _,  _,  fake_B_z = self.disB(fake_A2B)

                    fake_B2A2B = self.gen2B(fake_B2A, fake_A_z, fake_A2B_x1, fake_A2B_x2, fake_A2B_x3)
                    fake_A2B2A = self.gen2A(fake_A2B, fake_B_z, fake_B2A_x1, fake_B2A_x2, fake_B2A_x3)

                    Ireal_B_x1, Ireal_B_x2, Ireal_B_x3,_,_, Ireal_B_z = self.disA(real_B)
                    Ireal_A_x1, Ireal_A_x2, Ireal_A_x3,_,_, Ireal_A_z = self.disB(real_A)
        
                    fake_A2A = self.gen2A(real_A, Ireal_A_z, Ireal_A_x1, Ireal_A_x2, Ireal_A_x3)
                    fake_B2B = self.gen2B(real_B, Ireal_B_z, Ireal_B_x1, Ireal_B_x2, Ireal_B_x3)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = testA_iter.next()
                    except:
                        testA_iter = iter(self.testA_loader)
                        real_A, _ = testA_iter.next()

                    try:
                        real_B, _ = testB_iter.next()
                    except:
                        testB_iter = iter(self.testB_loader)
                        real_B, _ = testB_iter.next()
                    real_A, real_B = real_A.to('cpu'), real_B.to('cpu')
                    # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                    real_A2B_x1, real_A2B_x2, real_A2B_x3,  _, _, real_A_z= self.disA(real_A)
                    real_B2A_x1, real_B2A_x2, real_B2A_x3,  _, _, real_B_z= self.disB(real_B)

                    fake_A2B = self.gen2B(real_A, real_A_z, real_A2B_x1, real_A2B_x2, real_A2B_x3)
                    fake_B2A = self.gen2A(real_B, real_B_z, real_B2A_x1, real_B2A_x2, real_B2A_x3)

                    fake_A2B_x1, fake_A2B_x2, fake_A2B_x3,  _,  _,  fake_A_z = self.disA(fake_B2A)
                    fake_B2A_x1, fake_B2A_x2, fake_B2A_x3,  _,  _,  fake_B_z = self.disB(fake_A2B)

                    fake_B2A2B = self.gen2B(fake_B2A, fake_A_z, fake_A2B_x1, fake_A2B_x2, fake_A2B_x3)
                    fake_A2B2A = self.gen2A(fake_A2B, fake_B_z, fake_B2A_x1, fake_B2A_x2, fake_B2A_x3)

                    Ireal_B_x1, Ireal_B_x2, Ireal_B_x3,_,_, Ireal_B_z = self.disA(real_B)
                    Ireal_A_x1, Ireal_A_x2, Ireal_A_x3,_,_, Ireal_A_z = self.disB(real_A)
        
                    fake_A2A = self.gen2A(real_A, Ireal_A_z, Ireal_A_x1, Ireal_A_x2, Ireal_A_x3)
                    fake_B2B = self.gen2B(real_B, Ireal_B_z, Ireal_B_x1, Ireal_B_x2, Ireal_B_x3)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                

                self.gen2B,self.gen2A = self.gen2B.to(self.device), self.gen2A.to(self.device)
                self.disA,self.disB = self.disA.to(self.device), self.disB.to(self.device)
                
                self.gen2B.train(), self.gen2A.train(), self.disA.train(), self.disB.train()

    def cal_gradient_penalty(self, x, w=True):
        constant = 1.0
        x.requires_grad_(True)
        if w:
            _, _, _, out1, out2, _ = self.disA(x)
        else:
            _, _, _, out1, out2, _ = self.disB(x)
        #y = model(x)
        #using torch.autograd.grad
        gradients1 = torch.autograd.grad(out1, x, retain_graph=True, grad_outputs=torch.ones_like(out1))[0] 
        gradients1 = gradients1[0].view(x.size(0), -1)
        gradients2 = torch.autograd.grad(out2, x, retain_graph=True, grad_outputs=torch.ones_like(out2))[0] 
        gradients2 = gradients2[0].view(x.size(0), -1)
        gradient_penalty1 = (((gradients1 + 1e-16).norm(2, dim=1) - constant) ** 2).mean()
        gradient_penalty2 = (((gradients2 + 1e-16).norm(2, dim=1) - constant) ** 2).mean()
        return gradient_penalty1+gradient_penalty2

    def ContrastLoss(self, a, p, n):
      loss = 0
      tau = 0.07

      d_ap, d_an = 0, 0
      for i in range(len(a)):

          d_ap = torch.exp(torch.pow((a[i] - p[i].detach()),2).mean()/tau)
          d_an = torch.exp(torch.pow((a[i] - n[i].detach()),2).mean()/tau)
          contrastive = - torch.log(d_ap / (d_an + d_ap))

          loss += contrastive

      return loss
    
    def save(self, dir, step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))


    def save_path(self, path_g,step):
        params = {}
        params['gen2B'] = self.gen2B.state_dict()
        params['gen2A'] = self.gen2A.state_dict()
        params['disA'] = self.disA.state_dict()
        params['disB'] = self.disB.state_dict()
        params['D_optimizer'] = self.D_optim.state_dict()
        params['G_optimizer'] = self.G_optim.state_dict()
        params['start_iter'] = step
        torch.save(params, os.path.join(self.result_dir, self.dataset + path_g))

    def load(self):
        params = torch.load(os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))
        self.gen2B.load_state_dict(params['gen2B'])
        self.gen2A.load_state_dict(params['gen2A'])
        self.disA.load_state_dict(params['disA'])
        self.disB.load_state_dict(params['disB'])
        self.D_optim.load_state_dict(params['D_optimizer'])
        self.G_optim.load_state_dict(params['G_optimizer'])
        self.start_iter = params['start_iter']

    def test(self):
        self.load()
        print(self.start_iter)

        self.gen2B.eval(), self.gen2A.eval(), self.disA.eval(),self.disB.eval()
        for n, (real_A, real_A_path) in enumerate(self.testA_loader):
            real_A = real_A.to(self.device)
            real_A2B_x1, real_A2B_x2, real_A2B_x3,  _, _, real_A_z= self.disA(real_A)
            fake_A2B = self.gen2B(real_A, real_A_z, real_A2B_x1, real_A2B_x2, real_A2B_x3)

            A2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
            print(real_A_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeB', real_A_path[0].split('/')[-1]), A2B * 255.0)

        for n, (real_B, real_B_path) in enumerate(self.testB_loader):
            real_B = real_B.to(self.device)

            real_B2A_x1, real_B2A_x2, real_B2A_x3,  _, _, real_B_z= self.disB(real_B)
            fake_B2A = self.gen2A(real_B, real_B_z, real_B2A_x1, real_B2A_x2, real_B2A_x3)

            B2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
            print(real_B_path[0])
            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'fakeA', real_B_path[0].split('/')[-1]), B2A * 255.0)
