import torch
import visdom
from Modules import *
from Dataset import *
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F


def train_test(cfg):
    """
    ALOCC 复现
    
    """""
    print("Prepare...")
    device = torch.device(cfg['device'])
    weight_decay = cfg['weight_decay']
    encoder = cfg['encoder'].to(device)
    decoder = cfg['decoder'].to(device)
    discriminator = cfg['discriminator'].to(device)
    lr = cfg['init_lr']
    # start_calculate = cfg['calculate_c']
    opt_encoder = optim.Adam([{'params': encoder.parameters(), 'initial_lr': lr}], lr=lr, weight_decay=weight_decay)
    opt_decoder = optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    opt_discriminator = optim.Adam([{'params': discriminator.parameters(), 'initial_lr': lr}], lr=lr,
                                   weight_decay=weight_decay)
    scheduler_encoder = optim.lr_scheduler.MultiStepLR(opt_encoder, milestones=[50, 60, 70, 80], gamma=0.8)
    scheduler_decoder = optim.lr_scheduler.MultiStepLR(opt_decoder, milestones=[50, 60, 70, 80], gamma=0.8)
    scheduler_discriminator = optim.lr_scheduler.MultiStepLR(opt_discriminator, milestones=[50, 60, 70, 80], gamma=0.8)
    viz = visdom.Visdom()
    epochs = cfg['epochs']
    batch_size = cfg['batch_size']
    positive = cfg['positive']
    dataset.setMode('train', positive_class=positive)
    datasetLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # # 模型参数：超球中心点c
    # c = hyperSphereCenter = None
    # 统计训练信息
    globalTrainStep = 0
    total_loss_dis_true = 0
    total_loss_dis_fake = 0
    total_loss_dis = 0
    total_loss_gen_fake = 0
    total_loss_gen_mse = 0
    # total_loss_gen_dist = 0
    total_loss_gen = 0
    total_ssim = 0
    best_precision = 0
    best_recall = 0
    best_accuracy = 0
    best_f1 = 0
    best_precision_epoch = 0
    best_recall_epoch = 0
    best_accuracy_epoch = 0
    best_f1_epoch = 0

    for epoch in range(epochs):

        # # 第50轮时，假定编码器能够正确提取到正样本数据特征
        # # 计算超球中心点c
        # if epoch == start_calculate:
        #     print('Calculate HyperSphere Center...')
        #     c = torch.zeros(cfg['feature_dim'], device=device)
        #     n_samples = 0
        #     eps = 0.1
        #     with torch.no_grad():
        #         dataset.setMode('train', positive_class=positive)
        #         encoder.eval()
        #         decoder.eval()
        #         for sourceImage, label in datasetLoader:
        #             sourceImage = sourceImage.to(device)
        #             feature_vectors = encoder(sourceImage)
        #             n_samples += feature_vectors.shape[0]
        #             c += torch.sum(feature_vectors, dim=0)
        #     c /= n_samples
        #     # 确保超球中心不要太接近于特征空间的原点
        #     c[(abs(c) < eps) & (c < 0)] = -eps
        #     c[(abs(c) < eps) & (c > 0)] = eps
        # 训练
        print(f'Train{epoch}...')
        dataset.setMode('train', positive_class=positive)
        encoder.train()
        decoder.train()
        discriminator.train()
        for sourceImage, label in datasetLoader:
            viz.images(sourceImage, nrow=16, win='Train-sourceImage', opts=dict(title='Train-sourceImage'))
            sourceImage = Variable(sourceImage.to(device))
            # 原图片添加噪声
            _, noisyImage = addNoise(sourceImage, 0.1, device)
            # 用编码器和解码器重建图像
            feature_vectors = encoder(noisyImage)
            reconstruction = decoder(feature_vectors)
            # 判别器分别对原图像和重建的图像进行打分
            # 对原图打分应趋近于1，对重建图像打分应趋近于0，用交叉熵分别计算损失，整个判别器的损失为原图损失和重建图像损失的和
            scores_true = discriminator(sourceImage)
            ones = torch.ones(len(scores_true), dtype=torch.long, device=device)
            loss_true = F.cross_entropy(scores_true, ones)
            scores_fake = discriminator(reconstruction)
            zeros = torch.zeros(len(scores_fake), dtype=torch.long, device=device)
            loss_fake = F.cross_entropy(scores_fake, zeros)

            loss_dis = torch.add(loss_true, loss_fake)

            opt_decoder.zero_grad()
            opt_encoder.zero_grad()
            opt_discriminator.zero_grad()
            loss_dis.backward(retain_graph=True)
            opt_discriminator.step()
            scheduler_discriminator.step()

            # 训练生成器
            # 生成器生成的图像应该尽量和原图相似
            # 用均方误差MSE评价重建图像与原图的误差，这个参数用于反向传播
            # 用更新过的判别器评判重建图像分数，令其趋于1-->生成器的工作是要欺骗判别器，用交叉熵计算这个损失
            # 用结构相似性SSIM评价重建图像与原图的相似性，这个参数不参与反向传播
            # 生成器的损失为欺骗损失和结构损失之和
            scores_fake = discriminator(reconstruction)
            ones2 = torch.ones(len(scores_fake), dtype=torch.long, device=device)
            loss_gen_mse = F.cross_entropy(scores_fake, ones2)

            loss_gen_l2 = F.mse_loss(sourceImage, reconstruction)
            loss_gen = torch.add(loss_gen_mse, torch.mul(0.2, loss_gen_l2))

            # # 第50轮后，计算出超球中心，约束,使提取到的特征向量到超球中心距离尽量小
            # if epoch >= start_calculate:
            #     loss_dist = torch.sum((feature_vectors - c) ** 2, dim=1)
            #     loss_dist = torch.mean(loss_dist)
            #     loss_dist = torch.mul(loss_dist, 0.2)
            #     loss_gen = torch.add(loss_gen, loss_dist)

            loss_ssim = SSIM(reconstruction, sourceImage)

            opt_decoder.zero_grad()
            opt_encoder.zero_grad()
            opt_discriminator.zero_grad()
            loss_gen.backward()
            opt_decoder.step()
            opt_encoder.step()
            scheduler_encoder.step()
            scheduler_decoder.step()

            total_loss_dis_true += loss_true.item()
            total_loss_dis_fake += loss_fake.item()
            total_loss_dis += loss_dis.item()
            total_loss_gen_fake += loss_gen_l2.item()
            total_loss_gen_mse += loss_gen_mse.item()
            # if epoch >= start_calculate:
            #     total_loss_gen_dist += loss_dist.item()
            total_loss_gen += loss_gen.item()
            total_ssim += loss_ssim
            globalTrainStep += 1

            viz.line([total_loss_dis_true / globalTrainStep], [globalTrainStep], win='Train-Discriminator True Loss',
                     update='append', opts=dict(title='Train-Discriminator True Loss'))
            viz.line([total_loss_dis_fake / globalTrainStep], [globalTrainStep], win='Train-Discriminator Fake Loss',
                     update='append', opts=dict(title='Train-Discriminator Fake Loss'))
            viz.line([total_loss_dis / globalTrainStep], [globalTrainStep], win='Train-Discriminator Loss',
                     update='append', opts=dict(title='Train-Discriminator Loss'))
            viz.line([total_loss_gen_fake / globalTrainStep], [globalTrainStep], win='Train-Generate Fake Loss',
                     update='append', opts=dict(title='Train-Generate Fake Loss'))
            viz.line([total_loss_gen_mse / globalTrainStep], [globalTrainStep], win='Train-Generate MSE Loss',
                     update='append', opts=dict(title='Train-Generate MSE Loss'))
            viz.line([total_loss_gen / globalTrainStep], [globalTrainStep], win='Train-Generate Loss',
                     update='append', opts=dict(title='Train-Generate Loss'))
            viz.line([total_ssim / globalTrainStep], [globalTrainStep], win='Train-SSIM', update='append',
                     opts=dict(title='Train-SSIM'))
            # if epoch >= start_calculate:
            #     viz.line([total_loss_gen_dist / globalTrainStep], [globalTrainStep], win='Train-Generate Dist Loss',
            #              update='append', opts=dict(title='Train-Generate Dist Loss'))
            viz.images(reconstruction, nrow=16, win='Reconstruction',
                       opts=dict(title='Reconstruction'))
            log.write(
                f'Epoch-{epoch}:[[DiscriminatorTrueLoss,{total_loss_dis_true / globalTrainStep}],'
                f'[DiscriminatorFakeLoss,{total_loss_dis_fake / globalTrainStep}],'
                f'[DiscriminatorLoss,{total_loss_dis_fake / globalTrainStep}],'
                f'[Generate Fake Loss,{total_loss_gen_fake / globalTrainStep}],'
                f'[Generate MSE Loss,{total_loss_gen_mse / globalTrainStep}],'
                f'[Generate Loss,{total_loss_gen / globalTrainStep}],'
                f'[SSIM,{total_ssim / globalTrainStep}]]\n')
            log.flush()
        # 测试验证
        with torch.no_grad():
            print('Test...')
            dataset.setMode('test', positive_class=positive, np_proportion=3)
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            total_recall = 0
            total_precision = 0
            total_accuracy = 0
            total_f1 = 0
            validate_step = 0

            for sourceImage, labels in datasetLoader:
                viz.images(sourceImage, nrow=16, win='Validate-sourceImage', opts=dict(title='Validate-sourceImage'))
                sourceImage = Variable(sourceImage.to(device))
                labels = Variable(labels).float().to(device)
                # 顺便验证一下生成器的重建效果
                feature_vectors = encoder(sourceImage)
                reconstruction = decoder(feature_vectors)
                # 判别器直接对测试用图片打分
                scores = discriminator(sourceImage)
                # 判别器应该将正类的打分更高
                predict = torch.argmax(scores, dim=1)
                predict = predict.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                # 计算准确率Accuracy,查准率Precision,查全率Recall,F1 参数
                accuracy = metrics.accuracy_score(labels, predict)
                precision, recall, f, _ = metrics.precision_recall_fscore_support(labels, predict, zero_division=0)
                validate_step += 1
                total_recall += np.max(recall)
                total_precision += np.max(precision)
                total_accuracy += accuracy
                total_f1 += np.max(f)
                viz.images(reconstruction, nrow=16, win='Validate-reconstruction',
                           opts=dict(title='Validate-reconstruction'))

            recall = total_recall / validate_step
            precision = total_precision / validate_step
            accuracy = total_accuracy / validate_step
            f1 = total_f1 / validate_step
            viz.line([recall], [epoch + 1], win='Test-Recall', update='append',
                     opts=dict(title='Train-Recall'))
            viz.line([precision], [epoch + 1], win='Test-Precision', update='append',
                     opts=dict(title='Train-Precision'))
            viz.line([accuracy], [epoch + 1], win='Test-Accuracy', update='append',
                     opts=dict(title='Train-Accuracy'))
            viz.line([f1], [epoch + 1], win='Test-F1 Score', update='append',
                     opts=dict(title='Train-F1 Score'))
            print('F1 Score:', f1)
            print('Recall:', recall)
            print('Precision:', precision)
            print('Accuracy:', accuracy)
            log.write(
                f'Epoch-{epoch}:[[Accuracy,{accuracy}],[Precision,{precision}],[Recall,{recall}],[F1,{f1}]]\n')
            log.flush()
            if recall > best_recall:
                best_recall = recall
                best_recall_epoch = epoch
                torch.save(encoder.state_dict(), 'best_recall_encoder.pth')
                torch.save(decoder.state_dict(), 'best_recall_decoder.pth')
                torch.save(discriminator.state_dict(), 'best_recall_discriminator.pth')
            if precision > best_precision:
                best_precision = precision
                best_precision_epoch = epoch
                torch.save(encoder.state_dict(), 'best_precision_encoder.pth')
                torch.save(decoder.state_dict(), 'best_precision_decoder.pth')
                torch.save(discriminator.state_dict(), 'best_precision_discriminator.pth')
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
                torch.save(encoder.state_dict(), 'best_accuracy_encoder.pth')
                torch.save(decoder.state_dict(), 'best_accuracy_decoder.pth')
                torch.save(discriminator.state_dict(), 'best_accuracy_discriminator.pth')
            if f1 > best_f1:
                best_f1 = f1
                best_f1_epoch = epoch
                torch.save(encoder.state_dict(), 'best_f1_encoder.pth')
                torch.save(decoder.state_dict(), 'best_f1_decoder.pth')
                torch.save(discriminator.state_dict(), 'best_f1_discriminator.pth')
    # 记录最好情况
    log.write(f'best_recall_epoch:{best_recall_epoch},best_recall:{best_recall},'
              f'best_precision_epoch:{best_precision_epoch},best_precision:{best_precision},'
              f'best_accuracy_epoch:{best_accuracy_epoch},best_accuracy:{best_accuracy},'
              f'best_f1_epoch:{best_f1_epoch},best_f1:{best_f1},')
    log.flush()
    print(f'best_recall_epoch:{best_recall_epoch},best_recall:{best_recall},'
          f'best_precision_epoch:{best_precision_epoch},best_precision:{best_precision},'
          f'best_accuracy_epoch:{best_accuracy_epoch},best_accuracy:{best_accuracy},'
          f'best_f1_epoch:{best_f1_epoch},best_f1:{best_f1},')


if __name__ == '__main__':
    dataset = myDataset('MNIST', '.')
    feature_dim = 128
    encoder = MNISTEncoder(feature_dim)
    decoder = MNISTDecoder(feature_dim)
    discriminator = Discriminator()
    log = open('MNISTlog.txt', mode='w+')

    cfg = {
        'dataset': dataset,
        'device': 'cuda:0',
        'encoder': encoder,
        'decoder': decoder,
        'discriminator': discriminator,
        'init_lr': 1e-3,
        'weight_decay': 5e-5,
        'epochs': 100,
        'batch_size': 64,
        'positive': 0,
        'log': log,
        'feature_dim': 128,
    }
    train_test(cfg)
    log.close()
