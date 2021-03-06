import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import kornia
import math
import segmentation_models_pytorch as smp


class MultiPatch(nn.Module):
    def __init__(self, pathes=None):
        super(MultiPatch, self).__init__()

        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()

        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()
        self.decoder_lv3 = Decoder()

        if pathes:
            self.encoder_lv1.load_state_dict(torch.load(pathes["encoder_lv1"]))
            self.encoder_lv2.load_state_dict(torch.load(pathes["encoder_lv2"]))
            self.encoder_lv3.load_state_dict(torch.load(pathes["encoder_lv3"]))
            self.decoder_lv1.load_state_dict(torch.load(pathes["decoder_lv1"]))
            self.decoder_lv2.load_state_dict(torch.load(pathes["decoder_lv2"]))
            self.decoder_lv3.load_state_dict(torch.load(pathes["decoder_lv3"]))

    def forward(self, x):
        H = x.size(2)
        W = x.size(3)

        images_lv2_1 = x[:,:,0:int(H/2),:]
        images_lv2_2 = x[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = self.decoder_lv2(feature_lv2)

        feature_lv1 = self.encoder_lv1(x + residual_lv2) + feature_lv2
        dehazed_image = self.decoder_lv1(feature_lv1)

        return dehazed_image


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        
    def forward(self,x):        
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x


class MultiPatchExtended(nn.Module):
    def __init__(self):
        super(MultiPatchExtended, self).__init__()

        self.encoder_lv1 = Encoder()
        self.encoder_lv2 = Encoder()
        self.encoder_lv3 = Encoder()

        self.decoder_lv1 = Decoder()
        self.decoder_lv2 = Decoder()
        self.decoder_lv3 = Decoder()

    def forward(self, x, f2=0, f3=0):
        H = x.size(2)
        W = x.size(3)

        images_lv2_1 = x[:,:,0:int(H/2),:]
        images_lv2_2 = x[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)
        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)
        residual_lv3_top = self.decoder_lv3(feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(feature_lv3_bot)

        if type(f3) != int:
            feature_lv3 = feature_lv3 + f3

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)
        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2) + feature_lv3
        residual_lv2 = self.decoder_lv2(feature_lv2)

        if type(f2) != int:
            feature_lv2 = feature_lv2 + f2

        feature_lv1 = self.encoder_lv1(x + residual_lv2) + feature_lv2
        dehazed_image = self.decoder_lv1(feature_lv1)

        return dehazed_image, feature_lv2, feature_lv3


class DoubleMultiPatchExtended(nn.Module):
    def __init__(self):
        super(DoubleMultiPatchExtended, self).__init__()

        self.first = MultiPatchExtended()
        self.second = MultiPatchExtended()

    def forward(self, x):
        x, a, b = self.first(x)
        y, c, d = self.second(x, a, b)
        return x, y


class DehazerColorer(nn.Module):
    def __init__(self):
        super(DehazerColorer, self).__init__()

        self.dehazer = MultiPatch()
        self.colorer = MultiPatch()

        self.colorer.encoder_lv1.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.colorer.encoder_lv2.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.colorer.encoder_lv3.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)

        self.colorer.decoder_lv1.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.colorer.decoder_lv2.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.colorer.decoder_lv3.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)

        self.final = nn.Conv2d(6, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        dehazed = self.dehazer(x)
        united = torch.cat([dehazed, x], 1)
        final = self.final(self.colorer(united))
        return final


class RefinementNet(nn.Module):
    def __init__(self):
        super(RefinementNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.decoder.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)

        self.final = nn.Conv2d(6, 3, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)

        return x


class invPixelShuffle(nn.Module):
    def __init__(self, ratio=2):
        super(invPixelShuffle, self).__init__()
        self.ratio = ratio

    def forward(self, tensor):
        ratio = self.ratio
        b = tensor.size(0)
        ch = tensor.size(1)
        y = tensor.size(2)
        x = tensor.size(3)
        assert x % ratio == 0 and y % ratio == 0, 'x, y, ratio : {}, {}, {}'.format(x, y, ratio)
        return tensor.view(b, ch, y // ratio, ratio, x // ratio, ratio).permute(0, 1, 3, 5, 2, 4).contiguous().view(b,-1,y // ratio,x // ratio)


#   Wide Activation Block
class WAB(nn.Module):
    def __init__(self,n_feats,expand=4):
        super(WAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * expand,3,1,1, bias=True),
            nn.BatchNorm2d(n_feats * expand),
            nn.ReLU(True),
            nn.Conv2d(n_feats* expand, n_feats , 3, 1, 1, bias=True),
            nn.BatchNorm2d(n_feats)
        )

    def forward(self, x):
        res = self.body(x).mul(0.2)+x
        return res


#   codes of UNet are modified from pix2pix
def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s_tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, nf=16):
        super(UNet, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=False, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        # dlayer6 = blockUNet(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)
        dlayer6 = blockUNet(nf*8, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer5 = blockUNet(nf*16, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer4 = blockUNet(nf*16, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer3 = blockUNet(nf*8, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx

        dlayer2 = blockUNet(nf*4, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = blockUNet(nf * 2, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1
        self.tail_conv = nn.Conv2d(nf*2, output_nc, 3,padding=1, bias=True)

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        dout1 = self.tail_conv(dout1)

        # dout1=torch.sigmoid(dout1)

        return dout1


class Hybrid(nn.Module):
    def __init__(self, path=None):
        super().__init__()

        self.d4u1=nn.Sequential(
            nn.Conv2d(3,16,3,1,1, bias=True),
            nn.BatchNorm2d(16),
            invPixelShuffle(4),
            nn.Conv2d(256,16,3,1,1, bias=True),
            nn.BatchNorm2d(16),
            nn.Sequential(*[WAB(16) for _ in range(3)]),
            nn.Conv2d(16, 256, 3, 1, 1, bias=True),
            nn.PixelShuffle(4),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 3, 3, 1, 1, bias=True)
        )

        self.tail = nn.Sequential(
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.Conv2d(9, 16, 3, 1, 1, bias=True),
            nn.Conv2d(16, 16, 3, 1, 1, bias=True),
            nn.Conv2d(16, 3, 3, 1, 1, bias=True)
        )

        self.dehazer = nn.DataParallel(DehazerColorer().cuda())

        if path:
            self.dehazer.load_state_dict(torch.load(path))

    def forward(self, x):
        b,c,h,w=x.shape
        mod1=h%64
        mod2=w%64
        if(mod1):
            down1=64-mod1
            x=F.pad(x,(0,0,0,down1),"reflect")
        if(mod2):
            down2=64-mod2
            x=F.pad(x,(0,down2,0,0),"reflect")

        refined = self.d4u1(x)
        dehazed = self.dehazer(x)
        x = self.tail(torch.cat([dehazed, refined, x], 1))

        if(mod1):x=x[:,:,:-down1,:]
        if(mod2):x=x[:,:,:,:-down2]

        return x


class SuperHybrid(nn.Module):
    def __init__(self, path1=None):
        super(SuperHybrid, self).__init__()

        self.first = nn.DataParallel(DehazerColorer().cuda())

        if path1:
            self.first.load_state_dict(torch.load(path1))

        self.colorer = nn.Sequential(
            torchvision.models.mobilenet_v2(pretrained=True),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )


    def forward(self, x):
        with torch.no_grad():
            fst = self.first(x) + 0.5

        colors = self.colorer(x)
        fst = fst.clamp(1/255, 1)
        btch = colors.shape[0]

        # r, g, b, gamma = colors[:,0].reshape(btch, 1, 1), colors[:,1].reshape(btch, 1, 1), colors[:,2].reshape(btch, 1, 1), colors[:,3].reshape(btch, 1, 1, 1)

        # r, g, b, gamma = torch.abs(r), torch.abs(g), torch.abs(b), torch.abs(gamma)

        # y = torch.stack([fst[:,0,::] / r, fst[:,1,::] / g, fst[:,2,::] / b], 1)

        # gamma = abs(colors[:,0].reshape(btch, 1, 1, 1))

        # y = torch.pow(fst, 1 / gamma)

        # print([float(j) for j in gamma.reshape(btch)])

        # r, g, b, sat = abs(colors[:,0].reshape(btch, 1, 1)), abs(colors[:,1].reshape(btch, 1, 1)), abs(colors[:,2].reshape(btch, 1, 1)), abs(colors[:,3].reshape(btch))

        # print(sum([float(j) for j in r.reshape(btch)]) / btch)
        # print(sum([float(j) for j in g.reshape(btch)]) / btch)
        # print(sum([float(j) for j in b.reshape(btch)]) / btch)
        # print(sum([float(j) for j in sat.reshape(btch)]) / btch)

        # y = torch.stack([torch.pow(fst[:,0,::], 1/r), torch.pow(fst[:,1,::], 1/g), torch.pow(fst[:,2,::], 1/b)], 1)

        sat = abs(colors[:,0].reshape(1))

        y = kornia.enhance.AdjustSaturation(sat)(fst)

        return y - 0.5


class SatRGB(nn.Module):
    def __init__(self):
        super(SatRGB, self).__init__()

        self.begin = nn.Sequential(
            nn.Conv2d(6, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.ReLU()
        )

        self.mobile = torchvision.models.mobilenet_v2(pretrained=True)

        self.end = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(1000, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
        )


    def forward(self, x, original, test=False):

        x = x + 0.5
        colors = self.end(self.mobile(self.begin(torch.cat([x, original], 1))))
        x = x.clamp(1/255, 1)
  
        batch = x.shape[0]

        hue, con, gamma, sat = abs(colors[:,0].reshape(batch)), abs(colors[:,1].reshape(batch)), abs(colors[:,2].reshape(batch)), abs(colors[:,3].reshape(batch))

        print(sum([float(j) for j in gamma]) / batch)
        print(sum([float(j) for j in con]) / batch)
        print(sum([float(j) for j in hue]) / batch)
        print(sum([float(j) for j in sat]) / batch)

        # x = torch.stack([torch.pow(x[:,0,::], 1/r), torch.pow(x[:,1,::], 1/g), torch.pow(x[:,2,::], 1/b)], 1)
        if test:
            x1 = kornia.enhance.AdjustGamma(gamma)(x)
            x2 = kornia.enhance.AdjustContrast(con)(x1)
            x3 = kornia.enhance.AdjustHue(hue)(x2)
            x4 = kornia.enhance.AdjustSaturation(sat)(x3)

            return x - 0.5, x1 - 0.5, x2 - 0.5, x3 - 0.5, x4 - 0.5

        else:
            x = kornia.enhance.AdjustGamma(gamma)(x)
            x = kornia.enhance.AdjustContrast(con)(x)
            x = kornia.enhance.AdjustHue(hue)(x)
            x = kornia.enhance.AdjustSaturation(sat)(x)

            return x - 0.5


class IndianColor(nn.Module):
    def __init__(self, path=None):
        super(IndianColor, self).__init__()

        self.indian = MultiPatch()

        if path:
            self.indian.load_state_dict(torch.load(path))

        self.rgbsat = SatRGB()


    def forward(self, x, test=False):
        
        dehazed = self.indian(x)
        enhanced = self.rgbsat(dehazed, x, test)

        return enhanced


class GrayScale(nn.Module):
    def __init__(self, path=None):
        super(GrayScale, self).__init__()

        self.indian = MultiPatch()

    def forward(self, x):
        
        dehazed = self.indian(x)

        return dehazed


class ColorEstimator(nn.Module):
    def __init__(self, path=None):
        super(ColorEstimator, self).__init__()

        self.gray_dehazer = GrayScale().cuda()

        if path:
            self.gray_dehazer.load_state_dict(torch.load(path))

        self.colorer = MultiPatch()
        self.colorer.encoder_lv1.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.colorer.encoder_lv2.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.colorer.encoder_lv3.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)

        self.colorer.decoder_lv1.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.colorer.decoder_lv2.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.colorer.decoder_lv3.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)

        self.final = nn.Conv2d(6, 3, kernel_size=3, padding=1)

    def forward(self, x):
        
        with torch.no_grad():
            dehazed = self.gray_dehazer(x)

        color = self.final(self.colorer(torch.cat([dehazed, x], 1)))

        # dehazed_image = color * dehazed / torchvision.transforms.functional.rgb_to_grayscale(color, num_output_channels=3)

        return color, dehazed


class HueEstimator(nn.Module):
    def __init__(self, path=None):
        super(HueEstimator, self).__init__()

        self.gray_dehazer = GrayScale().cuda()

        self.recognizer = MultiPatch()

        self.recognizer.encoder_lv1.layer1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.recognizer.encoder_lv2.layer1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.recognizer.encoder_lv3.layer1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)

        self.recognizer.decoder_lv1.layer24 = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.recognizer.decoder_lv2.layer24 = nn.Conv2d(32, 4, kernel_size=3, padding=1)
        self.recognizer.decoder_lv3.layer24 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        if path:
            self.gray_dehazer.load_state_dict(torch.load(path))

        self.final = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def forward(self, x):
        
        with torch.no_grad():
            dehazed = self.gray_dehazer(x)

        hue = self.final(self.recognizer(torch.cat([dehazed[:,0,::].unsqueeze(1), x], 1)))

        return hue


class MonsterBlock(nn.Module):
    def __init__(self):
        super(MonsterBlock, self).__init__()

        # self.worker = MultiPatch()
        self.worker = nn.Sequential(
            Encoder(),
            Decoder()
        )

        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        
        x = self.worker(x)
        x = self.final(x)

        return x


class Monster(nn.Module):
    def __init__(self):
        super(Monster, self).__init__()

        self.hue = nn.Sequential(
            MultiPatch(),
            nn.ReLU(),
            nn.Conv2d(3, 2, kernel_size=3, padding=1)
        )
        self.saturation = nn.Sequential(
            MultiPatch(),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, padding=1)
        )
        self.value = nn.Sequential(
            MultiPatch(),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):

        hue = self.hue(x)
        saturation = self.saturation(x)
        value = self.value(x)

        hue = hue / (hue[:,0,:,:].unsqueeze(1)**2 + hue[:,1,:,:].unsqueeze(1)**2)**0.5

        return hue, saturation, value

    @staticmethod
    def xy2hue(tns):
        x, y = torch.chunk(tns, 2, 1)

        mask1 = torch.logical_and(x > 0, y >= 0) * 0
        mask2 = torch.logical_and(x < 0, y >= 0) * math.pi
        mask3 = torch.logical_and(x < 0, y < 0) * math.pi
        mask4 = torch.logical_and(x > 0, y < 0) * 2 * math.pi

        result = torch.arctan(y / x) + mask1 + mask2 + mask3 + mask4

        return result

    @staticmethod
    def hue2xy(angle):
        return torch.cat([torch.cos(angle), torch.sin(angle)], 1)


class HueTrainer(nn.Module):
    def __init__(self, path1, path2):
        super(HueTrainer, self).__init__()

        self.hue = MultiPatch()

        self.hue.encoder_lv1.layer1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.hue.encoder_lv2.layer1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)
        self.hue.encoder_lv3.layer1 = nn.Conv2d(9, 32, kernel_size=3, padding=1)

        self.hue.decoder_lv1.layer24 = nn.Conv2d(32, 9, kernel_size=3, padding=1)
        self.hue.decoder_lv2.layer24 = nn.Conv2d(32, 9, kernel_size=3, padding=1)
        self.hue.decoder_lv3.layer24 = nn.Conv2d(32, 9, kernel_size=3, padding=1)

        self.saturation = nn.Sequential(
            MultiPatch(),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, padding=1)
        )
        self.value = nn.Sequential(
            MultiPatch(),
            nn.ReLU(),
            nn.Conv2d(3, 1, kernel_size=3, padding=1)
        )

        # self.saturation.load_state_dict(torch.load(path1))
        # self.value.load_state_dict(torch.load(path2))

        self.tail = nn.Conv2d(9, 3, kernel_size=3, padding=1)

    def forward(self, x):

        with torch.no_grad():
            val = self.value(x)
            sat = self.saturation(x)

        result = self.tail(self.hue(torch.cat([x, val, val, val, sat, sat, sat], 1)))

        return result


class Titan(nn.Module):
    def __init__(self, path):
        super(Titan, self).__init__()

        self.first = HueTrainer(1, 1)
        self.first.load_state_dict(torch.load(path))

        self.second = MultiPatch()

        self.second.encoder_lv1.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.second.encoder_lv2.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.second.encoder_lv3.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)

        self.second.decoder_lv1.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.second.decoder_lv2.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.second.decoder_lv3.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)

        self.tail = nn.Conv2d(6, 3, kernel_size=3, padding=1)

    def forward(self, x):

        with torch.no_grad():
            val = self.first(x)

        result = self.tail(self.second(torch.cat([x, val], 1)))

        return result


class MultiPatchSMP(nn.Module):
    def __init__(self, inp=3):
        super(MultiPatchSMP, self).__init__()

        out_channels = smp.encoders.dpn.dpn_encoders["dpn92"]["params"]["out_channels"]
        out_channels = [inp] + list(out_channels[1:])
        params = smp.encoders.dpn.dpn_encoders["dpn92"]["params"]
        params["inp"] = inp
        # pretrained = smp.encoders.dpn.dpn_encoders["dpn92"]["pretrained_settings"]

        self.encoder_lv1 = smp.encoders.dpn.DPNEncoder(**params)
        self.encoder_lv2 = smp.encoders.dpn.DPNEncoder(**params)
        self.encoder_lv3 = smp.encoders.dpn.DPNEncoder(**params)

        # self.encoder_lv1.load_state_dict(torch.utils.model_zoo.load_url(pretrained['imagenet+5k']["url"]))
        # self.encoder_lv2.load_state_dict(torch.utils.model_zoo.load_url(pretrained['imagenet+5k']["url"]))
        # self.encoder_lv3.load_state_dict(torch.utils.model_zoo.load_url(pretrained['imagenet+5k']["url"]))

        self.decoder_lv1 = smp.unetplusplus.decoder.UnetPlusPlusDecoder(out_channels, out_channels[:-1][::-1], n_blocks=len(out_channels)-1)
        self.decoder_lv2 = smp.unetplusplus.decoder.UnetPlusPlusDecoder(out_channels, out_channels[:-1][::-1], n_blocks=len(out_channels)-1)
        self.decoder_lv3 = smp.unetplusplus.decoder.UnetPlusPlusDecoder(out_channels, out_channels[:-1][::-1], n_blocks=len(out_channels)-1)

        self.tail = nn.Conv2d(inp, 3, 3, 1, 1)

        # self.decoder_lv1 = smp.unet.decoder.UnetDecoder(out_channels, out_channels[:-1][::-1], n_blocks=len(out_channels)-1)
        # self.decoder_lv2 = smp.unet.decoder.UnetDecoder(out_channels, out_channels[:-1][::-1], n_blocks=len(out_channels)-1)
        # self.decoder_lv3 = smp.unet.decoder.UnetDecoder(out_channels, out_channels[:-1][::-1], n_blocks=len(out_channels)-1)

    
    def forward(self, x):
        b,c,h,w=x.shape
        mod1=h%64
        mod2=w%64
        if(mod1):
            down1=64-mod1
            x=F.pad(x,(0,0,0,down1),"reflect")
        if(mod2):
            down2=64-mod2
            x=F.pad(x,(0,down2,0,0),"reflect")

        H = x.size(2)
        W = x.size(3)

        images_lv2_1 = x[:,:,0:int(H/2),:]
        images_lv2_2 = x[:,:,int(H/2):H,:]
        images_lv3_1 = images_lv2_1[:,:,:,0:int(W/2)]
        images_lv3_2 = images_lv2_1[:,:,:,int(W/2):W]
        images_lv3_3 = images_lv2_2[:,:,:,0:int(W/2)]
        images_lv3_4 = images_lv2_2[:,:,:,int(W/2):W]

        feature_lv3_1 = self.encoder_lv3(images_lv3_1)
        feature_lv3_2 = self.encoder_lv3(images_lv3_2)
        feature_lv3_3 = self.encoder_lv3(images_lv3_3)
        feature_lv3_4 = self.encoder_lv3(images_lv3_4)

        # print([x.shape for x in feature_lv3_1])

        feature_lv3_top = [torch.cat((feature_lv3_1[i], feature_lv3_2[i]), 3) for i in range(len(feature_lv3_1))]
        feature_lv3_bot = [torch.cat((feature_lv3_3[i], feature_lv3_4[i]), 3) for i in range(len(feature_lv3_1))]
        feature_lv3 = [torch.cat((feature_lv3_top[i], feature_lv3_bot[i]), 2) for i in range(len(feature_lv3_1))]

        residual_lv3_top = self.decoder_lv3(*feature_lv3_top)
        residual_lv3_bot = self.decoder_lv3(*feature_lv3_bot)

        feature_lv2_1 = self.encoder_lv2(images_lv2_1 + residual_lv3_top)
        feature_lv2_2 = self.encoder_lv2(images_lv2_2 + residual_lv3_bot)


        feature_lv2 = [torch.cat((feature_lv2_1[i], feature_lv2_2[i]), 2) + feature_lv3[i] for i in range(len(feature_lv2_1))]
        residual_lv2 = self.decoder_lv2(*feature_lv2)

        feature_lv1 = self.encoder_lv1(x + residual_lv2)
        feature_lv1 = [feature_lv1[i] + feature_lv2[i] for i in range(len(feature_lv1))]
        dehazed_image = self.decoder_lv1(*feature_lv1)

        dehazed_image = self.tail(dehazed_image)

        if(mod1):dehazed_image=dehazed_image[:,:,:-down1,:]
        if(mod2):dehazed_image=dehazed_image[:,:,:,:-down2]

        return dehazed_image


class SuperSat(nn.Module):
    def __init__(self, path=None):
        super(SuperSat, self).__init__()

        self.val_estimator = MultiPatchSMP()
        self.sat_estimator = MultiPatch()

        self.sat_estimator.encoder_lv1.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.sat_estimator.encoder_lv2.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.sat_estimator.encoder_lv3.layer1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)

        self.sat_estimator.decoder_lv1.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.sat_estimator.decoder_lv2.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)
        self.sat_estimator.decoder_lv3.layer24 = nn.Conv2d(32, 6, kernel_size=3, padding=1)

        if path:
            self.val_estimator.load_state_dict(torch.load(path))

        self.tail = nn.Conv2d(6, 1, kernel_size=3, padding=1)

    def forward(self, x):

        with torch.no_grad():
            val = self.val_estimator(x)

        result = self.tail(self.sat_estimator(torch.cat([x[:,0,:,:].unsqueeze(1), val, x[:,1,:,:].unsqueeze(1), val, x[:,2,:,:].unsqueeze(1), val], 1)))

        return result, val


class SuperHue(nn.Module):
    def __init__(self):
        super(SuperHue, self).__init__()

        self.hue_estimator = MultiPatchSMP(6)

    def forward(self, x):

        result = self.hue_estimator(x)

        return result


class ColorSegmentator(nn.Module):
    def __init__(self):
        super(ColorSegmentator, self).__init__()

        self.segmentator = smp.unetplusplus.UnetPlusPlus(in_channels=6, classes=3, decoder_attention_type="scse", activation="softmax")

    def forward(self, x):
        b,c,h,w=x.shape
        mod1=h%64
        mod2=w%64
        if(mod1):
            down1=64-mod1
            x=F.pad(x,(0,0,0,down1),"reflect")
        if(mod2):
            down2=64-mod2
            x=F.pad(x,(0,down2,0,0),"reflect")

        result = self.segmentator(x)

        if(mod1):result=result[:,:,:-down1,:]
        if(mod2):result=result[:,:,:,:-down2]

        return result


class GrandCombine(nn.Module):
    def __init__(self):
        super(GrandCombine, self).__init__()

        self.boom = MultiPatchSMP(6)

    def forward(self, x):

        return self.boom(x)