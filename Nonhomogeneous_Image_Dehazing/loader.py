import torch
import models


model1 = models.MultiPatchExtended()

model1.encoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/encoder_lv1.pkl"))
model1.encoder_lv2.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/encoder_lv2.pkl"))
model1.encoder_lv3.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/encoder_lv3.pkl"))

model1.decoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/decoder_lv1.pkl"))
model1.decoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/decoder_lv2.pkl"))
model1.decoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/decoder_lv3.pkl"))

model2 = models.MultiPatchExtended()

model2.encoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/encoder_lv1.pkl"))
model2.encoder_lv2.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/encoder_lv2.pkl"))
model2.encoder_lv3.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/encoder_lv3.pkl"))

model2.decoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/decoder_lv1.pkl"))
model2.decoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/decoder_lv2.pkl"))
model2.decoder_lv1.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/models/ind_v0/decoder_lv3.pkl"))


model = models.DoubleMultiPatchExtended()
model.first.load_state_dict(model1.state_dict())
model.second.load_state_dict(model2.state_dict())

model.cuda()
model = torch.nn.DataParallel(model)

print(model.state_dict()['module.second.decoder_lv3.layer24.bias'])

torch.save(model.state_dict(), "/home/fedynyak/NTIRE2021/pre_trained_combined.pkl")
model.load_state_dict(torch.load("/home/fedynyak/NTIRE2021/pre_trained_combined.pkl"))

print(model.state_dict()['module.second.decoder_lv3.layer24.bias'])

print("done")