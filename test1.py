import models
import torch


model = models.Hybrid("/home/fedynyak/NTIRE2021/models/colorer_dehazer_v1/mutipatch_v-3.pkl")
torch.save(model.state_dict(), "hybrid_v0.pkl")