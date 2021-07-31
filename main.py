import utils
from torchvision import transforms
import dataset
import torch
from video_dataset import  data_set
import matplotlib.pyplot as plt
from tqdm import tqdm
import epilepsy_classification.training as ec

#utils.make_labels('temp_set/')

RESOLUTION_1 = 224
RESOLUTION_2 = 224

composed_train = transforms.Compose([
                                transforms.Resize((RESOLUTION_1, RESOLUTION_2)),
                                transforms.ToTensor(),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                               ])

#utils.change_videos_fps("datasets/")
#labels = utils.read_from_csv("datasets/labels.csv")
#utils.change_into_frames("datasets/", labels)

train_set = data_set('temp_set/', composed_train, 'temp_set/labels.csv', segmentation=None)
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=2, shuffle=True, num_workers=4)

trainer = ec.Trainer(train_set, 12, composed_train)

trainer.train(10)



#train_data = dataset.data_set("datasets/", composed_train, "datasets/labels.csv")
#train_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=weighted_train_sampler, shuffle=False)