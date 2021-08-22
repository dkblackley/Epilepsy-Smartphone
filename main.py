import utils
from torchvision import transforms
import torch
from video_dataset import  data_set
import matplotlib.pyplot as plt
from tqdm import tqdm
import epilepsy_classification.training as ec
import segmentation.segment as segment

#utils.make_labels('temp_set/')

RESOLUTION_1 = 224
RESOLUTION_2 = 224

#utils.change_videos_fps("datasets/")

#segment.set_up_boxes("datasets/")

composed_train = transforms.Compose([
                                transforms.Resize((RESOLUTION_1, RESOLUTION_2)),
                                transforms.ToTensor(),
                                #transforms.ColorJitter(brightness=0.1, contrast=0.1),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                               ])

composed_test = transforms.Compose([
                                transforms.Resize((RESOLUTION_1, RESOLUTION_2)),
                                transforms.ToTensor(),
                                #transforms.ColorJitter(brightness=0.1, contrast=0.1),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                # call helper.get_mean_and_std(data_set) to get mean and std
                                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                               ])


#labels = utils.read_from_csv("datasets/labels.csv")
#utils.change_into_frames("datasets/", labels)

train_set = data_set('datasets/', composed_train, composed_test, 'datasets/labels.csv', segmentation=None)
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=2, shuffle=True, num_workers=4)

trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment='body', early_stop=False)
trainer2 = ec.Trainer(train_set, 60, composed_train, composed_test, segment='face', early_stop=False)
trainer3 = ec.Trainer(train_set, 60, composed_train, composed_test, segment='', early_stop=False)


trainer.LOSO(10, debug=True)
trainer2.LOSO(10, debug=True)
trainer3.LOSO(10, debug=True)

#trainer.train(10)



#train_data = dataset.data_set("datasets/", composed_train, "datasets/labels.csv")
#train_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=weighted_train_sampler, shuffle=False)