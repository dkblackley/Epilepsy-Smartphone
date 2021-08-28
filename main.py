import segmentation.segment
import utils
from video_dataset import data_set
import epilepsy_classification.training as ec
import config


__author__     = ["Daniel Blackley", "Jacob Carse"]
__copyright__  = "Copyright 2021, Cost-Sensitive Selective Classification for Skin Lesions"
__credits__    = ["Daniel Blackley", "Jacob Carse", "Stephen McKenna"]
__licence__    = "MIT"
__version__    = "0.0.1"
__maintainer__ = "Daniel Blackley"
__email__      = "dkblackley@dundee.ac.uk"
__status__     = "Development"

from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt

RESOLUTION_1 = 224
RESOLUTION_2 = 224

composed_train = transforms.Compose([
                                transforms.Resize((RESOLUTION_1, RESOLUTION_2)),
                                transforms.ToTensor(),
                                transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.RandomErasing(p=0.2, scale=(0.001, 0.005)),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                               ])

composed_test = transforms.Compose([
                                transforms.Resize((RESOLUTION_1, RESOLUTION_2)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                               ])

#print(utils.make_results_LATEX())


#train_set = data_set('datasets/', composed_train, composed_test, 'datasets/labels.csv', segmentation=None)
#train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=2, shuffle=True, num_workers=4)

"""
for i in range(1, 10):
    current_dir = f"models/{i}-FOLD_MODEL/"

    trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment='body', early_stop=False, device=device,
                         save_dir=current_dir, save_model=False)
    trainer.LOSO(7, debug=True)
    trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment='', early_stop=False, device=device,
                         save_dir=current_dir, save_model=False)
    trainer.LOSO(7, debug=True)
    trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment='face', early_stop=False, device=device,
                         save_dir=current_dir, save_model=False)
    trainer.LOSO(7, debug=True)


results = []

for i in range(0, 10):
    current_dir = f"models/{i}-FOLD_MODEL/"
    result = utils.read_from_csv(current_dir + "LOSO_face_RESULTS.csv", to_num=True)
    #result = utils.read_from_csv(current_dir + "LOSO_body_RESULTS.csv", to_num=True)
    #result = utils.read_from_csv(current_dir + "LOSO__RESULTS.csv", to_num=True)

    results.append(result)


results = np.ndarray(results)
results = np.average(results, axis=2)



utils.write_to_csv("models/averaged_face_results.csv", results)
#utils.write_to_csv("models/averaged_body_results.csv", results)
#utils.write_to_csv("models/averaged_results.csv", results)

#trainer.train(10)



#train_data = dataset.data_set("datasets/", composed_train, "datasets/labels.csv")
#train_set = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, sampler=weighted_train_sampler, shuffle=Fals"""

if __name__ == "__main__":
    # Loads the arguments from a config file and command line arguments.
    description = "Cost-Sensitive Selective Classification for Skin Lesions Using Bayesian Inference"
    arguments = config.load_arguments(description)

    train_set = data_set(arguments.dataset_dir, composed_train, composed_test, arguments.dataset_dir + 'labels.csv')

    if arguments.cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if arguments.setup:
        segmentation.segment.set_up_boxes(arguments.dataset_dir, device)

    if arguments.LOSO:
        for i in range(0, 10):
            current_dir = arguments.model_dir + f"{i}-FOLD_MODEL/"

            trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment=arguments.segment, early_stop=False,
                                 device=device, save_dir=current_dir, save_model=False)
            trainer.LOSO(arguments.epochs, debug=arguments.debug)
            trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment=arguments.segment, early_stop=False,
                                 device=device, save_dir=current_dir, save_model=False)
            trainer.LOSO(arguments.epochs, debug=arguments.debug)
            trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment=arguments.segment, early_stop=False,
                                 device=device, save_dir=current_dir, save_model=False)
            trainer.LOSO(arguments.epochs, debug=arguments.debug)
    else:
        trainer = ec.Trainer(train_set, 60, composed_train, composed_test, segment=arguments.segment, early_stop=False,
                             device=device, save_dir=arguments.model_dir, save_model=False)

        trainer.train(arguments.epochs, split=arguments.validation_split, debug=arguments.debug)



