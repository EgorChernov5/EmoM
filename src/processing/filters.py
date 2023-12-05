from pathlib import Path


def fltr_human_emo(image_path: Path | str, save_path: Path | str):
    """
    Saves the image if there is a human face on it and the emotion is correctly classified.

    :param image_path: the path to the image.
    :param save_path: the path to the folder where to save.
    """
    pass


# from PIL import Image
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import torch
# from torch.utils.data import DataLoader
#
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))
#
# # If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN(
#     image_size=160, margin=0, min_face_size=20,
#     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
#     device=device
# )
#
# # Create an inception resnet (in eval mode):
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
#
#
# def collate_fn(x):
#     return x[0]
#
#
# dataset = datasets.ImageFolder('../data/test_images')
# dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
#
# aligned = []
# names = []
# for x, y in loader:
#     x_aligned, prob = mtcnn(x, return_prob=True)
#     if x_aligned is not None:
#         print('Face detected with probability: {:8f}'.format(prob))
#         aligned.append(x_aligned)
#         names.append(dataset.idx_to_class[y])
