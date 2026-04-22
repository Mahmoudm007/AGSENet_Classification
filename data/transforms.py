import torch
from torchvision import transforms


def get_transforms(image_size: int, split: str = 'train'):
    """
    Returns the appropriate transform pipeline for the specified split.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.88, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.35),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.15),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.16, contrast=0.18, saturation=0.12, hue=0.03)],
                p=0.55,
            ),
            transforms.RandomGrayscale(p=0.06),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.12),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(
                p=0.15,
                scale=(0.02, 0.08),
                ratio=(0.5, 2.0),
                value='random',
            ),
        ])
    else:
        # For validation and test
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])

def get_inverse_transform():
    """
    Returns a transform to un-normalize tensors back to PIL images (useful for visualization).
    """
    # inverse of normalize: x * std + mean
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    def denorm(tensor):
        t = tensor.clone().detach() * std.to(tensor.device) + mean.to(tensor.device)
        t = torch.clamp(t, 0, 1)
        return transforms.ToPILImage()(t)
        
    return denorm
