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
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            normalize
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
