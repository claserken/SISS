import torch

class TShirtClassifier:
    @staticmethod
    def get_tshirt_frequency(imgs, tshirt_img, threshold=10):
        """
        Input: imgs is a torch tensor of shape [N, C, H, W] with range [0, 1] 
               and tshirt_img is a torch tensor of shape [C, H, W] with the same range.
               threshold is a float value used for determining if an image contains the t-shirt.
        Output: float with frequency of tshirt_img amongst N images.
        """
        
        # Flatten the tshirt_img for comparison
        tshirt_flattened = tshirt_img.view(-1)
        
        # Reshape the imgs tensor to [N, C*H*W] for batch comparison
        imgs_flattened = imgs.view(imgs.size(0), -1)
        
        # Compute the L2 distance (Euclidean distance) between each image and the tshirt_img
        l2_distances = torch.norm(imgs_flattened - tshirt_flattened, dim=1)
        
        # Determine which images have a distance below the threshold
        matches = l2_distances < threshold
        
        # Calculate the frequency of images that contain the t-shirt
        frequency = matches.float().mean().item()
        
        return frequency, matches
