import torch.nn as nn
import segmentation_models_pytorch as smp

class BCEDiceLoss(nn.Module):

    __name__ = 'bce_dice_loss'

    def __init__(self,  lambda_dice=1.0, lambda_bce=1.0):
        super(BCEDiceLoss, self).__init__()

        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary")
        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce


    def forward(self, y_pr, y_gt):
        dice = self.dice(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce* bce)
