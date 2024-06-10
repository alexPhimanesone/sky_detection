from os.path import join as opj
import mmcv
from mmseg.apis import init_model, inference_model, show_result_pyplot
from crop_around_disk import estimate_radius, crop_around_disk

data_dir = "../data/"
pprad_path = opj(data_dir, "pprad.yml")
img_cropped_path = opj(data_dir, "img_cropped.jpg")
output_path = opj(data_dir, "output.jpg")

config_path = opj("./unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py")
checkpoint_path = opj(data_dir, "model.pth")


def inference(calibration_path, img_path):
    """
    Return a torch tensor of shape (1, H, W)
    """
    
    # Initiate the model
    model = init_model(config_path, checkpoint_path, 'cpu')

    # Crop the image
    estimate_radius(calibration_path)
    img = mmcv.imread(img_path)
    img_cropped = crop_around_disk(pprad_path, img)
    mmcv.image.imwrite(img_cropped, img_cropped_path)

    # Run inference
    result = inference_model(model, img_cropped_path)
    
    # Create a visual representation and save it
    #show_result_pyplot(model, img_cropped_path, result, out_file=output_path, show=False)
    
    return result.pred_sem_seg.data
