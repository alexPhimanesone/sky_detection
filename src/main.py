from inference import inference
import matplotlib.pyplot as plt


calibration_path = "Z:/Work/Projects/solar_estimation/data/calibration_0425-1742/calibs/calib0/calib0.yml"
img_path = "Z:/Work/Projects/solar_estimation/data/dataset/pics_0606-1153/test/pic0008002.jpg"
output = inference(calibration_path, img_path)

# Convert to numpy
output = output[0].numpy()

# Display the predicted mask
plt.imshow(output)
plt.show()
