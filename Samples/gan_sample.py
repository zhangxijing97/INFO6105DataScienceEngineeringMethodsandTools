from pygan._torch.gan_image_generator import GANImageGenerator
from matplotlib import pyplot as plt

#Create GAN Model
gan_image_generator = GANImageGenerator(
    dir_list=["C:/Sources/DataScienceLecture/lec11/data",],  # list of path to image directories
    width=256,
    height=256,
    channel=1,
    batch_size=40,
    learning_rate=1e-06
)

#Train Model
gan_image_generator.learn(
    iter_n=3,
    k_step=3
)

# Draw samples from fake distribution
arr = gan_image_generator.GAN.generative_model.draw()

# Visualize results
for i in range(5):
    image2d = arr[i, 0, :, :]
    img = image2d * 255  # scale to 0-255 (grayscale)
    img =img.detach().numpy() # convert to a numpy ndarray
    plt.imshow(img, cmap='gray')
    plt.show()
