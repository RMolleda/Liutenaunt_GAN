from utils.models import define_discriminator, define_generator, define_gan, load_real_samples, generate_fake_samples, generate_latent_points, generate_real_samples, save_plot, summarize_performance, train
from utils.mining_data_tb import cargar_imagenes, augmentator
import os

cwd = os.getcwd()

# train model
def train_run():
    #augmentation of the uploads
    augmentator(num_samples=25000)
    # size of the latent space
    latent_dim = 100
    # create the discriminator
    d_model = define_discriminator()
    # create the generator
    g_model = define_generator(latent_dim)
    # create the gan
    gan_model = define_gan(g_model, d_model)
    # load image data
    dataset = load_real_samples(load_from=cargar_imagenes)
    train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=1000)

def delete_images_new(mydir):
    for f in os.listdir(mydir):
        if f.endswith(".jpg"):
            continue
        os.remove(os.path.join(mydir, f))

def delete_images_real(mydir_real):
    for f in os.listdir(mydir_real):
        if f.endswith(".jpeg"):
            continue
        os.remove(os.path.join(mydir_real, f))