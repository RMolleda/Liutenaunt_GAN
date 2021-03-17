import os

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