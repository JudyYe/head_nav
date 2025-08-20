set -x

# download model to weights/
mkdir -p weights
gdown 17NCkl2YISKJZXi091DIRojCiNJgYE-56 -O weights/release_labroom.tar.gz
tar -xf weights/release_labroom.tar.gz -C weights/

gdown 1mNBpCy1eGo6XsnBEgAuxNiUnHZkFzmoa -O weights/release_kitchen.tar.gz
tar -xf weights/release_kitchen.tar.gz -C weights/




