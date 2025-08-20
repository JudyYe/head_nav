set -x

# download model to weights/
mkdir -p data/
gdown 1qFSSapd3qftU8wVawgvjLBWBhQ-kRntf -O data/
tar -xf data/data_h.tar.gz -C data/

gdown 1Fl95Gu5avadOp0DVmFuY87UvJ8nNyg1H -O data/
tar -xf data/data_r.tar.gz -C data/


# adt: uploading