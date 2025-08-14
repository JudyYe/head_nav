TODO:
- clean data, tar them, rename them, merge them if neccessary
- preprocess guide 
- install guide requirements.txt
- paths

python -m demo -m     \
    expname=final_train/srccsv_aug_branch0 \
    num=-1  vis=False ds=g1-kit name=\${ds}


python -m train -m \
    expname=dev/tmp \
    dataset=mix_all \
    dec.use_2_branch=0    