TODO:
- clean data, tar them, rename them, merge them if neccessary
- install guide requirements.txt download data path
- paths

python -m demo -m     \
    expname=release/kitchen \
    num=-1  vis=False ds=kitchen_r[,lab_r] name=\${ds}


python -m train -m \
    expname=dev/tmp \
    dataset=mix_all \
    dec.use_2_branch=0    