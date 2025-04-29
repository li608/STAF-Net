# STAF-Net
STAF-Net for Skeleton-Based Action Recognition


# Data Preparation

 - Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:
 
        -data\  
          -kinetics_raw\  
            -kinetics_train\
              ...
            -kinetics_val\
              ...
            -kinetics_train_label.json
            -keintics_val_label.json
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
            

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

 - Preprocess the data with
  
    `python data_gen/ntu_gendata.py`
    
    `python data_gen/kinetics-gendata.py.`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`



[https://drive.google.com/drive/folders/1co3n3OCUZOVIhW-T-VXUuQegLHRLRKzU?usp=sharing]: weights
# Training & Testing

Change the config file depending on what you want.


    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`
    `python main.py --config ./config/nturgbd-cross-view/test_joint.yaml`
