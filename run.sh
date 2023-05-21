CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch \
                        --nproc_per_node=8\
                        contras.py \
                        --batch_size 32\
                        --output ./output \
                        --cache_dir BioclinicalBERT_Path \
                        --img_dir Image_Path  \
                        --ann_path Annotation_Path \
                        --epochs 200 \
                        --out_dim 512\
                        --way token\
                        --soft_label\
                        --threshold 0.98\
                        --threshold1 0.97\
                        --lr 4e-4\
                        --alpha_weight 0.50\
                        --comment sat_convirt \

                       


                        



                        
                        