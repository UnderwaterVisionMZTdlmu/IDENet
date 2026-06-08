
# Interaction-Driven Edge Crisping for Underwater Salient Object Detection

>Official repository for the paper:
>
>**Interaction-Driven Edge Crisping for Underwater Salient Object Detection**  
>Zetian Mi, Shuaiyong Jiang, Yuanyuan Li, Guanxi Li, Jiqing Zhang, Huibing Wang, Xianping Fu  
>
>You can find our paper on ([IEEE Transactions on Circuits and Systems for Video Technology](https://ieeexplore.ieee.org/document/11458694))

## Testing
1. Clone the repo
2. Download the checkpoint: [Baidu Netdisk](https://pan.baidu.com/s/1Yl4mxyDs0DImSFpnticSMA?pwd=0209)
3. Put the inputs to corresponding folders (raw and depth images)
4. Python infer.py (set "--checkpoint"to the path of the checkpoint, "--rgb_dir" to the path of the raw images, "--depth_dir" to the path of the depth images, "--output_dir" to the path of the output folder)
5. Find the result in "--output_dir"

## Results
You can download our results and pre-weighted models on [Baidu Netdisk](https://pan.baidu.com/s/1Yl4mxyDs0DImSFpnticSMA?pwd=0209).

## Citation
If you find this work useful, please cite:

```bibtex
@ARTICLE{11458694,
  author={Mi, Zetian and Jiang, Shuaiyong and Li, Yuanyuan and Li, Guanxi and Zhang, Jiqing and Wang, Huibing and Fu, Xianping},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Interaction-Driven Edge Crisping for Underwater Salient Object Detection}, 
  year={2026},
  pages={1-1},
  doi={10.1109/TCSVT.2026.3679543}
}
```