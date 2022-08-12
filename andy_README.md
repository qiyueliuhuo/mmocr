
########## adny event log ##########

python tools/data/textdet/totaltext_converter.py /home/andy/workplace/machineLearing/scientific-experiments/mmocr/data/totaltext -o /home/andy/workplace/machineLearing/scientific-experiments/mmocr/data/totaltext --split-list training test


git merge origin/main  


########  2022.07.15 log  ##########

python tools/test.py configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth --eval hmean-ic13 --show-score-thr 0.7 --eval-option rank_list=result_analyze/eval_hmean-ic13_thr_0.7_result.json

nohup python tools/train.py  configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py 

python tools/analyze_logs.py plot_curve  result_analyze/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.log.json  --out result_analyze/loss --legend loss


如何得到模型在某个数据集中推理效果较差的图像及推理结果？

1. 先通过 `python tools/test.py configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth --eval hmean-ic13 --show-score-thr 0.7 --eval-option rank_list=result_analyze/eval_hmean-ic13_thr_0.7_result.json
` 命令，推理得到某个指标下推理效果排序（效果越差越考前），保存为 `result_analyze/eval_hmean-ic13_thr_0.7_result.json` 文件。
2. 将推理结果较差的图像路径保存在 `data/icdar2015/test_list.txt` 文件中。
3. 最后，通过命令 `python tools/det_test_imgs.py data/icdar2015 data/icdar2015/test_lists.txt configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py checkpoints/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth ` 得到这些图像的推理结果并保存在 `results` 目录下。

第3步，修改命令为 `python tools/det_test_imgs.py data/icdar2015/imgs data/icdar2015/test_lists.txt configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py checkpoints/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth --compare --annotations_root data/icdar2015/annotations/test/`
增加了 `--compare`等参数，表示生成与ground truth 对比图像。

