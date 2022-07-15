
########## adny event log ##########

python tools/data/textdet/totaltext_converter.py /home/andy/workplace/machineLearing/scientific-experiments/mmocr/data/totaltext -o /home/andy/workplace/machineLearing/scientific-experiments/mmocr/data/totaltext --split-list training test


git merge origin/main  


########  2022.07.15 log  ##########

python tools/test.py configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py https://download.openmmlab.com/mmocr/textdet/dbnet/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.pth --eval hmean-ic13 --show-score-thr 0.7 --eval-option rank_list=result_analyze/eval_hmean-ic13_thr_0.7_result.json

nohup python tools/train.py  configs/textdet/dbnetpp/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015.py 

python tools/analyze_logs.py plot_curve  result_analyze/dbnetpp_r50dcnv2_fpnc_1200e_icdar2015-20220502-d7a76fff.log.json  --out result_analyze/loss --legend loss