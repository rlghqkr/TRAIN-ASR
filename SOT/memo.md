/home/cssong/workspace/TRAIN-ASR/SOT
지금 이 리포는 포함해야할게
1. 데이터 전처리 : /home/cssong/workspace/TRAIN-ASR/data_preproc
- kiho, yejin 각 작업자가 본인 디렉토리에서 전처리 작업 코드 작성해놨음. (일단 벤치마크 데이터셋만 전처리)
- 원천데이터 경로 : /data/ASR/RAW
- 벤치마크(silver) 경로 : /data/ASR/BENCHMARK/SILVER
- 벤치마크(gold) 경로 : /data/ASR/BENCHMARK/GOLD (실버 데이터셋이 너무 많아서 샘플링 함)
- 전처리 작업 코드 경로(벤치마크): 작업자 두명이 작업해서 나뉘어져있음.
  - /home/kiho/workspace/TRAIN-ASR/notebooks/RAW_to_SILVER
  - /home/yejin/workspace/TRAIN-ASR/notebooks/EDA
- kiho, yejin 작업물을 정리해서 /home/cssong/workspace/TRAIN-ASR/data_preproc 로 일원화하려고해.
2. 학습 : 
- /home/cssong/workspace/TRAIN-ASR/scripts/train.py
- /home/cssong/workspace/TRAIN-ASR/scripts/train.sh
- 학습 중에는 밸리데이션셋
- 학습 완료후에는 벤치마크에 대해서 평가 자동으로 돌도록.. train.sh 스크립트 마지막에 eval.sh 돌리게 하기
- 학습시에는 wandb 로 볼수 있도록

3. 평가(벤치마크) : 
- eval.sh 스크립트로 만들기
- 옵션은 적절히.. 각 벤치마크 경로 지정하면 좋을듯. 일단 GOLD의 경로로 해줘 (그런데 SILVER로 돌릴수도있음. 자주그런건아니고 보통 GOLD로 하는데, 가끔 silver 로 할거니까 silver, gold 로 옵션을 넣는다기보다는 그냥 벤치마크 데이터셋명, 경로 이렇게만하면 경로에 알아서 silver 를 gold로 수정하면 될듯)

4. 관련문서: /home/cssong/workspace/TRAIN-ASR/docs
5. 설명 : /home/cssong/workspace/TRAIN-ASR/GUIDELINE

여기에 docs, SOT를 최신화하려면 작성하면 좋을게 뭐가 있을까?
docs, SOT를 이원화해서 가져가는게 좋을까?
