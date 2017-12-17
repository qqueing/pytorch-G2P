# (semi) Grapheme-to-Phoneme (G2P) - seq2seq model using PyTorch for Korean
한국어 문자를 위한 G2P seq2seq 알고리즘의 구현 코드의 한국어 용입니다. 아주 minor 한 개선으로 기존 코드 살짝 수정한 수준입니다.

이 코드는 목적은 기존의 G2P seq2seq 모델을 이용해서,
한국어 자,모음 기준으로 표준어 형태의 문자열을 발음 나는 문자열로 변환 혹은 반대로 발음 문자열을 표준어 형태의 문자열로 변환하는 데에 있습니다. 
이것은 캐릭터 레벨의 CTC를 학습하기 위해, 문자열을 발음 형태로 변환하거나 혹은 언어모델이 결합되지 않은 CTC 디코딩의 결과물을 문자열로 변환하는 용도로 고려되었습니다.
정확하게는 G2P라 할수 없지만 표음문자인 한국어 특성상 semi G2P라고 표현하였습니다. 

또한 이 코드는 코드 원래의 목적대로 DB만 있다면 IPA 형식의 G2P로 사용하는 것에도 크게 문제가 없을 것으로 예측 됩니다.(계획 중입니다)

이러한 코드가 같은 분야를 연구하는 분들에게 도움이 될 수 있었으면 합니다.
  

## Credits
Original paper:
- Luong's paper:
```
@article{
  author    = {Minh-Thang Luong, Hieu Pham and Christopher D. Manning},
  title     = {Effective Approaches to Attention-based Neural Machine Translation},
  journal   = {CoRR},
  volume    = {abs/1508.04025},
  year      = {2015},
}
```
- Kaisheng's paper:
```
@article{
  author    = {Kaisheng Yao and Geoffrey Zweig},
  title     = {Sequence-to-Sequence Neural Net Models for Grapheme-to-Phoneme Conversion},
  journal   = {CoRR},
  volume    = {abs/1506.00196},
  year      = {2015},
}
```

Also, use the part of code:
- [fehiepsi's git repository](https://fehiepsi.github.io/blog/grapheme-to-phoneme/)
   - Baseline code
   - [Beamsearch](https://github.com/MaximumEntropy/Seq2Seq-PyTorch/)
- [cmusphinx git repository](https://github.com/cmusphinx/g2p-seq2seq)
   - Build vocab utils
   
## Requirements
- Python (3.6 maybe 3.5 이상이면 가능할 듯)
- NumPy
- Pytorch(0.2) and torchtext 
- [python-Levenshtein](https://github.com/ztane/python-Levenshtein/)
- [hangul-utils](https://github.com/kaniblu/hangul-utils/)

## Features
- [x] 표준 문장열 --> 발음 문장열
- [x] 발음 문장열 --> 표준 문장열(위의 모델과 인풋만 반대로 넣어주면 가능하며, build_vocab 에서 생성 가능합니다.)
- [ ] 한국어 G2P (IPA 형식)

## Usage
### Preperation:
1) 학습 데이터를 준비합니다 형식은 다음과 같습니다.
```
train_text.txt
무슨 일이 있어야 할까
...
train_trans.txt
무슨 이리 이써야 할까 
...
```
2) build_vocab을 실행 합니다.
```
python build_vocab.py --model_dir="../prepared_data/" --train_file="../naive_data/train"
```
3) main.py를 실행합니다.
```
예시 : 발음 문자열 to 표준 문자열 
> 바라보앋따 (입력)
= 바라보았다 (정답)
< 바라보았다 (출력)
반대의 케이스도 입력만 조정해주면 가능하며, 또한 Beamserch의 조정함으로써 후보군을 여러개 참조하도록 변환가능하합니다(문자열에 대한 여러개의 발음열 필요시).
```

## Authors
qqueing@gmail.com( or kindsinu@naver.com)


