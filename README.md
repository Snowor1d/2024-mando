# 2024-mando
2024 만도 하계실습 레포입니다. 


# 영어 텍스트 마이닝 
-> 키워드 추출, 요약, 단락 구분 모두 가능함

##   (1) 키워드 추출
https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c

키워드 추출 라이브러리들
spaCy, YAKE, Rake-Nltk, Gensim

spaCy : NLP위한 파이썬 라이브러리 (summarization도 가능)

YAKE : 키워드 추출 개수등 여러 지원 기능

RAKE-Nltk : RAKE(Rapid Automatic Keyword Extraction) algorithm + NLTK toolkit

Gensim : keyword 추출 뿐만 아니라 summarization, text similarity 모두 가능 

(기술적 얘기)
정제 : 데이터 전처리, RE를 이용하여 특수 기호 제거 (토큰화 하기전)
정규화 (normalization) : 표현 방법이 다른 단어들 통합 (UK = United Kingdom, 10-10-22 = 10-10-2022)  
토큰화 (Tokenizatin) : 문장, 단어 분리 NLTK라는 유명한 토큰화 도구 존재  
Pos(Parts of Speech) 태깅 : 각 토큰에 해당하는 품사 붙여주기, averaged-perceptron-tagger   
불용어 : 자주 등장하지만 큰 의미가 없는 단어 처리 (I, you, she..)  
단수화, 복수화 : 단수와 복수를 같은 단어로 취급, TextBlob 토큰 와이저 함수 이용  
표제어 추출 : 서로 다른 단어 형태 하나로 취급 (am, are, is -> be)  
개체명 인식 : 개체(entity)의 유형 인식    
DTM (문서 단어 행렬) : 정형 데이터로 만듦  (텍스트를 테이블 형 데이터로) (row -> (문서 or 문장), column -> 단어)  
Bag of Words (단어들이 들어있는 가방) : 단어들의 출현빈도(frequency) 수치화에 표현)    

정제(cleaning) -> 정규화(normalization) -> 토큰화(Tokenization) -> PoS(Parts of Speech)태깅 -> 불용어(stopwords)처리, 교정, 추출 -> 철자 교정 -> 단수화/복수화
-> 어간 추출(stemming) -> 표제어 추출(Lemmatization) -> 개체명 인식(Named Entity Recognition) -> Bag of Words -> 키워드 추출*


##   (2) 텍스트 요약
요약에는 두가지 방법이 있는데  
Extractive methods -> 원본에서 문장 그대로 추출, 텍스트 재구성  
Abstractive methods -> 완전히 새로운 문장으로 텍스트 재구성  

요약또한 여러 모듈로 가능 (newspaper3k, spacy..)  
성능은? 10%까지 요약했을때 번역까지 자연스럽게 된다고 함 

요약 APIs -> Chat GPT, NaverCloud, Ms Azure . . . 그러나 아직 에러 有  

Text summarization 연구 주제
 - multi/long summarization (여러 소스를 요약할 수록 요약 효율 효용 증가, 그러나 난이도도 증가, Multi documents summarization, Long documents summarization)

**SOAT : BART(오픈소스)**  
BERT(bidirectional encoder) + GPT(generation task) -> BART (seq2seq 형태) 

(기술적 얘기)
요약문 생성에 사용되는 기술들
- Transfer Learning
- Knowledge-enhanced texdt generation
- Post-editing Correction
- GNN (Graph Neural Network)
  
https://github.com/uoneway/Text-Summarization-Repo

##   (3) 단락구분
- Free Chat GPT  
- Hyperwrite

Use pre-trained embedding ( ex) all-mpnet-base-v2)  
                       |
Dot product - cosine similarity(유사도) -> 파이썬 라이브러리 sklearn 사용
                       |
                       
 <img width="287" alt="image" src="https://github.com/Snowor1d/2024-mando/assets/96639889/c8a68a3c-a018-459c-8430-ce222c301948">  
 (극소값 기준으로 text split, identifying split spots)  
 https://eehoeskrap.tistory.com/186

##   (4) Multipe documents matching
 <img width="340" alt="image" src="https://github.com/Snowor1d/2024-mando/assets/96639889/68754d08-53b3-4b73-83cd-9d8ddfaf0578">  


# Vision scaling
이미지 vision processing시 사이즈나 scaling의 문제가 있을 수 있음
이미지의 사이즈를 맞추려고 하면? -> scale의 문제 발생

![image](https://github.com/Snowor1d/2024-mando/assets/96639889/720a909f-64a3-454a-8562-0abf461a1187)

스케일과 상관없이 프로세싱을 하려면 이미지의 크기를 변화시켜가며 분석 작업을 하는 것이 중요 
Gaussian Pyramid -> 피라미드를 생성할때 블러링과 다운 샘플링을 통해, 이미지를 축소해가며 샘플링  

scale space : 대상이 가질 수 있는 다양한 스케일의 범위를 한꺼번에 표현
(대상의 구조를 여러 스케일에 걸쳐 다루기 위함)
https://darkpgmr.tistory.com/137

Machine Learning and Computer Vision for PCB Verification - CHEN YANG

faster R­CNN is a improved algorithm from R­CNN, while there is
another intermediate version called fast R­CNN   

SIFT (Scale-invarinat Feature Transform)
1. 다양한 크기의 Laplace feature map을 통해 blob 위치 검출
2. feature map의 크기를 변경해 가며 blob의 크기 검출
   -> 다양한 크기의 특징점을 얻을 수 있게됨

image resize
- Pillow (python imaging library)
- openCV

Resizing 기법
-> Nearest Neighbor Interpolation, Bilinear Interpolation, Bicubic Interpolation, Resizing with Anti-Aliasing, Content-Aware Resizing

image normalization -> 이미지 픽셀값의 범위를 조정하는 거지, size나 scale과 관련된 건 아닌 듯 . .  
주로 vision 쪽에서 행해지는 segmentation, recognition, object detection 쪽에서는 이미지 입력 크기를 resizing(input 이미지 사이즈를 통일 -> 계산 효율적)하는 것이 전부  

# Other?

https://csdl.postech.ac.kr/bbs/board.php?bo_table=sub6_1&wr_id=265
https://ieeexplore.ieee.org/document/9181149
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/55cdf943-9955-43db-9601-69096782ba05)

AutoCkt (Reinforcement 기반 회로설계 플랫폼) 
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/9306701f-ef98-4ae9-ace7-ac19afa8e6b2)
https://arxiv.org/pdf/2001.01808









                   


  


