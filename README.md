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





진행 사항 
## llama3 사용 텍스트 매치 시도
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/e14e96a9-7cdc-4604-adc5-17a8ce18b656)  

llama 모델 사용, 텍스트 분류 테스트

![image](https://github.com/Snowor1d/2024-mando/assets/96639889/b8477bf8-65d9-417e-a1fa-e58720ad022c)    
Template


![image](https://github.com/Snowor1d/2024-mando/assets/96639889/fda87e97-6b07-4295-9539-6f6c893277fa)  
Query

-> 모든 texts들을 2번 (Conducted Emission-Voltage Method)로 매치함, **references가 있더라도 단락만 보고 키워드를 매치시키는 건 어렵다는 결론**


![image](https://github.com/Snowor1d/2024-mando/assets/96639889/9efb85b3-6604-45a8-93c4-37563ee3bd83)  
상세 질문에도 바로 적용해 보았으나, 답을 도출해내지 못함.  
전문용어가 많고 텍스트 분량이 난해해서 의미 도출을 힘들어하는 것 같음  

**텍스트 분류는, 따로 text split (similarity 분석)을 통해 해야 할듯**    

## text split

![image](https://github.com/Snowor1d/2024-mando/assets/96639889/b627f269-1b91-4377-8551-38bf7e153c5a)  
spec 문서 단락 구분에 사용된 정규 표현식 

text 단락 구분 방법  

1. 지정된 규칙대로 text split (1.1 과 같은 구분자가 있거나, 텍스트가 일정 길이 이상이거나 하면 자르기)
2. split된 항목들을 cosine similarity 비교하여 구분  

   ![image](https://github.com/Snowor1d/2024-mando/assets/96639889/4a9add59-a2ab-4ba8-bc95-990288659896)

                   

## keyword extraction

1. yake 알고리즘
-  ![image](https://github.com/Snowor1d/2024-mando/assets/96639889/793d1105-5af5-4dfa-8230-ed2598797afb)  
  (keyword_extraction.py)
- 결과
  
 ![image](https://github.com/Snowor1d/2024-mando/assets/96639889/2d70d115-33b8-4a6d-99b5-e22c1b7ea388)  
**keyword는 잘 뽑아내는 것 처럼 보이나, stopwords를 정의해도 잘 반영되지 않는다는 문제가 있는듯**  

2. rake 알고리즘 & cosine similarity 분석 통해 테스트명 매치
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/7310af56-3908-4973-a278-2d2132295091)  
(keyword_extraction_and_match.py)  

결과  
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/a06eb006-d4be-4c07-bb08-e886c842e94b)  
(ford specs 9, 10, 12 매치시 결과)  
**검증 필요, stop words또한 잘 정의해줘야 할 듯**  

EMC specs쪽에 특화된 임베딩 모델을 사전학습시켜야할 필요가 있지 않을까?  

rake 알고리즘 & cosine similarity 분석, 매치시 
(ford specs 8, 9, 10, 11, 13, 17, 25)  
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/bcf8fcf9-9abc-4807-b039-5a76b1578111)  
85퍼센트의 정답률 보임 

## document_keyword_match.py  
-> 단락 구분 기준 문장을 제시하면, pdf나 docx파일을 제시된 문장에 맞게 나누고 test 항목과 match  

![image](https://github.com/user-attachments/assets/5747592d-1e1f-4035-ae46-250f0e123cf9)  


각 FAW, NIO, CVTC 문서로 테스트  

**NIO 결과**  
![image](https://github.com/user-attachments/assets/ae42e549-e983-4c11-a14d-9ad5df1d4da3)  
정답률 20%  

**FAW 결과**  
![image](https://github.com/user-attachments/assets/3207d35e-5f46-4714-9e35-045f8419cb62)  
정답률 60%

**CVTC 결과**  
![image](https://github.com/user-attachments/assets/b9988b13-aa24-4bd5-9093-c2e7f6f3d08a)  
정답률 60%  

## keyword scoring   
- 각 test 별 단어 빈도 TOP 20 (stop words 제외)  
![image](https://github.com/user-attachments/assets/d4bac28d-729a-404b-b3ba-6c5589daacbb)

- test 별로 키워드 추가 (Excel표 참고)
![image](https://github.com/user-attachments/assets/8cf8d001-1579-4448-9298-7f30dfcb294f)

**CVTC 결과**
![image](https://github.com/user-attachments/assets/cb8330bd-8566-4ac5-9a4c-8e7064cc6456)  
정답률 90%, 오답 : immunity test  

**FAW 결과**
정답률 60% 동일

**NIO 결과**
![image](https://github.com/user-attachments/assets/605b733b-7a02-4e2f-a2c0-7845a98c859b)  
정답률 85프로, 오답 : Conducted Emission- Current Method, Electrostatic Discharge (ESD) - Powered-up Discharge 

![image](https://github.com/user-attachments/assets/19d64961-f69e-4190-a6cf-4a932c097168)  
추가 키워드

**CVTC 결과**  
![image](https://github.com/user-attachments/assets/e1bf8d55-8d32-4de4-accc-4258f25125cd)    
정답률 : 80%  

**FAW 결과** 
![image](https://github.com/user-attachments/assets/3e5a52c3-8bc7-47c6-b67a-1a2f6d360ce4)    
정답률 100프로  

**NIO 결과**   
![image](https://github.com/user-attachments/assets/b92f3182-38ca-4bff-a57a-442a3ba27a19)  
정답률 80프로  






