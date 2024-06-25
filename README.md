# 2024-mando
2024 만도 하계실습 레포입니다. 


1 - 텍스트마이닝
-> 추출된 문장에서 인사이트를 얻는 것

영어 텍스트 마이닝 
-> 키워드 추출, 요약, 단락 구분 모두 가능함

(1) 키워드 추출
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
-> 어간 추출(stemming) -> 표제어 추출(Lemmatization) -> 개체명 인식(Named Entity Recognition) -> Bag of Words -> 키워드 추출

(2) 텍스트 요약
요약에는 두가지 방법이 있는데
Extractive methods -> 원본에서 문장 그대로 추출, 텍스트 재구성
Abstractive methods -> 완전히 새로운 문장으로 텍스트 재구성

요약또한 여러 모듈로 가능 (newspaper3k, spacy..)
성능
