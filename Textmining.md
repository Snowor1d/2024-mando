# LLaMA
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/9e5f261e-1741-4ce5-b9c2-ffd4c92d1216)

연구 및 상업적 용도로 사용할 수 있는 Meta의 **오픈 소스** 대규모 언어 Ai 모델
LLaMA2 - 2023.07 release  
LLaMA3 - 2024.04 relesae  

경량화 모델 有, best performance
컨텍스트 길이 8K (영어 9000자)

paragraph-keyword match? or keyword question - answer?

#1. using fine tunning

![image](https://github.com/Snowor1d/2024-mando/assets/96639889/e8ee4d37-76f6-4bcf-a7e9-b5b52368b07f)

  1) 문서들을 단락별로 분리 (8.2.2.1, 8.2.2.2 ... )  
  2) 학습 데이터로 Fine tunning ( Q. 단락들을 keyword와 매치시킨 학습 데이터가 존재하는가?)  
  3) Fine tunning된 LLaMA3에 query형태로 keyword match  

#2. using RAG
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/42f0bae5-c58c-468c-aa26-b1f9dfae7068)

target 문서, references를 외부 DB로 활용, 검색 강화 기반 응답

https://fornewchallenge.tistory.com/entry/%F0%9F%A6%99Ollama%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-LLAMA3-RAG-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0  

#2. using RLHF (사람 피드백 기반 강화학습)  
LLAMA는 RLHF 또한 가능
사람이 직접 결과에 피드백을 주며 LLAMA 재학습

