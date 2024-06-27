# LLaMA
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/9e5f261e-1741-4ce5-b9c2-ffd4c92d1216)

연구 및 상업적 용도로 사용할 수 있는 Meta의 **오픈 소스** 대규모 언어 Ai 모델
LLaMA2 - 2023.07 release  
LLaMA3 - 2024.04 relesae  

경량화 모델 有, best performance
컨텍스트 길이 8K (영어 9000자)

paragraph-keyword match? or keyword question - answer?

## 1. using fine tunning

![image](https://github.com/Snowor1d/2024-mando/assets/96639889/e8ee4d37-76f6-4bcf-a7e9-b5b52368b07f)
https://www.youtube.com/watch?v=YJNbgusTSF0

  1) 문서들을 단락별로 분리 (8.2.2.1, 8.2.2.2 ... )  
  2) 학습 데이터로 Fine tunning ( Q. 단락들을 keyword와 매치시킨 학습 데이터가 존재하는가?)  
  3) Fine tunning된 LLaMA3에 query형태로 keyword match  

## 2. using RAG
![image](https://github.com/Snowor1d/2024-mando/assets/96639889/42f0bae5-c58c-468c-aa26-b1f9dfae7068)
https://velog.io/@judy_choi/LLaMA3-%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-RAG-%EA%B5%AC%EC%B6%95-Ollama-%EC%82%AC%EC%9A%A9%EB%B2%95-%EC%A0%95%EB%A6%AC

target 문서, references를 외부 DB로 활용, 검색 강화 기반 응답

https://fornewchallenge.tistory.com/entry/%F0%9F%A6%99Ollama%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-LLAMA3-RAG-%EC%8B%9C%EC%8A%A4%ED%85%9C-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0  

## 3. using RLHF (사람 피드백 기반 강화학습)  
LLAMA는 RLHF 또한 가능  
사람이 직접 결과에 피드백을 주며 LLAMA 재학습

## 개발 순서
1. RAG
2. RAG + RLHF
3. RAG + RLHF + fine tunning

## 필요 resources (local에서 돌리려면)
CPU : at least 8 cores (cpu i5)
GPU : Nvidia with CUDA architecture, RTX 3000 series recommend 
RAM : (8B) 모델의 경우 at least 16GB RAM
 
MacBook Pro (6-Core Intel Core i7 @ 2.60GHz, 16 GB RAM)  
- llama-2-13b-chat.ggmlv3.q4_0.bin (CPU only): 3.32 tokens per second  
- llama-2-13b-chat.ggmlv3.q8_0.bin (CPU only): can run, but extremely slow, unusable  

Gaming Laptop (12-Core Intel Core i5 @ 2.70GHz, 16GB RAM, GeForce RTX 3050 mobile 4GB)  
- llama-2-13b-chat.ggmlv3.q4_0.bin (CPU only): 2.12 tokens per second  
- llama-2-13b-chat.ggmlv3.q8_0.bin (CPU only): 2.10 tokens per second  
- llama-2-13b-chat.ggmlv3.q4_0.bin (offloaded  8/43 layers to GPU): 5.51 tokens per second  
- llama-2-13b-chat.ggmlv3.q4_0.bin (offloaded 16/43 layers to GPU): 6.68 tokens per second  
- llama-2-13b-chat.ggmlv3.q8_0.bin (offloaded  8/43 layers to GPU): 3.10 tokens per second  

Cloud Server (4-Core Intel Xeon Skylake @ 2.40GHz, 12GB RAM, NVIDIA GeForce RTX 3060 Ti 8GB)  
- llama-2-13b-chat.ggmlv3.q4_0.bin (offloaded 38/43 layers to GPU): 11.06 tokens per second  
- llama-2-13b-chat.ggmlv3.q8_0.bin (offloaded 21/43 layers to GPU):  2.56 tokens per second    

Cloud Server with 2x GPUs (8-Core Intel Xeon Skylake @ 2.40GHz, 24GB RAM, 2x NVIDIA GeForce RTX 3080 10GB)  
- llama-2-13b-chat.ggmlv3.q4_0.bin (offloaded 43/43 layers to GPU): 33.27 tokens per second    
- llama-2-13b-chat.ggmlv3.q8_0.bin (offloaded 43/43 layers to GPU): 28.27 tokens per second    

Cloud Server (4-Core Intel Xeon Skylake @ 2.40GHz, 24GB RAM, NVIDIA GeForce RTX 3090 24GB)    
- llama-2-13b-chat.ggmlv3.q4_0.bin (CPU only): 1.52 tokens per second    
- llama-2-13b-chat.ggmlv3.q8_0.bin (CPU only): 1.18 tokens per second    
- llama-2-13b-chat.ggmlv3.q4_0.bin (offloaded 43/43 layers to GPU): 62.81 tokens per second    
- llama-2-13b-chat.ggmlv3.q8_0.bin (offloaded 43/43 layers to GPU): 36.39 tokens per second  

RLHF까지는 Nvidia 3000대의 GPU로도 충분할 것. (CPU만으로도 돌아갈 것 같긴 함)  
그러나 fine-tunning에는 최소 24GB의 VRAM을 가진 GPU필요.  

표, 사진 데이터는 ?  
word에 있는 표는 col-row 데이터로 변환 가능  
scan파일에 있는 표는 고민해봐야 될듯  

## 계획  
7월 1주차 - Lamma 개발 환경 세팅, docx파일 paragraph로 분리  
    2주차 - RAG에 쓰일 References, documents 전처리   
    3주차 - RAG 모델 외부 db와 연결, 테스트   
    4주차 - RAG 모델 디버깅 및 RLHF 적용  

8월 1주차 - RLHF 방식 적용 및 테스트  
    2주차 - Fine tunning 기법 적용 방안 모색, 배포 및 문서화  
    3주차 - 배포 (python exe or docker . . ) 및 문서화
  
