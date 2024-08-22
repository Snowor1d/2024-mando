## 전체 시스템 개요도
![image](https://github.com/user-attachments/assets/928ba09f-5572-4f46-9d89-d2677999aa29)  
classification_cloud.py와, parameter_update.py는 테스트 항목 별 paragraph 분류 파라미터 튜닝에 사용됩니다.  
paragraph_classification_ver2.py와, LAMMA_query.py는 테스트 항목 별 문서 분류와 키워드 추출에 사용됩니다.  

## 환경설정  
python version : 3.12.4  
![image](https://github.com/user-attachments/assets/651fbbbe-bb55-4940-8b39-c86a2330ff01)  
pip install -r requirements.txt 을 통해 필요 라이브러리를 설치합니다.  

![image](https://github.com/user-attachments/assets/927300d0-9b19-47ec-b44f-375d13bf0434)  
python setup.py를 통해 nltk에 필요한 하위 파일을 다운받습니다.  

Meta의 오픈소스 LLM인 LAMMA를 다운 받습니다.  
**cpu 사용시**  
https://ollama.com/  
ollama를 다운받고, llama3 모델을 다운받습니다.  

**gpu 사용시**
https://huggingface.co/  
gpu 사용시에는 huggingface에서 사용하려는 모델을 다운받아야 합니다. 로그인 후 토큰을 발급받고, 로컬에 저장합니다.  
또한 사용하려는 모델에 대해 access request 승인을 받아야 합니다.  
위 과정이 끝나면 download_llama.py를 통해 해당 llama 모델을 다운받고 로컬에 저장합니다. 이후 LAMMA_query_gpu.py를 통해 저장한 llama 모델을 통해서 키워드를 추출할 수 있습니다.  

pargraph_classification_ver2.py는 paragraph_classifiaction.py와 다르게 문서에서 표와 이미지까지 접근하여 Test 항목 별로 분류해 저장합니다.  
이를 위해서는 java설치와 환경변수 설정이 필요합니다.  
https://www.oracle.com/java/technologies/downloads/  






