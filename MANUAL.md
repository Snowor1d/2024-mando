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

## input data  
**config.txt**  
![image](https://github.com/user-attachments/assets/42379c34-52c2-4ba1-9171-f7bd7166444e)  
분석하려는 파일에 대한 정보가 들어있는 텍스트 파일입니다. 위 형식에 맞게 파일명, 나누고자 하는 sections 등을 입력하면 이를 바탕으로 Test항목과 매치합니다.   
나누고자 하는 sections을 입력할때는 소제목등을 파일에서 그대로 복사 붙여넣기 하는 것을 추천합니다.

**optimization_data**  
![image](https://github.com/user-attachments/assets/210bb3dd-c7d4-4363-a907-da8b1a86ca28)  
optimization_data는 문서 분류의 정확도를 높이고자 할때 수정할 수 있습니다. 

![image](https://github.com/user-attachments/assets/590d6d78-fe99-427a-8e0e-888cb21a9101)  
optimization_data 안에는 다음과 같은 txt파일이 있습니다. 내부의 첫번째 라인은 학습하고자 하는 파일의 경로, 두번째는 파일 문서의 종류, 세번째는 분석하고자 하는 start page, 네번째는 end page이고 그 다음 줄 부터 각 섹션의 제목과
섹션에 해당하는 분류 정답이 기록되어 있습니다. 0~12의 수가 무엇을 의미하는지는 parameter_update.py의 'tests' 변수에서 확인하실 수 있습니다. 

## output  
![image](https://github.com/user-attachments/assets/3000e8f0-6bc7-4844-87f1-d4639721b2b6) 
classification_contents 폴더에서 다음과 같이 각 테스트 항목 별로 텍스트, 표, 그림이 저장됩니다.  
![image](https://github.com/user-attachments/assets/728fd87c-395b-4b67-877f-4015d190b8a3)  
가장 앞에 붙은 숫자는 매치된 섹션을 구분하기 위함입니다. 0, 1, 2까지 있다면 해당 Test에 매치된 sections가 3개가 있다는 의미입니다.  

llama_result.txt에서는 LLM 모델이 각 테스트에서 뽑아낸 키워드가 기록됩니다.    
![image](https://github.com/user-attachments/assets/393e2164-8975-41b0-b6a6-d96718963279)













