from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
import os

def get_all_files_and_titles(directory):
    
    file_list = []
    file_titles = []
    
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            file_list.append(full_path)
            file_titles.append(file_name)
    
    return file_list, file_titles

def get_all_subdirectories(directory):
    subdir_list = []
    subdir_names = []

    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            subdir_list.append(full_path)
            subdir_names.append(entry)
    
    return subdir_list, subdir_names

def read_txt_files_in_folders(folder_list):
    folder_contents = {}

    for folder in folder_list:
        txt_files_contents = []
        
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder, filename)

                with open(file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                    content_from_third_line = lines[2:]
                    content = ""
                    for line in content_from_third_line:
                        content += line + " "
                    txt_files_contents.append(content)
        folder_contents[folder] = txt_files_contents 
    return folder_contents
directory_path = "classification_contents"
folder_list, folder_titles = get_all_subdirectories(directory_path)
print(folder_titles)
contents = read_txt_files_in_folders(folder_list)

for key in contents.keys():
    print(key)

llm = ChatOllama(model="llama3:latest")
prompt_text = ''''''
with open("llama_prompt.txt", 'r', encoding='utf-8') as file:
    prompt_text = file.read()

prompt = ChatPromptTemplate.from_messages([
    ("system", prompt_text),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()
result = ''''''
count = 0
for key in contents.keys():
    result += "---------------------------------\n"
    result += f"<{key.split("\\")[1]}>\n"
    for candidates in contents[key]:
        q = ''''''
        q += candidates + "\n"
        section_result = ""
        for token in chain.stream(
            {"input" : q + ("in this text, what is the test name? and what is the base specification? just answer with two words: test name, base specification. Don't show me any ohter words. please answer with form Test name : , Base specification : ")}
        ):
            section_result += token
        print(section_result)
        section_result += "\n\n"
        result += section_result
    count += 1
    print(f"{count}/{len(contents.keys())}")

with open("llama_result.txt", 'w', encoding='utf-8') as file:
    file.write(result)

print("프로그램을 종료하려면 아무키나 누르세요.")
input()