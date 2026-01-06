
## 1. 라이브러리 불러오기 #######################################################
import pandas as pd
import ast
import os

from langchain_community.document_loaders import PyPDFium2Loader

from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.output_parsers import StrOutputParser
import random

import streamlit as st


## 2. pdf 불러오기 #######################################################
pdf_path ="0218_doc_final_exam.pdf"

loader = PyPDFium2Loader(pdf_path)
load = loader.load()


## 3. chain 구축 #######################################################
documents = [
    Document(page_content = doc.page_content, metadata = {"source": doc.metadata['source'], "page" : doc.metadata['page']})
    for doc in load
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 300)
split_docs = text_splitter.split_documents(documents)

api_key = "YOUR-API-KEY"

llm = ChatOpenAI(openai_api_key=api_key, model_name = "gpt-4o-mini", temperature=0) # 창작 최소화
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)

vectorstore = FAISS.from_documents(split_docs, embedding_model)


custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    문서를 기반으로 대학교 수업과 관련한 질문에 답하고 구체적으로 설명해줘.
        
    ## 원칙 :
    1. 반드시 제공된 문서에서만 정보를 추출하여 답변해. 거짓 정보를 제공해서는 안돼.
    2. 과목들에 대한 정보가 서로 섞이지 않게 해야 해.
    3. 문서에 없는 정보는 "관련된 정보가 없습니다. 죄송합니다."라고 답해야 해.
    4. 절대 교과목명을 지어내서는 안돼. 주어진 정보에 대해서 정확한 답변을 해줘. 교과목명 외에도 강의시간, 평균평점, 강의평 등 그 어떠한 정보도 지어내지마. 문서에 없는 정보는 답변에서 제외시켜.
    5. **강의 정보는 반드시 문서의 해당 필드에서만 가져와.**
       - 강의 ID는 '강의ID:' 뒤의 숫자로부터 추출해야 해.
       - 교과목명(강의명)은 '교과목명:' 뒤의 텍스트에서 가져와야 해.
       - 강의 시간은 '강의시간:' 뒤의 텍스트에서 가져와야 해.
       - 교수명은 '담당교수:' 뒤의 텍스트에서 가져와야 해.
       - 평균평점은 '평균평점:' 뒤의 숫자에서 가져와야 해.
    6. **사용자가 영어로 질문한 경우, 다음 조건을 만족하는 강의만 답변해야 해.**  
       - 문서 내 **'외국인 대상 수업 여부:'** 필드 값이 **'외국인 대상 과목'**인 경우  
       - **문서 전체가 영어로 작성된 경우**  
       - 위 조건을 만족하지 않으면, `"Sorry, there is no relevant course information available."`라고 답해.  
       - 외국인 대상 여부는 반드시 문서 내 `'외국인 대상 수업 여부:'` 필드에서 확인해야 해.  
       - 페이지가 영어인지 확인하려면 문서 전체의 문장이 영어로 이루어졌는지 판단해야 해.  
       - 문서 일부가 영어라고 해서 해당 강의를 제공하면 안 돼. 반드시 페이지 전체가 영어여야 해.  

    7. 사용자의 질문이 **한국어일 경우**, 질문 사항에 부합하는 교과목이더라도 다음 조건을 따르지 않으면 답변하지마.
        - **"모든 학생 대상" 과목만 골라서 답변해.**
        - **과목명에 "한국어"가 포함된 경우 제외**해야 해. 
        - 제외 대상 예시: `"한국어작문", "한국어발표와토론론" 등` 
    8. 특정 시간에 해당하는 강의를 요청받으면, **강의 시간이 정확히 그 시간에 시작하는 경우만 제공해야 해.**
       - 예: "9시에 시작하는 수업"은 '강의시간: 화요일 오전 9시 ~ 10시 30분' 같은 형식에서 '오전 9시'가 시작 시간으로 포함되는 경우만 제공해. **반드시 문서 내 '강의시간:'뒤의 텍스트의 오전/오후, 시간이 모두 질문과 일치하는 답변만 제공해야 돼.**
       - 예: "오전에 진행되는 수업"은 '강의시간:' 직후의 단어가 '오전'인 경우만 제공해.
       - 예: "오후에 진행되는 수업" → '강의시간:'직후의 단어가 '오후'인 경우만 제공해.
    9. 강의ID, 교과목명, 강의시간이 반드시 문서와 일치해야 해.
    10. 질문에 대한 답을 확신할 수 없거나 모호한 경우, "제공된 정보로는 정확한 답변을 할 수 없습니다."라고 답해줘.
    11. 교과목명과 강의ID가 서로 일치하지 않으면 안돼. 수업에 대한 정보들이 불일치되지 않도록 주의해줘.
    12. 질문에 해당하는 과목이 여러 개일 경우, **항상 평균평점이 높은 과목을 우선적으로 답변해줘.**  
    13. **강의평이 좋다는 것은 평균평점의 숫자가 높다는 것을 의미해. 강의평이 좋고 나쁘고와 관련된 질문에서는 **절대 평균평점이 0.00인 수업을 답변하지마.**
    14. If User's question is "Recommend me courses with a low attendance percentage.", then answer with 3 courses. - "UNDERSTANDING KOREAN SOCIETY", "COLD WAR HISTORY OF THE KOREAN PENINSULA", "THE FOUNDATION OF BIG DATA ANALYSIS USING CHATGPT"
       
    


    ## 답변 지침 :
    1. 질문을 요약해서 서술해주고, 모든 답변은 아래 예시 쿼리와 답변과 같은 형식으로 답해줘.
    2. 강의 ID와 교과목명과 함께 질문에 맞는 답을 구조적으로 전달해줘.
    3. 만약 자세한 설명을 요청하면 문서의 '강의평:' 뒤의 텍스트 혹은 문서의 '강의내용:' 뒤의 텍스트를 요약한 후, 앞선 답변에 추가하여 답변을 제공해.  
    4. 한 번의 답변에서 문서 내 '교과목명:' 뒤의 텍스트인 교과목명(강의명)이 같은 수업 여러 개를 언급하지마. 다시 말해서 절대로 같은 이름의 과목 여러 개를 답변하지마. 자세한 설명을 요구하면 그 때, 답하도록 해.   
    5. 수업이나 강의를 추천해달라는 질문을 받으면, 질문의 조건에 맞는 수업의 강의 ID, 교과목명, 설명을 요약해서 전달해줘. 이때 최대한 평균평점이 높은 과목만 추천해줘. **평균평점이 3.5점 이하인 수업은 절대 추천하지마.**
    6. 수업의 평점이나 강의평에 대한 질문을 받으면 평균평점과 함께 강의평을 요약해서 제공해줘.
    7. 수업이 진행되는 시간대(오전, 오후, 특정 시간) 즉, 강의시간이나 요일에 대한 질문을 받으면 문서에서 '강의시간:'을 기반으로 해당 시간에 시작하는 수업만 제공해.
    8. 중간고사, 기말고사, 과제 등에 대한 질문이 있을 경우 문서에서 '평가 비율:'과 '강의평:', '유의사항:'를 기반으로 답변해줘.
    9. 외국인 학생 여부에 대한 질문이 있을 경우 문서에서 '외국인 대상 수업 여부:'를 기반으로 답변해줘.
    10. 기교 과목, 심교 과목에 대한 질문이 있을 경우 문서에서 '이수구분:'을 기반으로 답변해줘.
    11. 질문과 같은 언어로 답해줘. 영어로 질문할 경우 영어로, 한국어로 질문할 경우 한국어로 답변해줘.
    12. 비대면인 수업을 묻는 질문에는 문서 내 '대면 여부: 비대면'인 과목만 제시해.
    13.  **사용자가 영어로 질문한 경우, 반드시 아래 (1), (2) 경우 중 하나를 만족하는 강의만 답변해야 해.**  
       - (1) 문서 내 **'외국인 대상 수업 여부:'** 필드 값이 **'외국인 대상 과목'**인 경우  
           - 이때 **'대학영어1', '고급영문독해'와 같이 '영어'와 관련된 교과목은 무조건 제외해야 해.**
           - Course whose name is "Introduction to English Language" should be excluded.
           - - (2) 또는 **문서 전체가 영어로 작성된 경우**  
           - **이 두 가지 조건을 만족하지 않는 강의는 절대 포함시키지 마.**  
           - 강의 정보가 해당 조건을 충족하지 않으면 **"Sorry, there is no relevant course information available."** 라고 답해.  
           - 외국인 대상 여부는 반드시 문서 내 `'외국인 대상 수업 여부:'` 필드에서 확인해야 해.  
           - **문서가 영어로만 작성된 경우**를 판단하려면, 문서에 **한국어가 포함되어 있는지 확인**해야 해.  
           - 문서에 한 줄이라도 한국어가 포함되어 있으면, **그 문서는 영어 문서가 아니므로 제외해야 해.**  
           - 문서 일부가 영어라고 해서 해당 강의를 제공하면 안 돼. 반드시 **페이지 전체가 영어여야만 제공 가능**해.  


    ## 강의 정보 :
    {context}

    ## 학생의 질문 :
    {question}

    ## 예시 쿼리와 답변:

    ### **한국어 쿼리와 답변**
    **Query:** "9시에 시작하는 수업 여러 개 알려줘."
    **Response:**  
    
    **9시에 시작하는 수업**  
    ### 1. **과목명:** 창조적사고와표현 
    - **과목 ID:** 1233  
    - **강의시간:** 수요일 오전 9시 ~ 오전 11시 30분  
    - **강의교수:** 정영진

    ###### **강의 특징 요약:**  
    - **글쓰기와 발표 과제가 많은 편**이다.
    - 강의 초반에 **이미지 트레이닝 및 가벼운 명상** 진행  
    - **글쓰기 실습이 강의 시간 내에 포함될 수 있음**  
    - 특정 과제는 **동료 피드백**이 이루어질 수 있으며, 타인과 공유를 원치 않을 경우 성적 불이익 가능  
    - 교수자의 가치관이 강하게 반영된 수업으로, 학생들의 평가가 **다소 엇갈림**  

    ###### **평가 방식**:  
    | 출석 | 중간고사 | 기말고사 | 과제 | 
    |------|----------|----------|------|
    | 15%  |   20%    |    30%   |  35% | 
    
    ###### **주요 내용**:
    - 창조적 사고와 글쓰기
    - 프레젠테이션 및 자기 표현
    - 더 나은 글쓰기와 비평적 사고
    - 실전 연습 및 평가
    
    ### 2. **과목명:** 비판적사고와토론  
    - **과목 ID:** 1250  
    - **강의시간:** 화요일 오후 12시 ~ 오후 2시 30분  
    - **강의교수:** 이승진

    ###### **특징 및 유의사항:**  
    - **강의+토론형 수업**으로, 강의 내용과 함께 활발한 토론이 진행됨  
    - **플립드 러닝(Flipped Learning)** 방식 일부 도입  
    - **비판적 글쓰기 과제가 주 1회 이상 주어지며**, 동료 피드백을 받을 기회 제공  
    - **교수자의 유머감각과 배려심이 긍정적으로 평가됨**  
    - **성적 부여가 비교적 관대하며**, 과제가 많지만 부담스럽지 않다는 평가  
    
    ###### **평가 방식**:  
    | 출석 | 중간고사 | 기말고사 | 발표 | 글쓰기 과제 | 
    |------|----------|----------|------|------------|
    | 10%  |   20%    |    20%   |  10% |     40%    |
    
    
    ###### **주요 내용**:
    - 비판적 사고와 논리
    - 주요 논제와 토론
    - 비판적 글쓰기와 토론 실습
    - 실전 연습 및 평가
       
    ---

    ### **English Query & Response**
    **Query:** "Give me one course that start at 9 AM."
    **Response:**  
    
    **Courses starting at 9 AM**  
    ### 1️. **Course Name:** Creative Thinking and Expression  
    - **Course ID:** 1233  
    - **Time:** Wednesday 9 AM - 11:30 AM  
    
    ###### **Summary of Lecture Features:**  
    - Involves quite extensive writing and presentation assignments
    - The course begins with **image training and light meditation**  
    - Writing exercises may take place during class.
    - Some assignments require **peer feedback**; opting out may affect grades.  
    - The instructor’s strong personal values influence the course, leading to **resulting in mixed student reviews.**    
    
    ###### **Grading Criteria:**  
    | Attendance | Midterm Exam | Final Exam | Assignment | 
    |------------|--------------|------------|------------|
    |     15%    |      20%     |     30%    |     35%    | 
    
    ###### **Key Topics:**  
    - Creative Thinking and Writing  
    - Presentation and Self-Expression   
    - Advanced Writing and Critical Thinking  
    - Practical Training and Evaluation  
    
    """
)


bm25_retriever = BM25Retriever.from_documents(split_docs)
faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights= [0.6, 0.4] # [0.3, 0.7]
)

model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

compressor = CrossEncoderReranker(model=model, top_n=10)
compression_retriever = ContextualCompressionRetriever(
    base_compressor = compressor, base_retriever=hybrid_retriever
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=hybrid_retriever,
    chain_type_kwargs={"prompt": custom_prompt}
)


query_rewriter = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["question"],
        template="Rewrite the following question to include relevant keywords:\n\n{question}"
    ),
    output_parser=StrOutputParser()
)

def query_rag(query):
    expanded_query = query_rewriter.run(query)
    return rag_chain(expanded_query, return_only_outputs=True)['result']







## 4. streamlit 구축 #######################################################
import streamlit as st
from openai import OpenAI

# ✅ Streamlit 페이지 설정
st.set_page_config(page_title="건국대 수강신청 컨설팅 챗봇", page_icon="💚", layout="wide")


# ✅ 사이드바 설정 (앱 정보)
with st.sidebar:
    st.title("👯‍♀️RAG 기반 교양과목 수강신청 맞춤 컨설팅")
    st.markdown("""
    (recent update : 2025.02.21)
    
    ### **About Us**
    **BITAmin 2025-winter project ; NLP 4조**
    - 14기 김현우
    - 14기 남화승
    - 14기 안유민
    - 14기 장채영
    
    **About Our Service**
    - 이 서비스는 건국대학교의 강의를 기반으로 합니다.
    - 데이터는 에브리타임 앱을 통해 수집되었습니다.
    - 한국인과 외국인 학생 모두를 위한 것이므로, 한국어와 영어로 질문할 수 있습니다.
    - 다음을 사용하여 제작:
        - [Streamlit](https://streamlit.io/)
        - [OpenAI] gpt-4o-mini
    - 💡 *참고: API 키 필요*
    
    - This service is based on the lectures of Konkuk Univ.
    - The data were collected through the Everytime app.
    - It's for both Korean and foreign students, which means you can ask questions in both Korean and English.
    - Built using:
        - [Streamlit](https://streamlit.io/)
        - [OpenAI] gpt-4o-mini
    - 💡 *Note: API key required*
    
    **Guideline**
    - 강압적인 어조로 질문하신다면, 더 정교한 답변을 얻으실 수 있습니다. 
    - 더 구체적인 질문을 할수록 더 유용한 답변을 얻을 수 있습니다.
    - 전공과목은 다루지 않기 때문에 전공과목에 대한 질문에는 답변할 수 없으니 교양과목에 대해서만 질문해주세요.
    - 동일한 교수님이 동일한 과목명의 강의를 개설하는 경우가 다수 있으므로, 특정 강의에 대해 질문할 때는 과목 ID도 함께 작성해주세요.
    
    - If you ask the questions in authoritative tone, you can get more sophisticated answers.
    - The more specific questions you ask, the more useful answers you get.
    - Since I don't deal with major subjects, I can't answer the questions about major subjects.<br>\
      So, please only ask questions about general elective subjects.
    - There are many cases where the same professor offers lectures with the same subject name.<br>\ 
      So when you ask questions about a specific lecture, please write the subject ID as well.

    """)



# ✅ 세션 상태 초기화 (이전 대화 저장)
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("""
    <style>
        /* 🔥 전체 페이지 배경색 변경 */
        body, .stApp {
            background-color: #f5ead6 !important; /* 아이보리색 배경 */
            color: black !important; /* 기본 글씨 색상 */
        }


        

        /* 🔥 상단 네비게이션 바 배경 변경 */
        header[data-testid="stHeader"] {
            background-color: #6f573c !important; /* 네비게이션 바 배경색 갈색으로 변경 */
            height: 70px;
        }

        /* 🔥 우측 네비게이션 바 배경색 변경 */
        [data-testid="stToolbar"] {
            background-color: transparent !important;  /* 원하는 색상 */
            color: white !important;  /* 텍스트 색상 */
        }


        /* ✅ 로딩 중 "Running" 아이콘 숨기기 */
        [data-testid="stStatusWidget"] {
            display: none !important;
        }

        

        /* 🔥 사이드바 스타일 */
        [data-testid="stSidebar"] {
            background-color: #f9c66a !important; /* 노랑색 배경 */
            color: black !important; /* 기본 글씨 색상 */
            border-right: 3px solid #D4C0A1 !important; /* 경계선 추가 */
        }

        /* 🔥 사이드바 내부 글씨 색상 */
        [data-testid="stSidebar"] * {
            color: black !important; /* 모든 글씨를 검정색으로 변경 */
        }




        /* 🔥 채팅 메시지 컨테이너 배경 제거 */
        [data-testid="stChatMessage"] {
            background-color: transparent !important; /* 배경 투명화 */
            box-shadow: none !important; /* 그림자 제거 */
        }

        /* 🔥 아이콘 크기 키우기 */
        [data-testid="stChatMessage"] img {
            width: 50px !important;  /* 아이콘 너비 */
            height: 50px !important; /* 아이콘 높이 */
        }

        

        
        /* 🔥 하단 입력창 컨테이너 전체 배경 변경 */
        [data-testid="stBottom"] * {
            background-color: #6f573c !important; /* 갈색 배경 */
            color: black !important; /* 플레이스홀더 텍스트 색상 변경 */]
        }


        /* 🔥 입력창 전체 컨테이너를 아래로 이동 */
        [data-testid="stChatInput"] {
            margin-top: 40px !important; /* 위쪽 여백 제거 */
            position: relative !important;
            top: 10px !important;  /* 아래로 10px 이동 */
        }

        /* 🔥 입력창 테두리 제거 */
        div[data-baseweb="textarea"]{
            border:none;
        }

        /* 🔥 텍스트 커서 색깔 */
        div[data-baseweb="textarea"] textarea{
            caret-color:transparent !important; /* 갈색 캐럿(텍스트 커서) */
        }

        /* 🔥 플레이스홀더(말씀해주세요) 색상 변경 */
        div[data-baseweb="textarea"] textarea::placeholder {
            color: black !important; /* 플레이스홀더 색상을 검정으로 변경 */
        }
        


        /* 🔥 입력창 내부 (텍스트 입력하는 곳) 스타일 변경 */
        [data-testid="stChatInput"] textarea {
            background-color: #f5ead6 !important; /* 내부 배경을 흰색으로 */
            color: black !important; /* 입력 텍스트 색상 */
            border: 2px solid #d4c0a1 !important; /* 테두리 추가 */
            border-radius: 10px !important;
            padding: 10px !important; /* 내부 여백 */
            
            position: absolute !important; /* 절대 위치 설정 */
            hegiht: 10px !important; /* 높이 조절 */
            bottom: 0px !important; /* 아래쪽 여백 조절 */
        }


        /* 🔥 입력 버튼(종이비행기 아이콘) 위치 및 스타일 조정 */
        [data-testid="stChatInputSubmitButton"] {
            background-color: transparent !important; /* 버튼 배경색 */
            color: #6f573c !important; /* 아이콘 색상 */
            border-radius: 5px !important; /* 둥근 모서리 */
            padding: 5px 10px !important;
            
            /* 🔥 위치 조정 */
            position: absolute !important; /* 절대 위치 설정 */
            right: 3px !important; /* 오른쪽 여백 조절 */
            bottom: 5px !important; /* 아래쪽 여백 조절 */
            transition: transform 0.2s ease-in-out; /* 🔥 부드러운 크기 변화 */
        }

        /*종이비행기*/
        [data-testid="stChatInputSubmitButton"] svg path {
            fill: #6f573c !important; /* 원하는 색상 코드 */
        }
        /*버튼 네모*/
        [data-testid="stChatInputSubmitButton"] svg rect {
            fill: #f5ead6 !important; /* 원하는 색상 코드 */
            border: none
        }
        
        /* 🔥 입력 버튼 hover 시 스타일 변경 */
        [data-testid="stChatInputSubmitButton"]:hover {
            background-color: #d4c0a1 !important; /* hover 시 밝은 갈색 */
            color: #f5ead6 !important; /* 아이콘 색상 */
            transform: scale(1.1) !important; /* 🔥 크기 1.1배 확대 */
        }

        /* hover시 네모*/
        [data-testid="stChatInputSubmitButton"]:hover svg rect {
            fill: #d4c0a1 !important; /* 원하는 색상 코드 */
            border: none
            outline: none !important; /* 포커스 시 테두리 제거 */
            box-shadow: none !important; /* 그림자 제거 */
        }

        .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
            display: flex;
            flex-direction: row-reverse;
            align-itmes: end;
        }
        
        .st-emotion-cache-janbn0 {
            flex-direction: row-reverse;
            text-align: right;
        }

        [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
        text-align: right;
    }


    </style>
""", unsafe_allow_html=True)






# ✅ 채팅 UI 헤더
st.markdown(
    """
    <style>
    .header {
        display: flex;
        align-items: center;
        background-color: transparent;
        padding: 15px 10px;
    }
    .header img {
        max-width: 100px; /* 이미지 크기 자동 조정 */
        height: auto;
        margin-right: 15px; /* 텍스트와 간격 */
    }
    .header h1 {
        margin: 0;
        color: #6f573c; /* 더 진한 갈색 */
        font-size: 28px; /* 글씨 크기 조정 */
        font-weight: bold; /* 글씨 두껍게 */
    }
    </style>
    
    <div class="header">
        <img src="https://i.namu.wiki/i/E4gAwg65fMroWtXG5POYiwcGseYpmfhrm9fYxCzSqXThXDMEG9yZAjkkq8_bQEkrIjAQZrQSObatdE-eDp86xQ.svg">
        <h1>건국대학교 교양과목 수강신청 맞춤 컨설팅 챗봇</h1>
    </div>
    """,
    unsafe_allow_html=True
)


if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", 
                                      "content": "안녕하세요! 건국대학교 교양과목 수강신청 맞춤 컨설팅 챗봇 RAKU입니다.<br>\
                                      교양과목과 관련하여 궁금하신 점이 있으시면 무엇이든 물어봐주세요!<br>\
                                      보다 정확한 답변을 얻고 싶으시다면 왼쪽 사이드바 내의 guideline을 참고해주세요 :)<br>\
                                      원하는 답변이 아닐 경우에는 재질문해주시면 감사하겠습니다.<br>\
                                      답변 시 시간이 다소 소요될 수 있는 점 양해 부탁드립니다.<br><br>\
                                      Hello! I am RAKU, the general elective course registration chatbot for Konkuk University students. <br>\
                                      If you have any questions regarding general elective courses, feel free to ask anything! <br>\
                                      If you would like to get a more accurate answer, please refer to the guideline in the left sidebar :) <br>\
                                      If my answer is not what you want, please feel free to ask me again about it.<br>\
                                      Please understand that it may take some time to get an answer."})


for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar='user_avatar.jpg'):
            # st.markdown(message['content'])
            st.markdown(
                f"""
                <style>
                .chat-contain {{
                    display: flex;
                    justify-content: flex-end;
                }}
                
                .chat-bubble {{
                    flex-direction: row-reverse;
                    display: inline-block;
                    max-width: 70%;
                    padding: 10px;
                    border-radius: 15px;
                    word-wrap: break-word;
                    text-align: left;
                    width: auto;
                }}
                    
                .user-message {{
                    background-color: #f9c66a;
                    color: #6f573c;
                    text-align: left;
                    }}
                </style>
    
                <div class="chat-contain">
                    <div class="chat-bubble user-message">{message['content']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    else:
        st.markdown("""
        <style>
        .chat-container {
            display: flex;
            align-items: flex-start;
            background-color: #005426;
            color: white; /* ✅ 글자 색상 지정 */
            padding: 15px;
            border-radius: 15px;
            max-width: 70%;
            margin-bottom: 10px;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }
            .chat-bubble {
                background-color: #005426;
                color: white;
                padding: 25px;
                border-radius: 15px;
                max-width: 70%;
                margin-bottom: 15px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                width: 100%;
                margin-left: 15px;
            }
            .bot-message {
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
                padding: 20px;
                line-height: 1.5;
            }
        </style>
    """, unsafe_allow_html=True)
        
        with st.chat_message("assistant", avatar="bot_avatar.png"):
            st.markdown(
                f"""
                <div class="chat-container">
                    <div class="chat-bubble bot-message">
                        {message['content']}
                """,
                unsafe_allow_html=True,
            )



if len(st.session_state.messages) > 1:
    st.markdown("""
        <div style="display: flex; align-items: center; text-align: center; margin: 20px 0;">
            <hr style="flex-grow: 1; border: 0.5px solid gray; margin: 0 10px;">
            <span style="white-space: nowrap; color: gray; font-size: 14px;">이전 대화</span>
            <hr style="flex-grow: 1; border: 0.5px solid gray; margin: 0 10px;">
        </div>
    """, unsafe_allow_html=True)




            
# ✅ 사용자 입력 받기
if user_input := st.chat_input("말씀해주세요."):
    with st.chat_message(message["role"], avatar='avatar_ivory.png'):
        st.markdown(
                f"""
                <style>
                .chat-contain {{
                    display: flex;
                    justify-content: flex-end;
                }}
                
                .chat-bubble {{
                    flex-direction: row-reverse;
                    display: inline-block;
                    max-width: 70%;
                    padding: 10px;
                    border-radius: 15px;
                    word-wrap: break-word;
                    text-align: left;
                    width: auto;
                }}
                    
                .user-message {{
                    background-color: #f9c66a;
                    color: #6f573c;
                    text-align: left;
                    }}
                </style>
    
                <div class="chat-contain">
                    <div class="chat-bubble user-message">{user_input}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ✅ 사용자 메시지 추가 (우측 정렬)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    
    import streamlit as st

    # ✅ CSS 적용
    st.markdown("""
        <style>=
            .chat-bubble {
                background-color: #005426;
                color: white;
                padding: 25px;
                border-radius: 15px;
                max-width: 70%;
                margin-bottom: 15px;
                word-wrap: break-word;
                overflow-wrap: break-word;
                width: 100%;
                margin-left: 15px;
            }
            .bot-message {
                white-space: normal;
                word-wrap: break-word;
                overflow-wrap: break-word;
                max-width: 100%;
                padding: 20px;
                line-height: 1.5;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # ✅ 예제 응답 (마크다운 그대로 사용)
    bot_response = query_rag(user_input)
    
    # ✅ 올바른 방식으로 출력
    with st.chat_message("assistant", avatar="bot_avatar.png"):
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="chat-bubble bot-message">
                    {bot_response}
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
