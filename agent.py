from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from googlenewsdecoder import new_decoderv1
from gnews import GNews
from newspaper import Article
from urllib.parse import urlparse
from pydantic.v1 import BaseModel, Field 
import json 
import re 
import os 
import requests 

# --- Selenium 라이브러리 임포트 ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# SSL 인증서 검증 오류 우회를 위한 requests 설정
requests.packages.urllib3.disable_warnings()

# --- LLM 설정 (Gemini API 사용, temperature=0.0) ---
MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, api_key=GEMINI_API_KEY)
llm_json = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0.0, response_mime_type="application/json", api_key=GEMINI_API_KEY) 

# --- Pydantic 스키마 정의 ---
class EvaluationVerdict(BaseModel):
    exaggeration_score: float = Field(..., description="과장 점수 (0.0=진실, 1.0=거짓)")
    exaggeration_reasoning: str = Field(..., description="과장 점수에 대한 간략한 근거 (1-2문장)")
    lack_of_sources_score: float = Field(..., description="출처 부족 점수 (0.0=진실, 1.0=거짓)")
    lack_of_sources_reasoning: str = Field(..., description="출처 부족 점수에 대한 간략한 근거 (1-2문장)")
    logical_errors_score: float = Field(..., description="논리적 오류 점수 (0.0=진실, 1.0=거짓)")
    logical_errors_reasoning: str = Field(..., description="논리적 오류 점수에 대한 간략한 근거 (1-2문장)")
    overall_fake_probability: float = Field(..., description="전체 허위 가능성 점수 (0.0=진실, 1.0=거짓)")
    final_judgment: str = Field(..., description="점수를 종합한 최종 판단 요약 문장")


# --- State 정의 ---
class NewsState(TypedDict):
    input_type: str 
    input: str
    article_title: str
    article_text: str 
    article_result: List[dict]
    search_queries: List[str]
    keyword_summary: str
    fact_check_draft: str
    fact_check: str
    verdict: EvaluationVerdict 
    reference: str 

# --- 0. URL에서 기사 본문 추출 (⭐ 네이트 뉴스(#article_body) 추가) ---
def extract_article_text(state: NewsState):
    print("\n[Node 0: extract_article_text] 🕵️ 기사 본문 추출 시도 (Selenium)...")
    if state['input_type'] == 'text':
        print("...오류: URL만 입력해야 합니다. 텍스트 입력을 차단합니다.")
        state['article_text'] = ""
        state['keyword_summary'] = "추출된_기사_없음"
        state['fact_check'] = "URL이 아닌 텍스트가 입력되어 분석을 진행할 수 없습니다."
        return state

    url = state['input']
    
    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
    
    driver = None 
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
        driver.delete_all_cookies()
        driver.get(url)
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "meta[property='og:title']"))
        )
        title = driver.find_element(By.CSS_SELECTOR, "meta[property='og:title']").get_attribute('content')

        # ⭐ 본문 추출 (네이버, 다음, 네이트 순으로 시도)
        extracted_text = ""
        try:
            main_content = driver.find_element(By.CSS_SELECTOR, "#articleBodyContents")
            extracted_text = main_content.text
            print("...네이버 뉴스 본문(#articleBodyContents) 텍스트 직접 추출 성공.")
        except Exception:
            try:
                main_content = driver.find_element(By.CSS_SELECTOR, "#dic_area")
                extracted_text = main_content.text
                print("...다음 뉴스 본문(#dic_area) 텍스트 직접 추출 성공.")
            except Exception:
                try:
                    # ⭐ 네이트 뉴스 본문 컨테이너 추가
                    main_content = driver.find_element(By.CSS_SELECTOR, "#article_body") 
                    extracted_text = main_content.text
                    print("...네이트 뉴스 본문(#article_body) 텍스트 직접 추출 성공.")
                except Exception:
                    # 위 세 방식이 모두 실패하면 Newspaper3k로 최후의 시도
                    print("...특정 컨테이너를 찾지 못해 Newspaper3k로 파싱 시도.")
                    html = driver.page_source
                    article = Article(url)
                    article.set_html(html) 
                    article.parse()
                    extracted_text = article.text

        if len(extracted_text) < 30: 
            raise ValueError("추출된 기사 본문의 길이가 너무 짧거나 내용이 부실합니다.")
            
        state['article_title'] = title
        state['article_text'] = extracted_text 
        print(f"...본문 추출 성공. (제목: {title})")
        
    except Exception as e:
        print(f"URL에서 기사 본문 추출 에러 발생: {e}")
        state['article_title'] = ""
        state['article_text'] = "" 
        state['keyword_summary'] = "추출된_기사_없음"
        state['fact_check'] = "URL에서 기사 본문 추출에 실패했거나 내용이 부실합니다. 팩트체크를 진행할 수 없습니다."
    
    finally:
        if driver:
            driver.quit() 
            
    return state


# --- 1. 초기 키워드 추출 ---
def extract_initial_keyword(state: NewsState):
    print("\n[Node 1: extract_initial_keyword] 🧠 Gemini API로 초기 키워드 추출 중...")
    title = state['article_title']
    if not title or title == "" or state['keyword_summary'] == "추출된_기사_없음":
        print("...제목이 없어 키워드 추출을 건너뜁니다.")
        return state 
        
    prompt = ChatPromptTemplate([('system', '당신은 외부 지식을 전혀 사용하지 않고, 오직 입력된 텍스트 "그대로" 키워드를 추출하는 기계적인 분석가입니다. 환각은 엄격히 금지됩니다.'),
    ('human', '''
        주어진 "기사 제목:"에서 **핵심 인물, 사건, 장소**를 중심으로 검색 키워드를 2~3개 추출하세요.

        **!!절대적인 규칙!!:**
        1. **오직 "기사 제목:" 안에 명시적으로 "존재하는 단어"만 사용하세요.**
        2. "기사 제목:"에 없는 단어를 절대로 연상하거나 추측하여 추가하지 마세요.
        3. 최종 출력은 추출된 키워드만 공백으로 구분하여 한 줄로 제공하세요.

        기사 제목: {title}
    ''')])

    chain = prompt | llm | StrOutputParser()
    raw_query = chain.invoke({'title': title}).strip()
    
    initial_query = " ".join(raw_query.split()) 
    
    state['keyword_summary'] = initial_query
    state['search_queries'] = [initial_query]
    print(f"...추출된 키워드: {initial_query}")
    return state

# --- 2. 뉴스 검색 및 요약 공통 로직 (⭐ 네이트 뉴스(#article_body) 추가) ---
def _search_and_summarize(state: NewsState):
    query = state['keyword_summary']
    if query == "추출된_기사_없음":
        return state
        
    def decode_url(url):
        interval_time = 5 
        try:
            decoded_url = new_decoderv1(url, interval=interval_time)
            return decoded_url["decoded_url"] if decoded_url.get("status") else None
        except Exception as e:
            print(f"URL 디코딩 중 에러 발생: {e}") 
            return None

    print(f"...GNews API로 '{query}' 검색 중...")
    google_news = GNews(language='ko', country='KR', max_results=3) 
    search_query = query.replace('+', ' ') 
    resp = google_news.get_news(search_query)

    chrome_options = Options()
    chrome_options.add_argument("--headless") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--log-level=3")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36")
    
    driver = None
    try:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=chrome_options)
    except Exception as e:
        print(f"🚨 크롬 드라이버 로드 실패: {e}")
        return state 

    article_list = []
    for item in resp:
        try:
            url = decode_url(item['url'])
            if not url: continue

            driver.delete_all_cookies()
            driver.get(url)
            
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "meta[property='og:title']"))
            )
            title = driver.find_element(By.CSS_SELECTOR, "meta[property='og:title']").get_attribute('content')
            
            # ⭐ 검색된 기사들도 동일하게 본문 영역 직접 지정
            extracted_text = ""
            try:
                main_content = driver.find_element(By.CSS_SELECTOR, "#articleBodyContents")
                extracted_text = main_content.text
            except Exception:
                try:
                    main_content = driver.find_element(By.CSS_SELECTOR, "#dic_area")
                    extracted_text = main_content.text
                except Exception:
                    try:
                        # ⭐ 네이트 뉴스 본문 컨테이너 추가
                        main_content = driver.find_element(By.CSS_SELECTOR, "#article_body")
                        extracted_text = main_content.text
                    except Exception:
                        print(f"    - [{url}] 특정 컨테이너 찾기 실패. Newspaper3k 폴백 사용.")
                        html = driver.page_source
                        article = Article(url)
                        article.set_html(html)
                        article.parse()
                        extracted_text = article.text
            
            if len(extracted_text) > 50:
                print(f"...'{title}' 기사 요약 중...")
                article_summary_prompt = ChatPromptTemplate([
                    ('system', '다음 기사를 3문장 이내로 핵심만 간결하게 요약하세요.'),
                    ('human', '기사: {text}')
                ])
                summary_chain = article_summary_prompt | llm | StrOutputParser()
                summary = summary_chain.invoke({'text': extracted_text})
            else:
                summary = "기사 본문 추출 실패 또는 내용 부족으로 요약 불가."

            article_list.append({
                'title': title, 
                'summary': summary.strip(),
                'source_url': url
            })
        except Exception as e:
            print(f'개별 기사 처리 중 에러 발생: {e}')
    
    if driver:
        driver.quit() 
        
    print(f"...검색/요약 완료. 총 {len(article_list)}개 기사 처리.")
    state['article_result'] = article_list
    return state

# 2-1. 1차 뉴스 검색
def search_initial(state: NewsState):
    print(f"\n[Node 2: search_initial] 🔍 1차 뉴스 검색 시도 (쿼리: {state['keyword_summary']})...")
    return _search_and_summarize(state)


# --- 3. 검색 실패 시 키워드 정제 ---
def refine_keyword(state: NewsState):
    print("\n[Node 3: refine_keyword] 🔄 1차 검색 실패. 키워드 정제 시도...")
    current_query = state['search_queries'][-1]
    
    prompt = ChatPromptTemplate([
        ('system', '당신은 검색 실패를 복구하는 검색어 정제 전문가입니다. 최초 검색어가 너무 구체적이어서 결과가 나오지 않았습니다.'),
        ('human', '''
            최초 쿼리: "{current_query}"

            지침:
            1. 위 쿼리에서 **핵심 사건(인물+행동)**이 무엇인지 파악하세요.
            2. 너무 세부적인 장소, 브랜드 이름, 수식어 등은 **제거**하여 검색 범위를 넓히세요.
            3. **핵심 사건**을 가장 잘 나타내는 새로운 검색어를 만드세요.
            4. 출력은 정제된 키워드만 공백으로 구분하여 제시하고, 다른 설명은 붙이지 마세요.

            예시 1:
            최초 쿼리: "이시영 둘째 출산 2주 5천만원 조리원 꿈의 집 LG전자"
            정제된 키워드: "이시영 출산"

            예시 2:
            최초 쿼리: "도널드 트럼프 블라디미르 푸틴 정상회담 알래스카"
            정제된 키워드: "트럼프 푸틴 정상회담"
        ''')
    ])
    
    chain = prompt | llm | StrOutputParser()
    raw_query = chain.invoke({'current_query': current_query}).strip()
    
    refined_query = " ".join(raw_query.split())
    
    state['keyword_summary'] = refined_query
    state['search_queries'].append(refined_query) 
    print(f"...정제된 키워드: {refined_query}")
    return state

# --- 4. 2차 뉴스 검색 ---
def search_refined(state: NewsState):
    print(f"\n[Node 4: search_refined] 🔍 2차 뉴스 검색 시도 (쿼리: {state['keyword_summary']})...")
    return _search_and_summarize(state)


# --- 5. 팩트체크 초안 생성 ---
def generate_draft(state: NewsState):
    print("\n[Node 5: generate_draft] 📝 팩트체크 초안 생성 중...")
    original_title = state['article_title']
    original_text = state['article_text']
    article_result = state['article_result']
    
    if not article_result:
        print("...검색된 기사가 없어 '판단 불가' 초안 생성.")
        state['fact_check'] = f"**{state['search_queries']}** 키워드로 구글 뉴스 검색 결과, 관련 기사를 찾을 수 없습니다. 뉴스 검색 결과 없이는 팩트체크 판단이 불가능합니다. 정보의 출처와 신뢰도를 직접 확인해 보세요."
        return state

    prompt = ChatPromptTemplate([
        ('system','당신은 전문 팩트체커입니다. 검색된 근거를 바탕으로 사실 여부 판단 초안을 작성하세요.'),
        ('human', '''
            다음 '원본 기사'와 '뉴스 검색 결과(요약)'를 기반으로 사실 여부를 판단하고 상세히 서술한 **최종 결과**를 작성하세요.

            원본 기사 제목: {original_title}
            원본 기사 본문: {original_text}
            
            뉴스 검색 결과(요약): {article_result}

            지침:
            1. '원본 기사'의 핵심 주장이 '뉴스 검색 결과'와 일치하는지 비교 분석하세요.
            2. '원본 기사'가 사실인지 거짓인지 최종 결론을 내리세요.
    ''')])

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        'original_title': original_title,
        'original_text': original_text,
        'article_result': article_result 
    })

    state['fact_check'] = result 
    print("...최종 결과 텍스트 생성 완료.")
    return state


# --- 7. 평가 ---
def evaluate(state: NewsState):
    print("\n[Node 7: evaluate] ⚖️ 최종 평가 및 점수 산출 중 (JSON Mode)...")
    fact_result = state['fact_check']
    
    error_reasoning = "분석 불가 또는 LLM 오류로 근거 생성 실패"
    
    try:
        if "판단이 불가능합니다." in fact_result:
            print("...판단 불가 상태로 최종 평가.")
            state['verdict'] = EvaluationVerdict(
                exaggeration_score=0.5,
                exaggeration_reasoning="판단 근거가 부족하여 점수를 0.5로 설정합니다.",
                lack_of_sources_score=1.0, 
                lack_of_sources_reasoning="검색된 관련 기사가 없어 출처 부족 점수를 1.0으로 설정합니다.",
                logical_errors_score=0.5,
                logical_errors_reasoning="판단 근거가 부족하여 점수를 0.5로 설정합니다.",
                overall_fake_probability=0.7,
                final_judgment=fact_result
            )
            return state

        json_schema_str = EvaluationVerdict.schema_json(indent=2)
        escaped_json_schema_str = json_schema_str.replace('{', '{{').replace('}', '}}')

        prompt = ChatPromptTemplate([
            ('system', f'''당신은 가짜 뉴스 탐지 전문가입니다. 다음 팩트체크 결과를 기반으로 뉴스 신뢰도를 평가하고, **반드시** JSON 형식으로 점수를 출력하세요. JSON은 아래 스키마를 완벽하게 따라야 합니다.

    스키마:
    {escaped_json_schema_str}
    '''), 
            ('human', '''
        다음 팩트체크 결과를 기반으로 뉴스의 신뢰도를 평가하고, 각 항목 점수를 0.0~1.0 사이로 배점하세요.
        0점에 가까우면 진실이고, 1점에 가까우면 거짓입니다.

        팩트체크 결과: {fact_result}
        
        결과를 바탕으로 다음 항목에 대한 점수와 **각 점수에 대한 간략한 근거(1-2문장)**를 정확하게 판단하고, 최종 판단 문장을 작성하세요.
        **주의:** 출력은 반드시 유효한 JSON 객체여야 하며, 어떤 설명이나 추가 텍스트 없이 JSON 객체만을 출력해야 합니다.
    ''')])
        
        chain = prompt | llm_json | StrOutputParser()
        
        json_string = chain.invoke({'fact_result': fact_result})
        
        if json_string.strip().startswith("```json"):
            json_string = json_string.strip()[7:-3].strip()
        
        result_dict = json.loads(json_string)
        state['verdict'] = EvaluationVerdict(**result_dict) 
        
        print("...JSON 평가 및 점수 산출 완료.")
        
    except Exception as e:
        print(f"JSON 처리/LLM 호출 최종 오류 발생: {e}")
        state['verdict'] = EvaluationVerdict(
            exaggeration_score=1.0, 
            exaggeration_reasoning=error_reasoning,
            lack_of_sources_score=1.0,
            lack_of_sources_reasoning=error_reasoning,
            logical_errors_score=1.0,
            logical_errors_reasoning=error_reasoning,
            overall_fake_probability=1.0, 
            final_judgment=f"LLM 호출 실패 또는 JSON 파싱 오류 발생: {e.__class__.__name__}"
        )
            
    return state

# --- 8. 검색 결과에 따른 라우팅 로직 ---
def route_on_search_result(state: NewsState):
    print("\n[Router] 🧭 검색 결과 라우팅...")
    if state['keyword_summary'] == "추출된_기사_없음":
        print("...기사 추출 실패. 평가로 즉시 이동.")
        return "skip_all" 
    if state['article_result']:
        print("...1차 검색 성공. 초안 생성으로 이동.")
        return "search_success" 
    else:
        print("...1차 검색 실패. 키워드 정제로 이동.")
        return "search_fail" 

# --- Graph Build and Run ---
def run_graph(input_data: str):
    """사용자 입력을 받아 전체 그래프를 실행하고 최종 결과를 반환합니다."""
    
    builder = StateGraph(NewsState)
    builder.add_node('extract_article_text', extract_article_text)
    builder.add_node('extract_initial_keyword', extract_initial_keyword)
    builder.add_node('search_initial', search_initial)
    builder.add_node('refine_keyword', refine_keyword)
    builder.add_node('search_refined', search_refined)
    builder.add_node('generate_draft', generate_draft)
    builder.add_node('evaluate', evaluate)

    builder.set_entry_point('extract_article_text') 
    builder.add_edge("extract_article_text", "extract_initial_keyword")
    builder.add_edge("extract_initial_keyword", "search_initial")
    
    builder.add_conditional_edges(
        "search_initial", 
        route_on_search_result, 
        {
            "search_success": "generate_draft",
            "search_fail": "refine_keyword",
            "skip_all": "evaluate" 
        }
    )
    
    builder.add_edge("refine_keyword", "search_refined")
    builder.add_edge("search_refined", "generate_draft")
    builder.add_edge("generate_draft", "evaluate")
    builder.add_edge("evaluate", END)

    graph = builder.compile()

    initial_state = NewsState(
        input_type='url',
        input=input_data,
        article_title="",
        article_text="",
        article_result=[],
        search_queries=[],
        keyword_summary="",
        fact_check="",
        verdict=None,
        reference="",
    ) 
    
    return graph.invoke(initial_state)