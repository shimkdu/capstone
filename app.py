import streamlit as st
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from typing import List, Dict
from agent import run_graph, EvaluationVerdict # run_graph를 직접 호출

# 봇 만들기
st.set_page_config(page_title="🕵️ FakeNews", page_icon="🛡️", layout="wide")

# --- 헤더 섹션 ---
st.title("🕵️ FakeNews: AI 기반 팩트체크")
st.markdown("""
    뉴스 내용의 신뢰도를 **Gemini AI 에이전트**가 분석하고 평가합니다.
    🔗 **분석을 원하는 뉴스 URL**을 아래에 입력해 주세요.
""")

# --- 입력 섹션 ---
with st.container(border=True):
    query = st.text_input("🔗 뉴스 URL 입력", placeholder="예: https://www.chosun.com/politics/2025/10/27/...")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    if col2.button("🔍 신뢰도 확인하기", use_container_width=True, type="primary") and query.strip():
        
        if not (query.startswith("http://") or query.startswith("https://")):
            st.error("🚨 유효한 URL 형식이 아닙니다. 'http://' 또는 'https://'로 시작하는 주소를 입력해 주세요.")
            st.stop()

        with st.spinner("⏳ 팩트체크 에이전트가 뉴스를 분석하고 있습니다..."):
            try:
                result = run_graph(query) 
            except Exception as e:
                st.error(f"❌ LangGraph 실행 중 치명적인 오류가 발생했습니다: {type(e).__name__}")
                st.exception(e)
                st.stop()


        if result is None or 'verdict' not in result:
            st.error("🚨 **시스템 오류:** 분석 결과 객체를 생성하지 못했습니다. 입력 URL을 확인해주세요.")
            st.stop()

        # 소요 시간은 run_graph에서 계산된 후 result에 포함되어야 함 (현재는 N/A)
        st.success(f"✅ 분석 완료: AI 평가 결과입니다.") 
        
        verdict: EvaluationVerdict = result['verdict']
        overall_score = verdict.overall_fake_probability
        
        # --- 최종 평가 및 메트릭스 섹션 ---
        st.header("🛡️ 최종 신뢰도 평가")
        
        col_main, col_sub = st.columns([2, 1])

        with col_main:
            st.subheader("종합 허위 가능성")
            st.progress(overall_score)
            
            if overall_score >= 0.75:
                status_emoji = "🚨"
                st.error(f"{status_emoji} **{overall_score*100:.1f}% (높음)**: 가짜뉴스일 확률이 매우 높습니다. 정보 확산을 멈추세요.")
            elif overall_score >= 0.45:
                status_emoji = "⚠️"
                st.warning(f"{status_emoji} **{overall_score*100:.1f}% (보통)**: 주의가 필요합니다. 다른 출처를 통해 검증하세요.")
            else:
                status_emoji = "🟢"
                st.info(f"{status_emoji} **{overall_score*100:.1f}% (낮음)**: 신뢰도가 높습니다.")
            
            st.markdown(f"**최종 판단 요약**")
            st.caption(verdict.final_judgment)


        with col_sub:
            st.metric("과장 점수 (0.0=진실)", f"{verdict.exaggeration_score:.2f}")
            st.metric("출처 부족 점수 (0.0=충분)", f"{verdict.lack_of_sources_score:.2f}")
            st.metric("논리적 오류 점수 (0.0=논리적)", f"{verdict.logical_errors_score:.2f}")

        st.divider()

        # --- 상세 분석 섹션 (UI 정리) ---
        st.header("🔎 에이전트 분석 상세 내역")
        
        # 1. 팩트체크 최종 근거
        with st.expander("📝 팩트체크 최종 결과 및 근거 보기", expanded=True):
            st.markdown(result['fact_check'])

        # 2. 분석 과정 (검색 쿼리만 남김)
        with st.expander("🔬 분석 과정 (Analysis Flow)"): 
            st.subheader("① 검색 쿼리 목록")
            st.write(result['search_queries'])

        # 3. 검색 결과 출처
        with st.expander("📰 검색 결과 출처 (AI가 검증에 사용한 자료)"):
            if result['article_result']:
                for idx, article in enumerate(result['article_result']):
                    st.markdown(f"""
                        **{idx+1}. {article['title']}**
                        > *{article['summary']}*
                        
                        [원문 보기]({article['source_url']})
                        ---
                    """)
            else:
                st.info("관련 기사를 찾지 못하여 외부 검증 없이 판단되었습니다.")

        st.divider()
        
        st.subheader("📊 항목별 상세 점수 및 근거")
        
        st.markdown(f"**과장 (Exaggeration): {verdict.exaggeration_score:.2f}**")
        st.caption(f"근거: {verdict.exaggeration_reasoning}")
        
        st.markdown(f"**출처 부족 (Lack of sources): {verdict.lack_of_sources_score:.2f}**")
        st.caption(f"근거: {verdict.lack_of_sources_reasoning}")
        
        st.markdown(f"**논리적 오류 (Logical errors): {verdict.logical_errors_score:.2f}**")
        st.caption(f"근거: {verdict.logical_errors_reasoning}")
        
    else: 
        st.warning("뉴스 URL을 입력하고 버튼을 눌러주세요.")