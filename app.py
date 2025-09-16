import streamlit as st
import os
from rfp_analyzer import RFPAnalyzer
from datetime import datetime
import json

# Streamlit 설정
st.set_page_config(
    page_title="RFP 분석 시스템",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("📋 RFP 분석 시스템")
    st.markdown("---")
    
    # 사이드바 - 서비스 설정
    with st.sidebar:
        st.header("⚙️ 서비스 설정")
        
        # 환경변수 로드
        from dotenv import load_dotenv
        load_dotenv()
        
        # Azure AI Search 설정
        search_endpoint = st.text_input("Azure AI Search 엔드포인트", value=os.getenv("AZURE_SEARCH_ENDPOINT", ""))
        search_key = st.text_input("Azure AI Search 키", value=os.getenv("AZURE_SEARCH_KEY", ""), type="password")
        search_index_name = st.text_input("검색 인덱스명", value=os.getenv("AZURE_SEARCH_INDEX_NAME", "rfp-index"))
        
        # Azure OpenAI 설정
        openai_endpoint = st.text_input("Azure OpenAI 엔드포인트", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        openai_key = st.text_input("Azure OpenAI 키", value=os.getenv("AZURE_OPENAI_KEY", ""), type="password")
        openai_api_version = st.text_input("API 버전", value=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))
        
        # 분석기 초기화
        if st.button("🔧 서비스 초기화"):
            try:
                st.session_state.analyzer = RFPAnalyzer(
                    search_endpoint=search_endpoint,
                    search_key=search_key,
                    search_index_name=search_index_name,
                    openai_endpoint=openai_endpoint,
                    openai_key=openai_key,
                    openai_api_version=openai_api_version
                )
                st.success("✅ 서비스가 초기화되었습니다!")
            except Exception as e:
                st.error(f"❌ 초기화 실패: {str(e)}")
    
    # 세션 상태 초기화
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "similar_rfps" not in st.session_state:
        st.session_state.similar_rfps = []
    if "qa_question" not in st.session_state:
        st.session_state.qa_question = ""
    if "qa_answer" not in st.session_state:
        st.session_state.qa_answer = ""
    if "example_question" not in st.session_state:
        st.session_state.example_question = ""
    
    # 메인 탭
    tab1, tab2, tab3, tab4 = st.tabs(["📄 RFP 분석", "❓ 질의응답", "🔍 유사 RFP 검색", "📝 제안서 생성"])
    
    with tab1:
        st.header("📄 RFP 파일 분석")
        
        if st.session_state.analyzer is None:
            st.warning("⚠️ 먼저 사이드바에서 서비스를 초기화해주세요.")
        else:
            # 파일 업로드
            uploaded_file = st.file_uploader("RFP 파일을 업로드하세요", type=['pdf', 'docx', 'txt'])
            
            if uploaded_file is not None:
                # 임시 파일로 저장
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ 파일이 업로드되었습니다: {uploaded_file.name}")
                
                # 분석 버튼
                if st.button("🔍 RFP 분석 시작"):
                    with st.spinner("RFP를 분석하고 있습니다..."):
                        try:
                            # 텍스트 추출 (제목도 함께 추출)
                            result = st.session_state.analyzer.extract_text_from_file(temp_file_path)
                            
                            if result and len(result) == 2:
                                rfp_content, pdf_title = result
                                
                                if rfp_content:
                                    # GPT 분석
                                    analysis_result = st.session_state.analyzer.analyze_rfp_with_gpt(rfp_content)
                                    
                                    if analysis_result:
                                        # Azure AI Search에 저장 (PDF 제목 포함)
                                        if st.session_state.analyzer.store_rfp_in_search(analysis_result, rfp_content, pdf_title):
                                            st.success("✅ RFP가 성공적으로 저장되었습니다!")
                                        
                                        st.session_state.analysis_result = analysis_result
                                        st.session_state.rfp_content = rfp_content
                                        st.session_state.pdf_title = pdf_title
                                        st.success("✅ RFP 분석이 완료되었습니다!")
                                        st.rerun()
                                    else:
                                        st.error("❌ RFP 분석에 실패했습니다.")
                                else:
                                    st.error("❌ 파일에서 텍스트를 추출할 수 없습니다.")
                            else:
                                st.error("❌ 파일에서 텍스트를 추출할 수 없습니다.")
                        except Exception as e:
                            st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
                        finally:
                            # 임시 파일 삭제
                            if os.path.exists(temp_file_path):
                                os.remove(temp_file_path)
                
                # 분석 결과 표시
                if st.session_state.analysis_result:
                    st.markdown("---")
                    st.subheader("📊 분석 결과")
                    st.json(st.session_state.analysis_result)
    
    with tab2:
        st.header("❓ 질의응답")
        
        if st.session_state.analysis_result is None:
            st.warning("❌ 먼저 RFP를 분석해주세요.")
        else:
            # 질문 입력
            question = st.text_area("RFP에 대해 질문하세요", value=st.session_state.qa_question, height=100)
            
            # 예시 질문 버튼들
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("예시 질문 1"):
                    st.session_state.example_question = "이 RFP의 주요 목적은 무엇인가요?"
                    st.rerun()
            with col2:
                if st.button("예시 질문 2"):
                    st.session_state.example_question = "프로젝트 일정은 어떻게 되어 있나요?"
                    st.rerun()
            with col3:
                if st.button("예시 질문 3"):
                    st.session_state.example_question = "예산 범위는 얼마인가요?"
                    st.rerun()
            
            if st.session_state.example_question:
                question = st.session_state.example_question
                st.session_state.example_question = ""
            
            # 질문 초기화 버튼
            if st.button("질문 초기화"):
                st.session_state.qa_question = ""
                st.session_state.qa_answer = ""
                st.rerun()
            
            # 답변 생성
            if st.button("답변 생성"):
                if question:
                    with st.spinner("답변을 생성하고 있습니다..."):
                        try:
                            answer = st.session_state.analyzer.ask_question_about_rfp(
                                question, 
                                st.session_state.rfp_content, 
                                st.session_state.analysis_result
                            )
                            st.session_state.qa_answer = answer
                            st.session_state.qa_question = question
                        except Exception as e:
                            st.error(f"❌ 답변 생성 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("⚠️ 질문을 입력해주세요.")
            
            # 답변 표시
            if st.session_state.qa_answer:
                st.markdown("---")
                st.subheader("💬 답변")
                st.write(st.session_state.qa_answer)
                
                # 답변 복사 버튼
                if st.button("답변 복사"):
                    st.write("답변이 클립보드에 복사되었습니다.")
    
    with tab3:
        st.header("🔍 유사 RFP 검색")
        
        if st.session_state.analyzer is None:
            st.warning("⚠️ 먼저 사이드바에서 서비스를 초기화해주세요.")
        else:
            search_query = st.text_input("검색어를 입력하세요")
            
            if st.button("🔍 유사 RFP 검색"):
                if search_query:
                    with st.spinner("유사 RFP를 검색하고 있습니다..."):
                        try:
                            similar_rfps = st.session_state.analyzer.search_similar_rfps(search_query)
                            st.session_state.similar_rfps = similar_rfps
                            
                            if similar_rfps:
                                st.success(f"✅ {len(similar_rfps)}개의 유사 RFP를 찾았습니다.")
                            else:
                                st.info("ℹ️ 유사한 RFP를 찾을 수 없습니다.")
                        except Exception as e:
                            st.error(f"❌ 검색 중 오류가 발생했습니다: {str(e)}")
                else:
                    st.warning("⚠️ 검색어를 입력해주세요.")
            
            # 검색 결과 표시
            if st.session_state.similar_rfps:
                st.markdown("---")
                st.subheader("📋 검색 결과")
                for i, rfp in enumerate(st.session_state.similar_rfps, 1):
                    with st.expander(f"{i}. {rfp['title']} (유사도: {rfp['score']:.2f})"):
                        st.write(f"**프로젝트 유형:** {rfp['project_type']}")
                        st.write(f"**요구사항:** {rfp['requirements']}")
                        st.write(f"**평가 기준:** {rfp['evaluation_criteria']}")
                        st.write(f"**생성일:** {rfp['created_date']}")
    
    # with tab4:
    #     st.header("📝 제안서 생성")
        
    #     if st.session_state.analysis_result is None:
    #         st.warning("❌ 먼저 RFP를 분석해주세요.")
    #     else:
            # 제안서 초안 생성 기능 주석 처리
            # if st.button("📝 제안서 초안 생성"):
            #     with st.spinner("제안서 초안을 생성하고 있습니다..."):
            #         similar_rfps = st.session_state.similar_rfps
            #         proposal_draft = st.session_state.analyzer.generate_proposal_draft(st.session_state.analysis_result, similar_rfps)
            #         
            #         if proposal_draft:
            #             st.markdown("---")
            #             st.subheader("📄 제안서 초안")
            #             st.markdown("---")
            #             st.text_area("제안서 내용", proposal_draft, height=400)
            #             
            #             # 파일 다운로드
            #             st.download_button(
            #                 label="💾 제안서 다운로드",
            #                 data=proposal_draft,
            #                 file_name=f"proposal_draft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            #                 mime="text/plain"
            #             )
            #         else:
            #             st.error("❌ 제안서 초안 생성에 실패했습니다.")
            
            # st.info("📝 제안서 초안 생성 기능이 비활성화되었습니다.")

if __name__ == "__main__":
    main()