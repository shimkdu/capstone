import subprocess
import time
import sys
import os

def run():
    print('스트림릿 앱 실행 준비중...')
    
    # GEMINI_API_KEY 환경 변수가 설정되어 있는지 확인
    if 'GEMINI_API_KEY' not in os.environ:
        print("🚨 오류: GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
        print("스크립트를 실행하기 전에 'set GEMINI_API_KEY=\"YOUR_API_KEY\"' 명령어로 키를 설정해주세요.")
        print("Windows PowerShell의 경우 '$env:GEMINI_API_KEY=\"YOUR_API_KEY\"'를 사용하세요.")
        sys.exit(1)
        
    print('✅ GEMINI_API_KEY 확인 완료. 스트림릿 앱을 실행합니다.')
        
    # 스트림릿 앱 실행
    try:
        # Streamlit은 앱이 종료될 때까지 블로킹됩니다.
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except Exception as e:
        print(f"스트림릿 앱 실행 오류: {e}")

if __name__ == '__main__':
    run()