# =========================================================================
# 🤖 파일명: app.py (v27 - '일체형' + 'Firebase DB' + '개발자 툴')
# =========================================================================
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import warnings
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

warnings.filterwarnings('ignore')

# --- 1. AI 모델(v1.4.2) 로드 ---
MODEL_VERSION = "v1"
model_filename_global = 'leisure_satisfaction_model.joblib'
loaded_model = joblib.load(model_filename_global)
print(f"✅ AI 서버가 모델({MODEL_VERSION})을 탑재했습니다.")

# --- 2. AI 모델의 변수 목록 (필수) ---
final_predictor_vars = [
    '성별', '나이', '거주지', '결혼상태', '직업', '가구월소득', 
    '전반적여가생활만족도_인프라', '문화예술스포츠참여지역', '참여여가활동1순위',
    'B0101020802', '문화예술스포츠비용금액', '문화예술스포츠참여동반자',
    '전반적여가생활만족도_시간', '여가목적1순위', '문화예술스포츠참여빈도'
]
categorical_cols = [
    '성별', '거주지', '결혼상태', '직업', '참여여가활동1순위',
    '문화예술스포츠참여동반자', '여가목적1순위',
    '문화예술스포츠참여지역'
]
SEARCH_SPACE = {
    "purpose": [('가족/지인', '1'), ('건강', '2'), ('남는 시간', '3'), 
                ('대인관계', '4'), ('휴식', '5'), ('스트레스 해소', '6'), 
                ('자기계발', '7'), ('자기만족/즐거움', '8'), ('기타', '9')],
    "activity": [('문화예술관람', '1'), ('문화예술직접', '2'), ('스포츠관람', '3'), 
                 ('스포츠직접', '4'), ('관광/여행', '5'), ('오락/휴식', '6'), 
                 ('자기계발', '7'), ('사회교류', '8'), ('기타', 'nan')],
    "partner": [('혼자', '1'), ('가족/친척', '2'), ('친구', '3'), ('연인', '4'), 
                ('직장동료', '5'), ('동호회', '6'), ('기타', '7')]
}
PURPOSE_MAP = {v: k for k, v in SEARCH_SPACE["purpose"]}
ACTIVITY_MAP = {v: k for k, v in SEARCH_SPACE["activity"]}
PARTNER_MAP = {v: k for k, v in SEARCH_SPACE["partner"]}

# --- 3. [v27] 개발자 대시보드 비밀번호 ---
# (이 비밀번호를 기억하세요. /admin 접속 시 필요)
ADMIN_PASSWORD = "0706" 

# --- 4. [v27] Firebase DB 연결 ---
try:
    # ⚠️ [필수!] 2단계에서 다운로드한 '비밀 키' 파일명으로 수정하세요.
    cred = credentials.Certificate("curation-5e526-firebase-adminsdk-fbsvc-0a615d5244.json") 
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    db_collection = db.collection('user_inputs') # AI가 성장할 '새 문제집'
    print("✅ Firebase DB 연결 성공. 'user_inputs' 컬렉션에 데이터를 저장합니다.")
except Exception as e:
    print(f"🚨 Firebase 연결 실패: {e}")
    print("   'serviceAccountKey.json' 파일이 app.py와 같은 폴더에 있는지 확인하세요.")

# --- 5. Flask 서버 앱 생성 ---
app = Flask(__name__, template_folder='templates')
CORS(app) 

# --- 6. AI 예측 헬퍼 함수 (v26과 동일) ---
def get_prediction(current_inputs):
    data = {col: [np.nan] for col in final_predictor_vars}
    for key, value in current_inputs.items():
        if key in data: data[key] = [value]
    for col in categorical_cols:
         if col in data: data[col] = [str(data[col][0])]
    input_df = pd.DataFrame(data, columns=final_predictor_vars)
    prob_5star = loaded_model.predict_proba(input_df)[0][1]
    return prob_5star

# --- 7. 공개용 '메뉴판' (v26과 동일) ---
@app.route('/', methods=['GET'])
def home():
    """손님이 '온라인 주소'('/')로 접속하면, 'templates/index.html' 파일을 보여줍니다."""
    return render_template('index.html')

# --- 8. 공개용 'AI 예측 API' (v26 + Firebase 저장) ---
@app.route('/predict', methods=['POST'])
def predict_and_recommend():
    global MODEL_VERSION
    try:
        inputs = request.json
        
        # 1. '현재 확률' 계산
        current_inputs = {
            '나이': inputs.get('age'), '직업': inputs.get('job'), '성별': inputs.get('gender'),
            '결혼상태': inputs.get('marriage'), '가구월소득': inputs.get('income'),
            '여가목적1순위': inputs.get('purpose'), '참여여가활동1순위': inputs.get('activity'),
            '문화예술스포츠참여동반자': inputs.get('partner'), 
            '전반적여가생활만족도_인프라': inputs.get('infra_sat'),
            '전반적여가생활만족도_시간': inputs.get('time_sat'),
            '거주지': "서울", '문화예술스포츠비용금액': 50000, '문화예술스포츠참여빈도': 1.5,
            'B0101020802': 0, '문화예술스포츠참여지역': np.nan
        }
        baseline_prob = get_prediction(current_inputs)
        baseline_prob_pct = baseline_prob * 100
        
        # 2. '페르소나' 정의
        if baseline_prob > 0.4: persona = "🏆 안정적 전문가"
        elif inputs.get('purpose') in ['3', '9'] or inputs.get('activity') == '6': persona = "🧭 이탈 위험군"
        else: persona = "🌱 성장형 탐험가"

        # 3. '여정 시뮬레이터' 실행
        simulation_results = []
        fixed_inputs = current_inputs.copy()
        for p_name, p_code in SEARCH_SPACE["purpose"]:
            for a_name, a_code in SEARCH_SPACE["activity"]:
                for t_name, t_code in SEARCH_SPACE["partner"]:
                    sim_inputs = fixed_inputs.copy()
                    sim_inputs['여가목적1순위'] = p_code
                    sim_inputs['참여여가활동1순위'] = a_code
                    sim_inputs['문화예술스포츠참여동반자'] = t_code
                    sim_prob = get_prediction(sim_inputs)
                    simulation_results.append((sim_prob, p_name, a_name, t_name))
        
        simulation_results.sort(key=lambda x: x[0], reverse=True)
        
        # 4. 최종 추천 멘트 생성
        recommendations = [f"AI가 {len(simulation_results)}개의 모든 여가 조합을 시뮬레이션 했습니다."]
        recommendations.append(f"귀하의 고정 정보(나이, 직업, 성별 등)를 기준으로,\n5점 만족 확률이 가장 높은 **Top 3 궤적**은 다음과 같습니다.")
        
        for i in range(3):
            prob, p_name, a_name, t_name = simulation_results[i]
            recommendations.append(f"**🥇 {i+1}순위 (예상: {prob*100:.1f}%)**\n   - **목적:** {p_name}\n   - **활동:** {a_name}\n   - **동반자:** {t_name}")
        
        # 5. [v27] '지속적 학습'을 위해 Firebase DB에 데이터 저장
        try:
            db_data = current_inputs.copy()
            db_data['timestamp'] = firestore.SERVER_TIMESTAMP
            db_data['predicted_prob_5star'] = baseline_prob
            db_data['persona'] = persona
            db_collection.add(db_data) # 'user_inputs'에 새 문서 추가
        except Exception as e:
            print(f"🚨 Firebase DB 저장 실패: {e}")
            
        # 6. 웹사이트에 JSON으로 결과 응답
        return jsonify({
            "success": True, "model_version": MODEL_VERSION,
            "probability_5star_percent": round(baseline_prob_pct, 2),
            "persona": persona, "recommendations": "\n\n".join(recommendations)
        })
        
    except Exception as e:
        print(f"🚨 예측 중 오류 발생: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# --- 9. [v27] '개발자용 비밀 대시보드' ---
@app.route('/admin', methods=['GET', 'POST'])
def admin_dashboard():
    # '개발자 툴' v24의 로직을 여기에 통합
    
    # 1. 암호 입력 폼 보여주기 (GET 요청)
    if request.method == 'GET':
        return '''
            <style>body { font-family: sans-serif; background: #f4f4f4; } .container { max-width: 800px; margin: 2rem auto; padding: 2rem; background: white; border-radius: 8px; } input { width: 100%; padding: 8px; box-sizing: border-box; } button { padding: 10px 15px; background: #0D9488; color: white; border: none; border-radius: 5px; cursor: pointer; }</style>
            <div class="container">
                <h2>🕵️ 개발자용 대시보드</h2>
                <form method="POST">
                    <label for="password">비밀번호:</label>
                    <input type="password" id="password" name="password">
                    <br><br>
                    <button type="submit">로그인</button>
                </form>
            </div>
        '''

    # 2. 암호 확인 및 리포트 생성 (POST 요청)
    if request.form.get('password') != ADMIN_PASSWORD:
        return '<script>alert("비밀번호가 틀렸습니다."); window.history.back();</script>'

    # --- (v24) 개발자 툴 1: 누적 데이터 확인 ---
    try:
        docs = db_collection.stream() # Firebase에서 모든 '새 문제집' 데이터 가져오기
        df_new = pd.DataFrame([doc.to_dict() for doc in docs])
        
        if df_new.empty:
            return "<div class='container'><h2>아직 AI가 학습한 새 데이터가 없습니다.</h2></div>"

        # 수치형/범주형 변수 분리
        numeric_cols_in_db = [col for col in numeric_cols if col in df_new.columns]
        categorical_cols_in_db = [col for col in categorical_cols if col in df_new.columns]
        
        summary_numeric = df_new[numeric_cols_in_db].describe()
        summary_categorical = df_new[categorical_cols_in_db].describe()
        
        report_html = f"""
            <style>body {{ font-family: sans-serif; }} .container {{ max-width: 1200px; margin: 2rem auto; }} table {{ border-collapse: collapse; width: 100%; margin-bottom: 1rem; }} th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }} th {{ background-color: #f2f2f2; }}</style>
            <div class="container">
                <h2>🕵️ 개발자용 대시보드 (v27)</h2>
                <h3>1. 누적 데이터 현황 (Firebase: 'user_inputs')</h3>
                <p>- <strong>총 {len(df_new)} 건</strong>의 새 데이터가 Firebase DB에 축적되었습니다.</p>
                <p>- (참고: 이 데이터는 '지속적 학습'에 사용될 수 있습니다.)</p>
                
                <h4>2. 수치형 변수 요약</h4>
                {summary_numeric.to_html()}
                
                <h4>3. 범주형 변수 요약 (최빈값)</h4>
                {summary_categorical.to_html()}
        """
        
    # --- (v24) 개발자 툴 2: 5점 만족자 분석 ---
        threshold = 0.5
        # [v27] Firebase에 저장된 예측 확률 사용
        df_5star_predicted = df_new[df_new['predicted_prob_5star'] >= threshold].copy()
        
        if df_5star_predicted.empty:
            report_html += "<h3>2. 5점 만족자 분석 (이론 검증)</h3>"
            report_html += f"<p>새 데이터 {len(df_new)}건 중 5점 만족으로 예측되는(50% 이상) 사용자가 아직 없습니다.</p>"
        else:
            purpose_analysis = df_5star_predicted['여가목적1순위'].map(PURPOSE_MAP).value_counts(normalize=True).to_frame().to_html()
            activity_analysis = df_5star_predicted['참여여가활동1순위'].map(ACTIVITY_MAP).value_counts(normalize=True).to_frame().to_html()
            partner_analysis = df_5star_predicted['문화예술스포츠참여동반자'].map(PARTNER_MAP).value_counts(normalize=True).to_frame().to_html()
            
            report_html += f"""
                <h3>2. 5점 만족자 분석 (이론 검증)</h3>
                <p>- 총 {len(df_new)}건의 신규 데이터 중 <strong>{len(df_5star_predicted)}명</strong>이 5점 만족(예측 확률 {threshold*100}%) 그룹으로 분류되었습니다.</p>
                
                <h4>(근거논문 비교) '예측 5점' 그룹의 주요 여가 목적</h4>
                {purpose_analysis}
                
                <h4>(근거논문 비교) '예측 5점' 그룹의 주요 여가 활동</h4>
                {activity_analysis}
                
                <h4>(근거논문 비교) '예측 5점' 그룹의 주요 동반자</h4>
                {partner_analysis}
            """
        
        report_html += "</div>"
        return report_html
        
    except Exception as e:
        return f"<h2>분석 중 오류 발생</h2><p>{e}</p>"


# --- 10. 서버 실행 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)