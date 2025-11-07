# =========================================================================
# 🤖 파일명: app.py (v26 - AI 서버와 웹사이트 일체형)
# =========================================================================
import joblib
import pandas as pd
import numpy as np
# [v26] render_template: 'index.html'을 메뉴판으로 나눠주기 위해
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import warnings
import os

warnings.filterwarnings('ignore')

# --- AI 모델(v1) 로드 ---
MODEL_VERSION = "v1"
model_filename_global = 'leisure_satisfaction_model.joblib'
loaded_model = joblib.load(model_filename_global)
print(f"✅ AI 서버가 모델({MODEL_VERSION})을 탑재했습니다.")

# --- AI 모델의 변수 목록 (필수) ---
# (1단계 Colab 훈련 코드와 100% 동일해야 함)
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
# 시뮬레이션용 검색 공간
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

# --- Flask 서버 앱 생성 ---
# [v26] 'templates' 폴더에서 index.html을 찾도록 설정
app = Flask(__name__, template_folder='templates')
CORS(app) # 모든 '온라인 주소' (Netlify 등)의 접속을 허용

# --- AI 예측 헬퍼 함수 ---
def get_prediction(current_inputs):
    data = {col: [np.nan] for col in final_predictor_vars}
    for key, value in current_inputs.items():
        if key in data: data[key] = [value]
    for col in categorical_cols:
         if col in data: data[col] = [str(data[col][0])]
    input_df = pd.DataFrame(data, columns=final_predictor_vars)
    prob_5star = loaded_model.predict_proba(input_df)[0][1]
    return prob_5star

# --- [v26] '메뉴판'을 보여주는 라우트 ---
@app.route('/', methods=['GET'])
def home():
    """
    손님이 '온라인 주소'('/')로 접속하면,
    'templates/index.html' 파일을 찾아서 보여줍니다.
    """
    return render_template('index.html')

# --- [v26] 'AI 셰프'가 주문을 받는 라우트 ---
@app.route('/predict', methods=['POST'])
def predict_and_recommend():
    global CURRENT_MODEL_VERSION
    
    try:
        # 1. 웹사이트로부터 JSON 입력 받기
        inputs = request.json
        
        # 2. '현재 확률' 계산
        current_inputs = {
            '나이': inputs.get('age'),
            '직업': inputs.get('job'),
            '성별': inputs.get('gender'),
            '결혼상태': inputs.get('marriage'),
            '가구월소득': inputs.get('income'),
            '여가목적1순위': inputs.get('purpose'),
            '참여여가활동1순위': inputs.get('activity'),
            '문화예술스포츠참여동반자': inputs.get('partner'),
            '전반적여가생활만족도_인프라': inputs.get('infra_sat'),
            '전반적여가생활만족도_시간': inputs.get('time_sat'),
            '거주지': "서울", '문화예술스포츠비용금액': 50000, '문화예술스포츠참여빈도': 1.5,
            'B0101020802': 0, '문화예술스포츠참여지역': np.nan
        }
        baseline_prob = get_prediction(current_inputs)
        baseline_prob_pct = baseline_prob * 100
        
        # 3. '페르소나' 정의
        if baseline_prob > 0.4: persona = "🏆 안정적 전문가"
        elif inputs.get('purpose') in ['3', '9'] or inputs.get('activity') == '6': persona = "🧭 이탈 위험군"
        else: persona = "🌱 성장형 탐험가"

        # 4. '여정 시뮬레이터' 실행
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
        
        # 5. 최종 추천 멘트 생성
        recommendations = [f"AI가 {len(simulation_results)}개의 모든 여가 조합을 시뮬레이션 했습니다."]
        recommendations.append(f"귀하의 고정 정보(나이, 직업, 성별 등)를 기준으로,\n5점 만족 확률이 가장 높은 **Top 3 궤적**은 다음과 같습니다.")
        
        for i in range(3):
            prob, p_name, a_name, t_name = simulation_results[i]
            recommendations.append(
                f"**🥇 {i+1}순위 (예상: {prob*100:.1f}%)**\n"
                f"   - **목적:** {p_name}\n"
                f"   - **활동:** {a_name}\n"
                f"   - **동반자:** {t_name}"
            )
        
        # 6. 웹사이트에 JSON으로 결과 응답
        return jsonify({
            "success": True,
            "model_version": MODEL_VERSION,
            "probability_5star_percent": round(baseline_prob_pct, 2),
            "persona": persona,
            "recommendations": "\n\n".join(recommendations)
        })
        
    except Exception as e:
        print(f"🚨 예측 중 오류 발생: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# 서버 실행 (Render.com이 이 부분을 자동으로 실행함)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host='0.0.0.0', port=port)