#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import os
import sys

# 设置页面配置（必须是第一个Streamlit命令）
st.set_page_config(
    page_title="Prediction for early-happened SIC",
    layout="centered",
)

import matplotlib as mpl
import csv
import datetime
import sys
import os
mpl.use('agg')  # matplotlib相关导入
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import re
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import make_pipeline
import shap
from sklearn.base import is_classifier


# --------------------------
# 关键：获取脚本所在目录的绝对路径
# --------------------------
if getattr(sys, 'frozen', False):
    script_dir = sys._MEIPASS
else:
    script_dir = os.path.dirname(os.path.abspath(__file__))

# --------------------------
# 用脚本目录拼接CSV文件的相对路径
# --------------------------
# filled文件夹在脚本所在目录下，因此路径为：script_dir + "/filled/xxx.csv"
X_train_path = os.path.join(script_dir, "filled", "X_train_imputed.csv")
X_valid_path = os.path.join(script_dir, "filled", "X_val_imputed.csv")
X_test_path = os.path.join(script_dir, "filled", "X_test_imputed.csv")
X_test_mimic_path = os.path.join(script_dir, "filled", "X_test_mimic_imputed.csv")
X_zdyy_path = os.path.join(script_dir, "filled", "X_zdyy_imputed.csv")

X_train_scaled_path = os.path.join(script_dir, "filled", "X_train_scaler.csv")
X_valid_scaled_path = os.path.join(script_dir, "filled", "X_val_scaler.csv")
X_test_scaled_path = os.path.join(script_dir, "filled", "X_test_scaler.csv")
X_test_mimic_scaled_path = os.path.join(script_dir, "filled", "X_test_mimic_scaler.csv")
X_zdyy_scaled_path = os.path.join(script_dir, "filled", "X_zdyy_scaler.csv")

y_train_path = os.path.join(script_dir, "filled", "y_train.csv")
y_valid_path = os.path.join(script_dir, "filled", "y_val.csv")
y_test_path = os.path.join(script_dir, "filled", "y_test.csv")
y_test_mimic_path = os.path.join(script_dir, "filled", "y_test_mimic.csv")
y_zdyy_path = os.path.join(script_dir, "filled", "y_test_zdyy.csv")

# 读取CSV文件（使用拼接后的绝对路径）
X_train = pd.read_csv(X_train_path)
X_valid = pd.read_csv(X_valid_path)
X_test = pd.read_csv(X_test_path)
X_test_mimic = pd.read_csv(X_test_mimic_path)
X_zdyy = pd.read_csv(X_zdyy_path)

X_train_scaled = pd.read_csv(X_train_scaled_path)
X_valid_scaled = pd.read_csv(X_valid_scaled_path)
X_test_scaled = pd.read_csv(X_test_scaled_path)
X_test_mimic_scaled = pd.read_csv(X_test_mimic_scaled_path)
X_zdyy_scaled = pd.read_csv(X_zdyy_scaled_path)

y_train = pd.read_csv(y_train_path)
y_valid = pd.read_csv(y_valid_path)
y_test = pd.read_csv(y_test_path)
y_test_mimic = pd.read_csv(y_test_mimic_path)
y_zdyy = pd.read_csv(y_zdyy_path)

# 处理标签列
y_train = y_train['SIC_early_happen']
y_valid = y_valid['SIC_early_happen']
y_test = y_test['0']
y_test_mimic = y_test_mimic['SIC_early_happen']
y_zdyy = y_zdyy['SIC_D3']

# 清理列名（避免特殊字符）
def clean_columns(df):
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    return df

X_train = clean_columns(X_train)
X_valid = clean_columns(X_valid)
X_test = clean_columns(X_test)
X_test_mimic = clean_columns(X_test_mimic)
X_zdyy = clean_columns(X_zdyy)

X_train_scaled = clean_columns(X_train_scaled)
X_valid_scaled = clean_columns(X_valid_scaled)
X_test_scaled = clean_columns(X_test_scaled)
X_test_mimic_scaled = clean_columns(X_test_mimic_scaled)
X_zdyy_scaled = clean_columns(X_zdyy_scaled)

# 读取特征重要性文件（同样使用绝对路径）
features_csv_path = os.path.join(script_dir, "filled", "Features_Final.csv")
try:
    importance_df = pd.read_csv(features_csv_path, encoding='utf-8')
    top_features = importance_df.head(10)['Feature_Names'].values
except FileNotFoundError:
    st.error(f"文件未找到：{features_csv_path}")
except Exception as e:
    st.error(f"读取文件失败：{str(e)}")

# 处理特征名（去除特殊字符）
top_features = [re.sub(r'[^\x00-\x7F]+', '', f) for f in top_features]
X_train = X_train[top_features]
X_valid = X_valid[top_features]
X_test_mimic = X_test_mimic[top_features]
X_zdyy = X_zdyy[top_features]

X_train_scaled = X_train_scaled[top_features]
X_valid_scaled = X_valid_scaled[top_features]
X_test_mimic_scaled = X_test_mimic_scaled[top_features]
X_zdyy_scaled = X_zdyy_scaled[top_features]

# 重置标签索引
y_train = y_train.reset_index(drop=True)
y_valid = y_valid.reset_index(drop=True)
y_test_mimic = y_test_mimic.reset_index(drop=True)
y_zdyy = y_zdyy.reset_index(drop=True)

# 训练模型（使用过采样和校准）
neg_count = len(y_train[y_train == 0])
pos_count = len(y_train[y_train == 1])
scale_pos_weight = neg_count / pos_count

xgb_model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=50,
    learning_rate=0.0585,
    max_depth=3,
    gamma=1.0,
    subsample=0.6,
    scale_pos_weight=scale_pos_weight,
    colsample_bytree=0.6,
    random_state=42
)

pipeline_xgb = make_pipeline(
    SMOTE(random_state=42),
    CalibratedClassifierCV(
        xgb_model,
        method='sigmoid',
        cv=StratifiedKFold(n_splits=5)
    )
)
pipeline_xgb.fit(X_train, y_train)

# 保存模型（路径基于脚本目录）
model_path = os.path.join(script_dir, "full_pipeline_sic.pkl")
joblib.dump(pipeline_xgb, model_path)

# 加载模型（使用绝对路径）
@st.cache_resource
def load_pipeline():
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except FileNotFoundError:
        st.error(f"模型文件未找到：{model_path}")
    except Exception as e:
        st.error(f"加载模型失败：{str(e)}")

pipeline_xgb = load_pipeline()
X_train_resampled, _ = pipeline_xgb.named_steps['smote'].fit_resample(X_train, y_train)

# 提取模型（用于预测和SHAP解释）
def extract_model(pipeline):
    calibrated = pipeline.named_steps['calibratedclassifiercv']
    return calibrated.calibrated_classifiers_[0].estimator

model = extract_model(pipeline_xgb)

# 初始化SHAP解释器
@st.cache_resource
def init_shap_explainer(_model, X_background=None):
    return shap.TreeExplainer(
        model=_model,
        data=X_background,
        model_output='probability',
    )



# 打印特征名（调试用）
print(top_features)

with st.sidebar:  # 使用侧边栏
    st.header("👤Patient basic information")

    with st.form("patient_info_form"):
        name = st.text_input("Name*", max_chars=50, help="Real Name")

        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender*", ["Male", "Female"], index=None)
        with col2:
            age = st.number_input("Age*", min_value=0, max_value=150, step=1)

        hospital_id = st.text_input("Addmission_ID*", max_chars=30)
        phone = st.text_input("Telephone Number")

        submit_basic = st.form_submit_button("OK")

    if submit_basic:
        errors = []
        if not name.strip():
            errors.append("姓名不能为空")
        if not gender:
            errors.append("请选择性别")
        if age <= 0 or age > 130:
            errors.append("请输入合理的年龄")
        if not hospital_id.strip():
            errors.append("住院号不能为空")
        if phone and not re.match(r"^1[3-9]\d{9}$", phone):
            errors.append("手机号格式不正确")

        if errors:
            for error in errors:
                st.error("! " + error)
        else:
            #生成唯一患者ID
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            patient_id = f"SIC_{timestamp}_{np.random.randint(100, 999)}"
             
            st.session_state.patient_info = {
            "patient_id": patient_id,
            "name": name,
            "gender": gender,
            "age": age,
            "hospital_id": hospital_id,
            "phone": phone,
            "timestamp": datetime.datetime.now().isoformat()
            }
             
             
             # 保存到CSV文件
            try:
            # 定义CSV文件路径
                patient_csv_path = os.path.join(script_dir, "patient_records.csv")
            
            # 检查文件是否存在，如果不存在则创建并写入表头
                file_exists = os.path.isfile(patient_csv_path)
            
                with open(patient_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['patient_id', 'name', 'gender', 'age', 'hospital_id', 'phone', 'timestamp']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 如果文件不存在，写入表头
                    if not file_exists:
                        writer.writeheader()
                
                # 写入患者数据
                    writer.writerow(st.session_state.patient_info)
            
                st.success(f"患者信息已保存！分配ID：`{patient_id}`")
                st.balloons()
            
            except Exception as e:
                st.error(f"保存患者信息到文件失败：{str(e)}")


# 创建预测表单
with st.form("prediction_form"):
    col_sofa, col_vent, col_lab1, col_lab2 = st.columns([2, 1.5, 3, 3])

    with col_sofa:
        st.markdown("SOFA Scores")
        sofa_circ = st.slider(
            "Circulation", 0, 4, 0, 1,
            help="Scoring based on MAP and vasoactive drug dose."
        )
        sofa_renal = st.slider(
            "Renal", 0, 4, 0, 1,
            help="Scoring based on creatinine and urine output."
        )
        sofa_resp = st.slider(
            "Respiratory", 0, 4, 0, 1,
            help="Scoring based on PaO2/FiO2 and ventilation support."
        )

    with col_vent:
        st.markdown("Respiratory Support")
        mech_vent_1 = st.radio(
            "Mechanical Ventilation Status on Day 1",
            options=("yes", "no"),
            index=1,
            help="Whether invasive mechanical ventilation was used on the first day of ICU admission.",
            horizontal=True
        )
    mech_vent_encoded = 1 if mech_vent_1 == "yes" else 0

    with col_lab1:
        st.markdown("Coagulation Laboratory Tests")
        platelet_count = st.number_input(
            "Platelet count (×10⁹/L)",
            min_value=20, max_value=600,
            value=150, step=10
        )
        inr = st.number_input(
            "INR",
            min_value=0.5, max_value=5.0,
            value=1.2, step=0.1, format="%.1f"
        )

    with col_lab2:
        st.markdown("Blood Gas Analysis")
        lactate = st.number_input(
            "lactate (mmol/L)",
            min_value=0.5, max_value=15.0,
            value=2.0, step=0.5, format="%.1f"
        )

    submitted = st.form_submit_button("Analyze")

if submitted:
    # 检查是否有患者基本信息
    if 'patient_info' not in st.session_state:
        st.warning("请先填写患者基本信息")
        st.stop()
    
    patient_id = st.session_state.patient_info['patient_id']
    
    try:
        # 在这里初始化SHAP解释器（确保在需要时才创建）
        @st.cache_resource
        def get_shap_explainer(_model, background_data):
            return shap.TreeExplainer(
                model=_model,
                data=background_data,
                model_output='probability'
            )
        
        explainer = get_shap_explainer(model, X_train_resampled)
        
        input_df = pd.DataFrame(
            [[sofa_circ, mech_vent_encoded, sofa_renal, platelet_count, inr, sofa_resp, lactate]],
            columns=["sofa_circ", "mech_vent_1", "sofa_renal", "platelet_count", "inr", "sofa_resp", "lactate"]
        )

        proba = model.predict_proba(input_df)[0][1]
        st.success(f"The probability of early-SIC happening: {proba:.1%}")

        # 计算SHAP值
        shap_explanation = explainer(input_df)
        shap_values = shap_explanation.values[0]

        # 处理基准值（二分类取正类）
        if isinstance(explainer.expected_value, (list, np.ndarray)) and len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[1]
        else:
            base_value = explainer.expected_value

        # 格式化特征值（用于可视化）
        formatted_features = input_df.copy().astype(object)
        for col in formatted_features.columns:
            if formatted_features[col].dtype in [np.float64, np.int64]:
                formatted_features[col] = formatted_features[col].apply(lambda x: f"{x:.1f}")

        # 绘制SHAP力导向图
        plt.figure(figsize=(18, 5), dpi=300, facecolor='white')
        force_plot = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values,
            feature_names=list(input_df.columns),
            features=formatted_features.iloc[0].values,
            matplotlib=True,
            show=False,
            contribution_threshold=0.03,
            figsize=(18, 5)
        )

        # 美化图像
        plt.title("Clinical Feature Impacts", fontsize=14, pad=20, fontweight='bold')
        plt.xlabel("SHAP Value Contribution", fontsize=12, labelpad=15)
        plt.xticks(fontsize=10, rotation=0)
        plt.gca().xaxis.set_ticks_position('bottom')
        plt.margins(x=0.15)

        # 保存图像（路径基于脚本目录，避免权限问题）
        shap_image_path = os.path.join(script_dir, "shap_force.png")
        plt.savefig(
            shap_image_path,
            dpi=900,
            bbox_inches='tight',
            pad_inches=0.2,
            facecolor='white'
        )

        # 显示图像
        st.image(shap_image_path, use_container_width=True)

        #患者预测信息保存
        prediction_data = {
            'patient_id': patient_id,
            'prediction_time': datetime.datetime.now().isoformat(),
            'sofa_circ': sofa_circ,
            'mech_vent_encoded': mech_vent_encoded,
            'sofa_renal': sofa_renal,
            'platelet_count': platelet_count,
            'inr': inr,
            'sofa_resp': sofa_resp,
            'lactate': lactate,
            'sic_probability': proba
        }
        
        # 保存预测结果到另一个CSV文件
        prediction_csv_path = os.path.join(script_dir, "prediction_results.csv")
        pred_file_exists = os.path.isfile(prediction_csv_path)
        
        with open(prediction_csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(prediction_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not pred_file_exists:
                writer.writeheader()
            
            writer.writerow(prediction_data)

    except Exception as e:
        st.error(f"可视化错误: {str(e)}")