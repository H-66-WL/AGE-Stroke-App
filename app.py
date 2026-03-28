import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone, timedelta

# ===== 页面配置 =====
st.set_page_config(
    page_title="AGE-Stroke Risk 评估系统 (NB)",
    page_icon="🧠",
    layout="wide"
)
# 初始化记录列表
if 'prediction_records' not in st.session_state:
    st.session_state.prediction_records = []

# ===== 加载 NB 模型 =====
@st.cache_resource
def load_model():
    model = joblib.load('nb_model.pkl')
    return model

model = load_model()

# ===== 标题与介绍 =====
st.title("🧠 AGE-Stroke Risk 辅助评估系统 (朴素贝叶斯)")
st.markdown("""
### 基于衰老相关基因的缺血性脑卒中风险预测
本研究筛选出 **FOS、PTGS2、LMNB1、CXCL1** 四个关键基因，构建朴素贝叶斯模型。
输入患者的四个基因表达量，即可获得卒中风险概率。
""")

# ===== 新增模块：项目背景（可折叠）=====
with st.expander("📘 项目背景与研究方法", expanded=False):
    st.markdown("""
    - **数据来源**：从GEO数据库获取GSE58294、GSE16561（训练集，合并后批量归一化），GSE22255（验证集）
    - **衰老相关基因**：从 HAGR、老化图谱及文献获取 1037 个 ARGs
    - **筛选流程**：差异表达分析 → WGCNA 模块筛选 → 与 ARGs 取交集 → 机器学习（12种算法对比）
    - **最终模型**：朴素贝叶斯（NB），基于 **FOS、PTGS2、LMNB1、CXCL1** 四个基因
    - **模型性能**：训练集 AUC=0.840，外部验证单基因 AUC 见下方
    """)

# ===== 侧边栏输入 =====
st.sidebar.header("📥 输入患者数据")
input_method = st.sidebar.radio(
    "选择输入方式",
    ["手动输入（滑块）", "手动输入（数字框）", "上传CSV文件"]
)

input_data = None
feature_cols = ['FOS', 'PTGS2', 'LMNB1', 'CXCL1']

if input_method == "手动输入（滑块）":
    st.sidebar.subheader("调整基因表达量")
    fos = st.sidebar.slider("FOS 表达量", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    ptgs2 = st.sidebar.slider("PTGS2 表达量", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    lmnb1 = st.sidebar.slider("LMNB1 表达量", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    cxcl1 = st.sidebar.slider("CXCL1 表达量", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
    input_data = pd.DataFrame([[fos, ptgs2, lmnb1, cxcl1]], columns=feature_cols)

elif input_method == "手动输入（数字框）":
    st.sidebar.subheader("输入基因表达量")
    fos = st.sidebar.number_input("FOS 表达量", value=0.0, step=0.1)
    ptgs2 = st.sidebar.number_input("PTGS2 表达量", value=0.0, step=0.1)
    lmnb1 = st.sidebar.number_input("LMNB1 表达量", value=0.0, step=0.1)
    cxcl1 = st.sidebar.number_input("CXCL1 表达量", value=0.0, step=0.1)
    input_data = pd.DataFrame([[fos, ptgs2, lmnb1, cxcl1]], columns=feature_cols)

else:
    st.sidebar.subheader("上传CSV文件")
    uploaded_file = st.sidebar.file_uploader(
        "选择CSV文件（需包含 FOS, PTGS2, LMNB1, CXCL1 四列）",
        type=['csv']
    )
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.sidebar.success(f"已上传文件：{uploaded_file.name}")
        st.sidebar.write("预览前5行：")
        st.sidebar.dataframe(input_data.head())
    else:
        st.sidebar.info("请上传CSV文件")

# ===== 新增模块：使用提示（侧边栏底部）=====
st.sidebar.markdown("---")
st.sidebar.markdown("""
**📌 使用提示**  
- 滑块/数字框可手动调整基因表达量  
- 点击「开始预测」查看结果  
- 支持上传 CSV 文件批量预测  
- 下方为模型验证图表及免疫浸润细胞分析
""")
# ===== 预测记录浏览 =====
with st.sidebar.expander("📜 预测记录", expanded=False):
    if st.session_state.prediction_records:
        records_df = pd.DataFrame(st.session_state.prediction_records)
        st.dataframe(records_df, use_container_width=True)
        csv = records_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ 下载记录 (CSV)",
            data=csv,
            file_name="prediction_records.csv",
            mime="text/csv"
        )
        if st.button("🗑️ 清空本次记录"):
            st.session_state.prediction_records = []
            st.rerun()
    else:
        st.info("暂无预测记录")

# ===== 模板下载 =====
st.sidebar.markdown("---")
st.sidebar.markdown("📎 **模板下载**")
example_csv = "FOS,PTGS2,LMNB1,CXCL1\n1.2,0.8,0.5,1.1\n-0.3,0.5,1.1,0.9"
st.sidebar.download_button(
    label="下载CSV模板",
    data=example_csv,
    file_name="template.csv",
    mime="text/csv"
)

# ===== 预测按钮和结果显示 =====
col1, col2, col3 = st.columns([1,2,1])
with col2:
    predict_button = st.button("🔍 开始预测", type="primary", use_container_width=True)

if input_data is not None and predict_button:
    if all(col in input_data.columns for col in feature_cols):
        X_input = input_data[feature_cols]
        predictions = model.predict(X_input)
        probabilities = model.predict_proba(X_input)

        st.markdown("---")
        st.header("📊 预测结果")

        # 定义北京时间时区（只定义一次）
        beijing_tz = timezone(timedelta(hours=8))

        for i in range(len(X_input)):
            st.subheader(f"样本 {i+1}")
            risk_prob = probabilities[i][1]

            if risk_prob >= 0.7:
                risk_level = "高风险"
                color = "red"
            elif risk_prob >= 0.4:
                risk_level = "中风险"
                color = "orange"
            else:
                risk_level = "低风险"
                color = "green"

            col_res1, col_res2, col_res3 = st.columns([2,1,2])
            with col_res1:
                st.markdown(f"### 风险等级：:{color}[{risk_level}]")
                st.progress(float(risk_prob))
            with col_res2:
                st.metric("患病概率", f"{risk_prob:.2%}")
                st.metric("预测类别", "IS患者" if predictions[i]==1 else "健康对照")
            with col_res3:
                st.write("**基因表达量：**")
                genes = X_input.iloc[i]
                for gene, value in genes.items():
                    st.write(f"{gene}: {value:.3f}")
            st.markdown("---")

            # 保存记录（使用北京时间，字段为风险概率）
            record = {
                '时间': datetime.now(beijing_tz).strftime("%Y-%m-%d %H:%M:%S"),
                'FOS': X_input.iloc[i]['FOS'],
                'PTGS2': X_input.iloc[i]['PTGS2'],
                'LMNB1': X_input.iloc[i]['LMNB1'],
                'CXCL1': X_input.iloc[i]['CXCL1'],
                '风险概率': risk_prob,
                '预测类别': 'IS患者' if predictions[i]==1 else '健康对照'
            }
            st.session_state.prediction_records.append(record)
    else:
        st.error(f"上传的文件必须包含以下列：{feature_cols}")

# ===== 模型验证与科学依据 =====
st.markdown("---")
st.header("📈 模型验证与科学依据")

col_img1, col_img2 = st.columns(2)
with col_img1:
    st.subheader("训练集ROC曲线")
    try:
        st.image("images/roc_curve.png", caption="AUC = 0.840 (训练集)",  use_container_width=True)
    except:
        st.info("请将ROC曲线图片放在 images/roc_curve.png")

with col_img2:
    st.subheader("校准曲线")
    try:
        st.image("images/calibration_curve.png", caption="模型校准效果",  use_container_width=True)
    except:
        st.info("请将校准曲线图片放在 images/calibration_curve.png")


# ===== 外部数据集验证（单基因AUC）=====
st.subheader("外部数据集验证（单基因AUC）")
col_val1, col_val2, col_val3, col_val4 = st.columns(4)
with col_val1:
    st.metric("FOS AUC", "0.670")   # 请替换为实际值
with col_val2:
    st.metric("PTGS2 AUC", "0.667")
with col_val3:
    st.metric("LMNB1 AUC", "0.503")
with col_val4:
    st.metric("CXCL1 AUC", "0.688")

# ===== 新增模块：单基因 ROC 曲线图 =====
st.subheader("📉 单基因 ROC 曲线（外部验证）")
try:
    st.image("images/roc_single_genes.png", caption="FOS, PTGS2, LMNB1, CXCL1 的 ROC 曲线", width=500)
except:
    st.info("请将单基因 ROC 曲线图放在 images/roc_single_genes.png (可选)")

# ===== 免疫浸润分析 =====
st.markdown("---")
st.header("🦠 免疫细胞浸润相关性分析")
st.markdown("基于 Spearman 相关性分析，展示四个枢纽基因与免疫细胞浸润水平的关系。")

# 读取 CSV 文件（请确保文件名正确）
try:
    corr_df = pd.read_csv("immune_corr.csv")
    pivot_table = corr_df.pivot(index='cell', columns='gene', values='r')
    st.subheader("相关系数矩阵")
    st.dataframe(pivot_table.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))
    st.subheader("相关性热图")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={'label': 'Spearman 相关系数'})
    ax.set_title('基因表达与免疫细胞浸润的相关性')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    st.markdown("""
**🔍 关键结论**：  
FOS、PTGS2、LMNB1、CXCL1 四个基因均与 **中性粒细胞 (Neutrophils)** 和 **M0 巨噬细胞 (Macrophages M0)** 呈显著正相关（p < 0.05），提示这些基因可能通过调节免疫细胞参与卒中后炎症反应。
""")
except:
    st.info("请将免疫相关性数据文件 immune_corr.csv 放在项目文件夹中")
# ===== 新增：基因在关键免疫细胞中的差异表达箱线图 =====
st.subheader("📦 基因在关键免疫细胞中的差异表达")
st.markdown("""
在 CD8T 细胞、初始 CD4T 细胞、M0 型巨噬细胞、中性粒细胞中，FOS、PTGS2、LMNB1、CXCL1 的表达量在 IS 患者组与健康对照组之间存在显著差异（p < 0.05）基因可能通过影响这些免疫细胞的功能，参与了卒中的发生发展。
""")
try:
    st.image("images/immune_boxplot.png", caption="四个基因在四种免疫细胞中的组间表达差异", use_container_width=True)
except:
    st.info("请将免疫细胞差异箱线图放在 images/immune_boxplot.png")

# ===== Nomogram诊断模型 =====
st.markdown("---")
st.header("📋 Nomogram诊断模型")
try:
    st.image("images/nomogram.png", caption="基于四个基因构建的Nomogram", use_container_width=True)
    st.markdown("""
    **Nomogram 使用说明**：  
    根据患者的 FOS、PTGS2、LMNB1、CXCL1 表达量，在对应基因轴上垂直向上投射得到分值，将四个基因的分值相加得到总分，再垂直向下投射到风险轴上，即可得到该患者发生缺血性脑卒中的预测概率。分值越高，风险越大。
    """)
except:
    st.info("请将Nomogram图片放在 images/nomogram.png")
# ===== 新增模块：团队信息与参考文献 =====
st.markdown("---")
st.markdown("""
**👥HQ 研究团队** | **指导老师**：袁浩 李金霞  
**📧 联系**：2977705201@qq.com| **项目时间**：2026年3月  
**📚 数据来源**  
1. [HAGR 老化基因组资源](https://genomics.senescence.info/)  
2. [老化图谱数据库](https://ngdc.cncb.ac.cn/ageing/)  
3. [GEO 数据库](https://www.ncbi.nlm.nih.gov/geo/) 
""") 
st.markdown("---")
st.caption("HQ 研究团队 | 基于GSE58294、GSE16561、GSE22255数据集开发，模型：朴素贝叶斯")
