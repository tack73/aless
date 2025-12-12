import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import statsmodels.api as sm
import re
import os

def analyze_and_visualize(file_path):
    """
    アンケートデータを読み込み、統計分析（相関・回帰）を行い、結果をグラフ化する関数
    """
    print(f"🚀 分析を開始します: {file_path}")

    # 1. データ読み込み
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    # 2. 前処理：カラム名の整理とスコア計算
    rename_dict = {
        '性別を選択してください。': 'Gender',
        '幼少期に最も長く住んでいた居住地域の種類を選択してください。': 'Residence_Area',
        '幼少期、屋外で遊ぶ際、よく山や川、海、田んぼなど、自然に近接した空間で遊んでいましたか？': 'Nature_Contact',
        '幼少期によく本を読んでいましたか？': 'Reading_Habit',
        '幼少期によく虫に関する本（図鑑等も含む）を読んでいましたか？': 'Insect_Book_Reading',
    }
    # Q1-Q11の自動抽出
    for col in df.columns:
        match = re.match(r'(\d+)\.', col)
        if match:
            rename_dict[col] = f'Q{match.group(1)}'

    df_clean = df.rename(columns=rename_dict)

    # 虫嫌いスコアの算出 (Q1-Q11の合計)
    q_cols = [f'Q{i}' for i in range(1, 12)]
    for col in q_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean['Insect_Dislike_Score'] = df_clean[q_cols].sum(axis=1)

    # 3. 数値化（分析用）
    # 順序尺度を数値に変換
    mapping_order = {'よく遊んでいた': 3, 'たまに遊んでいた': 2, 'ほとんど遊ばなかった': 1,
                     'よく読んでいた': 3, 'たまに読んでいた': 2, 'ほとんど読まなかった': 1}
    
    df_clean['Nature_Contact_Num'] = df_clean['Nature_Contact'].map(mapping_order)
    df_clean['Reading_Habit_Num'] = df_clean['Reading_Habit'].map(mapping_order)
    df_clean['Insect_Book_Reading_Num'] = df_clean['Insect_Book_Reading'].map(mapping_order)
    # 性別 (男性=0, 女性=1)
    df_clean['Gender_Num'] = df_clean['Gender'].map({'男性': 0, '女性': 1})
    # 居住地域 (都市化度: 農村=1 → 都心=4)
    mapping_area = {'農村・漁村': 1, '地方中心市街地': 2, '郊外住宅地・団地': 3, '都心・都市部': 4}
    df_clean['Residence_Area_Num'] = df_clean['Residence_Area'].map(mapping_area)

    # --- 分析パート ---

    # (A) 相関分析（スピアマンの順位相関）
    from scipy.stats import spearmanr
    print("\n📊 --- 各環境要因と虫嫌いスコアの相関分析 ---")
    factors = {
        '自然接触頻度': 'Nature_Contact_Num',
        '読書習慣': 'Reading_Habit_Num',
        '虫本読書頻度': 'Insect_Book_Reading_Num',
        '性別(女性=1)': 'Gender_Num',
        '居住地域(都市化度)': 'Residence_Area_Num'
    }
    
    # 相関分析結果をファイルに出力
    with open('correlation_results.txt', 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("各環境要因と虫嫌いスコアの相関分析結果\n")
        f.write("=" * 60 + "\n\n")
        f.write("分析方法: スピアマンの順位相関係数\n")
        f.write(f"サンプルサイズ: N={len(df_clean.dropna(subset=['Insect_Dislike_Score']))}\n\n")
        f.write("-" * 60 + "\n")
        
        for label, col in factors.items():
            data_corr = df_clean[['Insect_Dislike_Score', col]].dropna()
            if len(data_corr) > 0:
                corr, p_value = spearmanr(data_corr['Insect_Dislike_Score'], data_corr[col])
                sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'
                
                result_line = f"{label:20s}: r={corr:6.3f}, p={p_value:.4f} {sig}\n"
                print(result_line.strip())
                f.write(result_line)
        
        f.write("-" * 60 + "\n\n")
        f.write("【有意水準】\n")
        f.write("  *** : p < 0.001 (非常に有意)\n")
        f.write("  **  : p < 0.01  (有意)\n")
        f.write("  *   : p < 0.05  (やや有意)\n")
        f.write("  n.s.: p >= 0.05 (有意でない)\n\n")
        f.write("【解釈】\n")
        f.write("  負の相関(r < 0): その要因が強いほど虫嫌いが減る傾向\n")
        f.write("  正の相関(r > 0): その要因が強いほど虫嫌いが増える傾向\n")
        f.write("  |r| > 0.5: 強い相関\n")
        f.write("  0.3 < |r| <= 0.5: 中程度の相関\n")
        f.write("  0.1 < |r| <= 0.3: 弱い相関\n")
        f.write("  |r| <= 0.1: ほぼ相関なし\n")
    
    print("\n有意水準: *** p<0.001, ** p<0.01, * p<0.05, n.s. 有意でない")
    print("負の相関 → その要因が強いほど虫嫌いが減る")
    print("正の相関 → その要因が強いほど虫嫌いが増える")
    print("✅ 相関分析結果を保存: correlation_results.txt")

    # (B) 重回帰分析（標準化偏回帰係数）
    # 目的変数: スコア, 説明変数: 各要因
    y = df_clean['Insect_Dislike_Score']
    X = df_clean[['Nature_Contact_Num', 'Reading_Habit_Num', 'Insect_Book_Reading_Num', 'Gender_Num', 'Residence_Area_Num']]
    
    # 欠損除去
    data_reg = pd.concat([y, X], axis=1).dropna()
    y = data_reg['Insect_Dislike_Score']
    X = data_reg[['Nature_Contact_Num', 'Reading_Habit_Num', 'Insect_Book_Reading_Num', 'Gender_Num', 'Residence_Area_Num']]
    
    # 標準化（影響度の大きさを比較するため）
    y_std = (y - y.mean()) / y.std()
    X_std = (X - X.mean()) / X.std()
    X_std = sm.add_constant(X_std) # 定数項追加

    model = sm.OLS(y_std, X_std).fit()
    print("\n📊 --- 重回帰分析結果（標準化係数） ---")
    print(model.params.drop('const')) # 定数項以外を表示
    print("-> 値がマイナスであるほど、その要因が強いと「虫嫌いが減る」ことを意味します。")


    # --- 可視化パート ---
    
    # フォント設定（Times New Romanに固定）
    plt.rcParams['font.family'] = 'Times New Roman'
    print(f"ℹ️ フォント設定: Times New Roman")
    
    sns.set(style="whitegrid", font='Times New Roman')

    # 図1-1: 自然接触頻度とスコアの箱ひげ図
    df_plot = df_clean.copy()
    df_plot['Nature_Contact'] = df_plot['Nature_Contact'].map({
        'よく遊んでいた': 'Frequently',
        'たまに遊んでいた': 'Occasionaly',
        'ほとんど遊ばなかった': 'Rarely'
    })
    order_nature = ['Frequently', 'Occasionaly', 'Rarely']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Nature_Contact', y='Insect_Dislike_Score', data=df_plot, order=order_nature, palette='viridis')
    # plt.title('Nature Contact Frequency vs Insect Dislike Score', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Insect Dislike Score', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Outdoor Play Frequency in Childhood', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('1-1_boxplot_nature_vs_score.png')
    print("✅ 図1-1 保存完了: 1-1_boxplot_nature_vs_score.png")

    # 図1-2: 読書習慣とスコアの箱ひげ図
    df_plot = df_clean.copy()
    df_plot['Reading_Habit'] = df_plot['Reading_Habit'].map({
        'よく読んでいた': 'Frequently',
        'たまに読んでいた': 'Occasionaly',
        'ほとんど読まなかった': 'Rarely'
    })
    order_reading = ['Frequently', 'Occasionaly', 'Rarely']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Reading_Habit', y='Insect_Dislike_Score', data=df_plot, order=order_reading, palette='viridis')
    # plt.title('Reading Habit in Childhood vs Insect Dislike Score', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Insect Dislike Score', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Reading Frequency', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('1-2_boxplot_reading_vs_score.png')
    print("✅ 図1-2 保存完了: 1-2_boxplot_reading_vs_score.png")

    # 図1-3: 虫本読書頻度とスコアの箱ひげ図
    df_plot = df_clean.copy()
    df_plot['Insect_Book_Reading'] = df_plot['Insect_Book_Reading'].map({
        'よく読んでいた': 'Frequently',
        'たまに読んでいた': 'Occasionaly',
        'ほとんど読まなかった': 'Rarely'
    })
    order_insect_book = ['Frequently', 'Occasionaly', 'Rarely']
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Insect_Book_Reading', y='Insect_Dislike_Score', data=df_plot, order=order_insect_book, palette='viridis')
    # plt.title('Insect Book Reading Frequency vs Insect Dislike Score', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Insect Dislike Score', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Insect-Related Book Reading Frequency', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('1-3_boxplot_insect_book_vs_score.png')
    print("✅ 図1-3 保存完了: 1-3_boxplot_insect_book_vs_score.png")

    # 図1-4: 性別とスコアの箱ひげ図
    df_plot = df_clean.copy()
    df_plot['Gender'] = df_plot['Gender'].map({
        '男性': 'Male',
        '女性': 'Female'
    })
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Gender', y='Insect_Dislike_Score', data=df_plot, order=['Male', 'Female'], palette='viridis')
    # plt.title('Gender vs Insect Dislike Score', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Insect Dislike Score', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Gender', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('1-4_boxplot_gender_vs_score.png')
    print("✅ 図1-4 保存完了: 1-4_boxplot_gender_vs_score.png")

    # 図1-5: 居住地域とスコアの箱ひげ図
    df_plot = df_clean.copy()
    df_plot['Residence_Area'] = df_plot['Residence_Area'].map({
        '農村・漁村': 'Rural',
        '地方中心市街地': 'Regional City',
        '郊外住宅地・団地': 'Suburban',
        '都心・都市部': 'Urban'
    })
    order_area = ['Rural', 'Regional City', 'Suburban', 'Urban']
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Residence_Area', y='Insect_Dislike_Score', data=df_plot, order=order_area, palette='viridis')
    # plt.title('Childhood Residence Area vs Insect Dislike Score', fontsize=14, fontname='Times New Roman')
    plt.ylabel('Insect Dislike Score', fontsize=12, fontname='Times New Roman')
    plt.xlabel('Residence Area Type', fontsize=12, fontname='Times New Roman')
    plt.xticks(rotation=15, fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('1-5_boxplot_residence_vs_score.png')
    print("✅ 図1-5 保存完了: 1-5_boxplot_residence_vs_score.png")

    # 図2: 相関行列のヒートマップ
    plt.figure(figsize=(11, 9))
    corr_cols = ['Insect_Dislike_Score', 'Nature_Contact_Num', 'Reading_Habit_Num', 'Insect_Book_Reading_Num', 'Gender_Num', 'Residence_Area_Num']
    corr_labels = ['Insect Dislike', 'Nature Contact', 'Reading Habit', 'Insect Book', 'Gender (F=1)', 'Urban Residence']
    corr_mat = df_clean[corr_cols].corr(method='spearman')
    corr_mat.index = corr_labels
    corr_mat.columns = corr_labels
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', square=True, 
                cbar_kws={'label': 'Spearman Correlation'}, annot_kws={'fontname': 'Times New Roman'})
    # plt.title('Correlation Matrix (Spearman Rank Correlation)', fontsize=14, fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('2_heatmap_correlation.png')
    print("✅ 図2 保存完了: 2_heatmap_correlation.png")

    # 図3: 回帰係数の棒グラフ（影響度の可視化）
    plt.figure(figsize=(10, 6))
    coefs = model.params.drop('const')
    coef_labels = ['Nature Contact', 'Reading Habit', 'Insect Book', 'Gender (F=1)', 'Urban Residence']
    coefs.index = coef_labels
    colors = ['blue' if c < 0 else 'red' for c in coefs]
    coefs.plot(kind='barh', color=colors)
    plt.axvline(0, color='black', linewidth=0.8)
    # plt.title('Impact of Factors on Insect Dislike (Standardized Regression Coefficients)', fontsize=14, fontname='Times New Roman')
    plt.xlabel('Coefficient (negative = reduces insect dislike)', fontsize=12, fontname='Times New Roman')
    plt.xticks(fontname='Times New Roman')
    plt.yticks(fontname='Times New Roman')
    plt.tight_layout()
    plt.savefig('3_regression_coefficients.png')
    print("✅ 図3 保存完了: 3_regression_coefficients.png")

    print("\n✨ 全ての処理が完了しました。")

# --- 実行 ---
if __name__ == "__main__":
    # ファイル名を指定して実行
    target_file = "data.csv"
    if os.path.exists(target_file):
        analyze_and_visualize(target_file)
    else:
        print(f"ファイルが見つかりません: {target_file}")