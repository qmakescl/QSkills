#!/usr/bin/env python3
"""테스트용 샘플 데이터 4종을 생성한다."""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# 현재 스크립트 위치 기준으로 evals/files 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
# ../evals/files
output_dir = os.path.abspath(os.path.join(script_dir, '..', 'evals', 'files'))
os.makedirs(output_dir, exist_ok=True)

print(f"Output directory: {output_dir}")

# ── 1. 독립표본 t-test용 데이터 ─────────────────────────────
# 시나리오: 새로운 교수법(실험군) vs 기존 교수법(대조군)의 시험 점수 비교
n = 30
control = np.random.normal(loc=72, scale=10, size=n)
experiment = np.random.normal(loc=78, scale=12, size=n)

df1 = pd.DataFrame({
    '집단': ['대조군'] * n + ['실험군'] * n,
    '시험점수': np.concatenate([control, experiment]).round(1)
})
path1 = os.path.join(output_dir, 'sample_independent.csv')
df1.to_csv(path1, index=False, encoding='utf-8-sig')
print(f"[1] 독립표본 데이터 생성: {path1} ({len(df1)}행)")

# ── 2. 대응표본 t-test용 데이터 (Wide Format - 권장) ─────────
# 시나리오: 동일 학생의 교육 프로그램 참여 전/후 자기효능감 점수
n2 = 25
pre = np.random.normal(loc=55, scale=8, size=n2)
post = pre + np.random.normal(loc=5, scale=4, size=n2)  # 평균 5점 향상

df_wide = pd.DataFrame({
    '학생ID': [f'S{i:03d}' for i in range(1, n2+1)],
    '사전_점수': pre.round(1),
    '사후_점수': post.round(1)
})
path_wide = os.path.join(output_dir, 'sample_paired_wide.csv')
df_wide.to_csv(path_wide, index=False, encoding='utf-8-sig')
print(f"[2] 대응표본(Wide) 데이터 생성: {path_wide} ({len(df_wide)}행)")

# ── 3. 대응표본 t-test용 데이터 (Long Format) ───────────────
# 위 데이터를 Long format으로 변환
df_long = pd.melt(df_wide, id_vars=['학생ID'], value_vars=['사전_점수', '사후_점수'], 
                  var_name='측정시점', value_name='점수')
# 측정시점 값을 '사전', '사후'로 변경 (컬럼명에서 '_점수' 제거)
df_long['측정시점'] = df_long['측정시점'].str.replace('_점수', '')

path_long = os.path.join(output_dir, 'sample_paired_long.csv')
df_long.to_csv(path_long, index=False, encoding='utf-8-sig')
print(f"[3] 대응표본(Long) 데이터 생성: {path_long} ({len(df_long)}행)")

# ── 4. One-way ANOVA용 데이터 ───────────────────────────────
# 시나리오: 3가지 운동 프로그램(A/B/C)에 따른 체중 감량(kg)
n3 = 20
group_a = np.random.normal(loc=3.0, scale=1.5, size=n3)   # 저강도
group_b = np.random.normal(loc=5.0, scale=1.8, size=n3)   # 중강도
group_c = np.random.normal(loc=6.5, scale=2.0, size=n3)   # 고강도

df3 = pd.DataFrame({
    '운동프로그램': ['저강도(A)'] * n3 + ['중강도(B)'] * n3 + ['고강도(C)'] * n3,
    '체중감량': np.concatenate([group_a, group_b, group_c]).round(2)
})
path3 = os.path.join(output_dir, 'sample_anova.csv')
df3.to_csv(path3, index=False, encoding='utf-8-sig')
print(f"[4] ANOVA 데이터 생성: {path3} ({len(df3)}행)")

print("\n모든 샘플 데이터 생성 완료!")
