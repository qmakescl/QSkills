#!/usr/bin/env python3
"""평균비교 검정 메인 분석 스크립트
- Independent two-sample t-test / Welch's t-test
- Paired t-test (Wide format 권장, Long format 지원)
- One-way ANOVA / Welch's ANOVA + Post-hoc (Tukey, Scheffé, Dunnett's T3 등)
- 정규성 검정 (Shapiro-Wilk) 및 비모수 검정 권장 안내
- 보고서 자동 생성 (Markdown, HTML, Charts)
"""
import argparse
import json
import os
import sys
import warnings
import datetime
import shutil

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['axes.unicode_minus'] = False


def setup_korean_font():
    """한글 폰트 설정 (macOS / Linux / Windows 대응)"""
    # 우선순위 순서대로 한글 폰트 탐색
    priority_keywords = [
        'apple sd gothic neo',   # macOS 기본
        'applesd',               # macOS 축약
        'applesdgothicneo',
        'appleGothic',           # macOS 기본 (구버전)
        'malgun gothic',         # Windows
        'nanum gothic',          # 범용 (macOS/Linux)
        'noto sans cjk kr',      # Linux
        'noto sans kr',          # Linux
        'nanumgothic',
    ]
    
    for kw in priority_keywords:
        for f in fm.fontManager.ttflist:
            if kw.lower() in f.name.lower():
                plt.rcParams['font.family'] = f.name
                return f.name
    
    # 폴백: 'gothic', 'nanum' 등 광범위 검색
    for f in fm.fontManager.ttflist:
        if any(k in f.name.lower() for k in ['gothic', 'nanum', 'noto sans cjk', 'malgun', 'batang', 'gulim', 'dotum']):
            plt.rcParams['font.family'] = f.name
            return f.name
    
    warnings.warn("한글 폰트를 찾을 수 없습니다. 차트 내 한글이 깨질 수 있습니다.")
    return None


setup_korean_font()


def load_data(fp):
    """데이터 로드 (CSV, Excel 지원)"""
    ext = os.path.splitext(fp)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(fp)
    
    # CSV 인코딩 자동 감지 시도
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    
    # 실패 시 utf-8로 읽고 오류 무시
    return pd.read_csv(fp, encoding='utf-8', errors='replace')


def cohen_d(g1, g2):
    """Independent t-test Cohen's d"""
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    
    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_sd = np.sqrt(pooled_var)
    
    return (g1.mean() - g2.mean()) / pooled_sd if pooled_sd > 0 else 0.0


def cohen_d_paired(diff):
    """Paired t-test Cohen's d"""
    sd = diff.std(ddof=1)
    return diff.mean() / sd if sd > 0 else 0.0


def eta_squared(groups):
    """ANOVA Eta-squared"""
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    
    # Sum of Squares Between
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    
    # Sum of Squares Total
    sst = np.sum((all_data - grand_mean) ** 2)
    
    return ssb / sst if sst > 0 else 0.0


def interp_d(d):
    """Cohen's d 해석"""
    a = abs(d)
    if a < 0.2: return "효과 없음(negligible)"
    elif a < 0.5: return "작은 효과(small)"
    elif a < 0.8: return "중간 효과(medium)"
    else: return "큰 효과(large)"


def interp_eta(e):
    """Eta-squared 해석"""
    if e < 0.01: return "효과 없음(negligible)"
    elif e < 0.06: return "작은 효과(small)"
    elif e < 0.14: return "중간 효과(medium)"
    else: return "큰 효과(large)"


def interp_p(p, alpha=0.05):
    """p-value 별표 표기"""
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < alpha: return "*"
    else: return "n.s."


# ── 소수점 자릿수 포맷팅 규칙 ──────────────────────────────
# 평균, 평균차이: 2자리 / SD, SE, CI: 3자리
# t, F, Cohen's d, η² 통계량: 3자리
# p-value: 3자리 (< .001 이면 "< .001")
DEC_MEAN = 2   # 평균
DEC_STAT = 3   # 통계량 (t, F, d, η², W, etc.)
DEC_SE   = 3   # SD, SE, CI
DEC_P    = 3   # p-value 표시 자릿수
DEC_P_FULL = 6 # p-value 내부 저장 정밀도

def r_mean(v):
    """평균/평균차이 반올림"""
    return round(float(v), DEC_MEAN)

def r_stat(v):
    """통계량(t, F, d, η², W 등) 반올림"""
    return round(float(v), DEC_STAT)

def r_se(v):
    """SD, SE, CI 반올림"""
    return round(float(v), DEC_SE)

def r_p(v):
    """p-value 내부 저장용(정밀)"""
    return round(float(v), DEC_P_FULL)

def fmt_p(p):
    """p-value 보고서 표시용. < .001 이면 '< .001', 아니면 3자리."""
    if p < 0.001:
        return "< .001"
    return f"{p:.{DEC_P}f}"

def fmt_ci_val(v):
    """CI 값 보고서 표시용 (inf 처리 포함)"""
    if isinstance(v, str):  # 이미 문자열("∞", "-∞")
        return v
    if v == float('inf'): return "∞"
    if v == float('-inf'): return "-∞"
    return f"{v:.{DEC_SE}f}"


def normality_test(groups, group_names, alpha=0.05):
    """Shapiro-Wilk 정규성 검정"""
    results = []
    recommend_nonparametric = False
    
    for g, name in zip(groups, group_names):
        n = len(g)
        if n < 3: # Shapiro-Wilk requires N >= 3
             results.append({
                'group': str(name),
                'n': int(n),
                'w_statistic': None,
                'p_value': None,
                'is_normal': 'N < 3 (검정 불가)',
                'message': '표본 수가 너무 적음'
            })
             recommend_nonparametric = True
             continue

        w, p = stats.shapiro(g)
        is_normal = p >= alpha
        if not is_normal or n <= 20:
            recommend_nonparametric = True
            
        results.append({
            'group': str(name),
            'n': int(n),
            'w_statistic': r_stat(w),
            'p_value': r_p(p),
            'is_normal': "정규성 충족" if is_normal else "정규성 기각"
        })
        
    return results, recommend_nonparametric


def group_desc(data, dv, iv, alternative='two-sided'):
    """기술통계 산출 (95% CI 포함 - 검정 방향 고려)"""
    desc = []
    
    # alternative에 따른 CI 계산을 위한 alpha 설정
    # t.ppf(q, df): q is the cumulative probability
    
    for name, grp in data.groupby(iv)[dv]:
        n = len(grp)
        m = grp.mean()
        s = grp.std(ddof=1)
        se = s / np.sqrt(n)
        
        # 기본적으로 양쪽검정 95% CI 계산 (보고서용)
        # alpha = 0.05, two-tailed -> 0.975
        t_crit_two = stats.t.ppf(0.975, n - 1)
        ci_lower_two = m - t_crit_two * se
        ci_upper_two = m + t_crit_two * se
        
        # 검정 방향에 따른 CI (검정 결과용)
        if alternative == 'two-sided':
            ci_lower_test = ci_lower_two
            ci_upper_test = ci_upper_two
        elif alternative == 'greater':
            # one-tailed (lower bound), alpha = 0.05 -> 0.95
            t_crit_one = stats.t.ppf(0.95, n - 1)
            ci_lower_test = m - t_crit_one * se
            ci_upper_test = float('inf') # 상한은 무한대
        else: # less
            # one-tailed (upper bound), alpha = 0.05 -> 0.95
            t_crit_one = stats.t.ppf(0.95, n - 1)
            ci_lower_test = float('-inf') # 하한은 -무한대
            ci_upper_test = m + t_crit_one * se

        desc.append({
            'group': str(name),
            'n': int(n),
            'mean': r_mean(m),
            'std': r_se(s),
            'se': r_se(se),
            'mean_ci_lower': r_se(ci_lower_two),
            'mean_ci_upper': r_se(ci_upper_two),
            'test_ci_lower': ci_lower_test if ci_lower_test not in [float('inf'), float('-inf')] else ( "∞" if ci_lower_test == float('inf') else "-∞"),
            'test_ci_upper': ci_upper_test if ci_upper_test not in [float('inf'), float('-inf')] else ( "∞" if ci_upper_test == float('inf') else "-∞")
        })
    return desc


def levene_test(groups, alpha=0.05):
    """Levene's 등분산 검정"""
    stat, p = stats.levene(*groups)
    df1 = len(groups) - 1
    df2 = sum(len(g) for g in groups) - len(groups)
    
    return {
        'test': "Levene's test",
        'statistic': r_stat(stat),
        'df1': int(df1),
        'df2': int(df2),
        'p_value': r_p(p),
        'equal_variance': bool(p >= alpha),
        'interpretation': "등분산 가정 충족" if p >= alpha else "등분산 가정 기각(이분산)"
    }


def run_ind_ttest(g1, g2, equal_var=True, alternative='two-sided', g1_name='Group1', g2_name='Group2'):
    """독립표본 t-test (Student's / Welch's)"""
    stat, p = stats.ttest_ind(g1, g2, equal_var=equal_var, alternative=alternative)
    
    if equal_var:
        df = len(g1) + len(g2) - 2
        tn = "Independent two-sample t-test (Student's)"
    else:
        # Welch-Satterthwaite equation
        n1, n2 = len(g1), len(g2)
        v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else n1 + n2 - 2
        tn = "Welch's t-test"
        
    d = cohen_d(g1, g2)
    alt_map = {
        'two-sided': f'양쪽검정(two-tailed, {g1_name} - {g2_name} ≠ 0)',
        'greater': f'한쪽검정(one-tailed, {g1_name} - {g2_name} > 0)',
        'less': f'한쪽검정(one-tailed, {g1_name} - {g2_name} < 0)'
    }
    
    # 평균 차이 및 CI 계산
    mean_diff = g1.mean() - g2.mean()
    if equal_var:
        # Pooled SE
        n1, n2 = len(g1), len(g2)
        v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2) * (1/n1 + 1/n2))
        se_diff = pooled_se
    else:
        # Welch SE
        se_diff = np.sqrt(g1.var(ddof=1)/len(g1) + g2.var(ddof=1)/len(g2))

    if alternative == 'two-sided':
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = mean_diff + t_crit * se_diff
    elif alternative == 'greater':
        t_crit = stats.t.ppf(0.95, df)
        ci_lower = mean_diff - t_crit * se_diff
        ci_upper = float('inf')
    else: # less
        t_crit = stats.t.ppf(0.95, df)
        ci_lower = float('-inf')
        ci_upper = mean_diff + t_crit * se_diff

    return {
        'test_name': tn,
        'statistic': r_stat(stat),
        'df': round(float(df), 2),
        'p_value': r_p(p),
        'alternative': alt_map.get(alternative, alternative),
        'significance': interp_p(p),
        'cohens_d': r_stat(d),
        'effect_interpretation': interp_d(d),
        'mean_diff': r_mean(mean_diff),
        'diff_ci_lower': r_se(ci_lower) if ci_lower not in [float('inf'), float('-inf')] else ("-∞" if ci_lower == float('-inf') else "∞"),
        'diff_ci_upper': r_se(ci_upper) if ci_upper not in [float('inf'), float('-inf')] else ("-∞" if ci_upper == float('-inf') else "∞")
    }


def run_paired(pre, post, alternative='two-sided', pre_name='Pre', post_name='Post'):
    """대응표본 t-test"""
    stat, p = stats.ttest_rel(pre, post, alternative=alternative)
    df = len(pre) - 1
    # 방향 수정: post - pre (변화량)
    diff = post - pre
    d = cohen_d_paired(diff)
    
    alt_map = {
        'two-sided': f'양쪽검정(two-tailed, {pre_name} - {post_name} ≠ 0)',
        'greater': f'한쪽검정(one-tailed, {pre_name} - {post_name} > 0)',
        'less': f'한쪽검정(one-tailed, {pre_name} - {post_name} < 0)'
    }
    
    # CI calculation for difference
    se_diff = diff.std(ddof=1) / np.sqrt(len(diff))
    mean_diff_val = diff.mean()
    
    if alternative == 'two-sided':
        t_crit = stats.t.ppf(0.975, df)
        ci_lower = mean_diff_val - t_crit * se_diff
        ci_upper = mean_diff_val + t_crit * se_diff
    elif alternative == 'greater':
        t_crit = stats.t.ppf(0.95, df)
        ci_lower = mean_diff_val - t_crit * se_diff
        ci_upper = float('inf')
    else:
        t_crit = stats.t.ppf(0.95, df)
        ci_lower = float('-inf')
        ci_upper = mean_diff_val + t_crit * se_diff

    return {
        'test_name': 'Paired t-test',
        'statistic': r_stat(stat),
        'df': int(df),
        'p_value': r_p(p),
        'alternative': alt_map.get(alternative, alternative),
        'significance': interp_p(p),
        'cohens_d': r_stat(d),
        'effect_interpretation': interp_d(d),
        'mean_diff': r_mean(mean_diff_val),
        'std_diff': r_se(diff.std(ddof=1)),
        'diff_direction_note': 'Difference calculated as (Post - Pre)',
        'diff_ci_lower': r_se(ci_lower) if ci_lower not in [float('inf'), float('-inf')] else ("-∞" if ci_lower == float('-inf') else "∞"),
        'diff_ci_upper': r_se(ci_upper) if ci_upper not in [float('inf'), float('-inf')] else ("-∞" if ci_upper == float('-inf') else "∞")
    }


def run_anova(groups, gnames):
    """One-way ANOVA (F-test)"""
    f_stat, p = stats.f_oneway(*groups)
    
    all_data = np.concatenate(groups)
    grand_mean = all_data.mean()
    N = len(all_data)
    k = len(groups)
    
    ssb = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ssw = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    sst = ssb + ssw
    
    dfb = k - 1
    dfw = N - k
    dft = N - 1
    
    msb = ssb / dfb if dfb > 0 else 0
    msw = ssw / dfw if dfw > 0 else 0
    
    eta = eta_squared(groups)
    
    return {
        'test_name': 'One-way ANOVA',
        'f_statistic': r_stat(f_stat),
        'p_value': r_p(p),
        'significance': interp_p(p),
        'eta_squared': r_stat(eta),
        'effect_interpretation': interp_eta(eta),
        'anova_table': {
            'between': {'source': '집단 간(Between)', 'ss': r_se(ssb), 'df': int(dfb), 
                        'ms': r_se(msb), 'f': r_stat(f_stat), 'p': r_p(p)},
            'within': {'source': '집단 내(Within)', 'ss': r_se(ssw), 'df': int(dfw), 
                       'ms': r_se(msw), 'f': '', 'p': ''},
            'total': {'source': '전체(Total)', 'ss': r_se(sst), 'df': int(dft), 
                      'ms': '', 'f': '', 'p': ''}
        }
    }


def run_welch_anova(groups, gnames):
    """Welch's ANOVA (이분산 가정 시)"""
    try:
        from pingouin import welch_anova
        # 데이터프레임으로 변환하여 사용해야 함 (pingouin interface)
        dfs = []
        for g, name in zip(groups, gnames):
            dfs.append(pd.DataFrame({'dv': g, 'group': name}))
        df_all = pd.concat(dfs)
        
        wa = welch_anova(data=df_all, dv='dv', between='group')
        f_stat = wa['F'][0]
        p_val = wa['p-unc'][0]
        df1 = wa['ddof1'][0]
        df2 = wa['ddof2'][0]
        
        return {
            'test_name': "Welch's ANOVA",
            'f_statistic': r_stat(f_stat),
            'p_value': r_p(p_val),
            'df1': int(df1),
            'df2': round(float(df2), 2),
            'significance': interp_p(p_val),
            'note': "등분산 가정이 기각되어 Welch's ANOVA를 수행했습니다."
        }
        
    except ImportError:
        # scipy.stats.alexandergovern (SciPy 1.7+) as fallback
        try:
            res = stats.alexandergovern(*groups)
            return {
                'test_name': "Alexander-Govern Test (Welch's ANOVA approx)",
                'statistic': r_stat(res.statistic),
                'p_value': r_p(res.pvalue),
                'significance': interp_p(res.pvalue),
                'note': "등분산 가정 기각됨. pingouin 미설치로 scipy Alexander-Govern 수행."
            }
        except AttributeError:
             return {
                'test_name': "Welch's ANOVA (Not Available)",
                'error': "pingouin 패키지 또는 최신 scipy가 필요합니다."
            }


def scheffe_manual(groups, gnames, alpha=0.05):
    """Scheffé Post-hoc (Manual implementation)"""
    # Scheffé는 보수적이므로 수동 계산도 널리 쓰임
    ad = np.concatenate(groups)
    N = len(ad)
    k = len(groups)
    ssw = sum(np.sum((g - g.mean()) ** 2) for g in groups)
    msw = ssw / (N - k)
    
    results = []
    for i in range(len(gnames)):
        for j in range(i + 1, len(gnames)):
            g1, g2 = groups[i], groups[j]
            md = g1.mean() - g2.mean()
            se = np.sqrt(msw * (1 / len(g1) + 1 / len(g2)))
            
            # F-value calculation
            f_val = (md / se) ** 2 / (k - 1) if se > 0 else 0
            p_val = 1 - stats.f.cdf(f_val, k - 1, N - k)
            
            results.append({
                'group1': str(gnames[i]),
                'group2': str(gnames[j]),
                'mean_diff': r_mean(md),
                'p_value': r_p(p_val),
                'significant': bool(p_val < alpha)
            })
    return results


def bonferroni_welch_pairwise(groups, gnames, alpha=0.05):
    """Bonferroni-corrected Welch t-test (Dunnett's T3 근사)"""
    # 이분산 가정 사후검정
    results = []
    k = len(gnames)
    
    for i in range(len(gnames)):
        for j in range(i + 1, len(gnames)):
            g1, g2 = groups[i], groups[j]
            n1, n2 = len(g1), len(g2)
            v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
            md = g1.mean() - g2.mean()
            se = np.sqrt(v1 / n1 + v2 / n2)
            
            if se == 0:
                results.append({
                    'group1': str(gnames[i]), 'group2': str(gnames[j]), 
                    'mean_diff': 0.0, 'p_value': 1.0, 'significant': False
                })
                continue
                
            t_val = abs(md) / se
            
            # Welch-Satterthwaite df
            num = (v1 / n1 + v2 / n2) ** 2
            den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            df = num / den if den > 0 else n1 + n2 - 2
            
            # Bonferroni-corrected Welch t-test as a robust fallback 
            p_unc = 2 * (1 - stats.t.cdf(t_val, df))
            p_adj = min(p_unc * k * (k-1) / 2, 1.0) 
            
            results.append({
                'group1': str(gnames[i]),
                'group2': str(gnames[j]),
                'mean_diff': r_mean(md),
                'p_value': r_p(p_adj),
                'significant': bool(p_adj < alpha),
                'note': "Bonferroni-corrected Welch t-test (Dunnett's T3 근사)"
            })
    return results


def run_posthoc(data, dv, iv, gnames, alpha=0.05):
    """사후검정 종합 실행"""
    results = {}
    groups = [data[data[iv] == g][dv].values for g in gnames]
    
    # 1. Tukey HSD (statsmodels)
    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        tukey = pairwise_tukeyhsd(endog=data[dv], groups=data[iv], alpha=alpha)
        
        tr = []
        # tukey.summary().data contains header + rows
        summary_data = tukey.summary().data[1:] 
        for row in summary_data:
            # group1, group2, meandiff, p-adj, lower, upper, reject
            tr.append({
                'group1': str(row[0]),
                'group2': str(row[1]),
                'mean_diff': r_mean(row[2]),
                'p_value': r_p(row[3]),
                'ci_lower': r_se(row[4]),
                'ci_upper': r_se(row[5]),
                'reject': bool(row[6]),
                'significant': bool(row[6])
            })
        results['tukey'] = tr
    except ImportError:
        results['tukey'] = "statsmodels 패키지 설치 필요 (분석 생략)"
        
    # 2. Scheffé (Manual)
    results['scheffe'] = scheffe_manual(groups, gnames, alpha=alpha)
    
    # 3. Duncan (Substitute)
    # Duncan is not available in standard packages. 
    # Providing Pairwise t-test with Bonferroni correction as a strict post-hoc.
    try:
        from statsmodels.stats.multicomp import MultiComparison
        mc = MultiComparison(data[dv], data[iv])
        # Bonferroni corrected pairwise t-tests
        bonf_res = mc.allpairtest(stats.ttest_ind, method='bonferroni')
        
        br = []
        try:
            # Statsmodels version compatibility check for header parsing
            # allpairtest result structure is table with headers
            header = bonf_res[0].data[0]
            # Try to find indices, fallback to defaults if not found
            pval_idx = 5 # default for p_adj
            reject_idx = 6 # default for reject
            
            # Dynamic header search
            for idx, col in enumerate(header):
                if 'pval_corr' in str(col) or 'adj' in str(col):
                    pval_idx = idx
                elif 'reject' in str(col):
                    reject_idx = idx

            for row in bonf_res[0].data[1:]:
                 br.append({
                    'group1': str(row[0]),
                    'group2': str(row[1]),
                    'p_value': r_p(row[pval_idx]), 
                    'significant': bool(row[reject_idx])
                })
        except Exception:
            # Fallback if structure is unexpected
            br = "결과 파싱 실패 (statsmodels 버전 호환성 문제)"
            
        results['duncan'] = br
        results['duncan_note'] = "Duncan MRT 미지원. 대안으로 Pairwise t-test (Bonferroni) 수행."
    except ImportError:
        results['duncan'] = "statsmodels 패키지 설치 필요"
        
    # 4. Dunnett's T3 (Heteroscedasticity)
    # Try pingouin Games-Howell if available, else manual fallback
    try:
        import pingouin as pg
        gh = pg.pairwise_gameshowell(data=data, dv=dv, between=iv)
        gr = []
        for _, row in gh.iterrows():
            gr.append({
                'group1': str(row['A']),
                'group2': str(row['B']),
                'mean_diff': r_mean(row['diff']),
                'p_value': r_p(row['pval']),
                'significant': bool(row['pval'] < alpha)
            })
        results['games_howell_or_fallback'] = gr
        results['games_howell_or_fallback_note'] = "Games-Howell Test (pingouin)"
    except ImportError:
        results['games_howell_or_fallback'] = bonferroni_welch_pairwise(data, dv, iv, gnames, alpha=alpha)
        
    return results


def plot_dist(data, dv, iv, output_path):
    """Mean ± 95% CI dot-and-dashed-line chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Group statistics
    unique_grps = sorted(data[iv].unique())
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(unique_grps), 3)))
    
    means = []
    ci_lowers = []
    ci_uppers = []
    labels = []
    n_list = []
    
    for idx, name in enumerate(unique_grps):
        grp = data[data[iv] == name][dv].dropna()
        n = len(grp)
        m = grp.mean()
        sd = grp.std(ddof=1)
        se = sd / np.sqrt(n)
        
        # 95% CI (t-distribution)
        t_crit = stats.t.ppf(0.975, n - 1)
        ci_lower = m - t_crit * se
        ci_upper = m + t_crit * se
        
        means.append(m)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)
        labels.append(str(name))
        n_list.append(n)
    
    x_pos = np.arange(len(labels))
    
    for i in range(len(labels)):
        # Dashed line from mean to CI lower
        ax.plot([x_pos[i], x_pos[i]], [means[i], ci_lowers[i]], 
                color=colors[i], linewidth=2, linestyle='--', zorder=3)
        # Dashed line from mean to CI upper
        ax.plot([x_pos[i], x_pos[i]], [means[i], ci_uppers[i]], 
                color=colors[i], linewidth=2, linestyle='--', zorder=3)
        
        # CI endpoint caps (horizontal ticks)
        cap_w = 0.06
        ax.plot([x_pos[i] - cap_w, x_pos[i] + cap_w], [ci_lowers[i], ci_lowers[i]], 
                color=colors[i], linewidth=2, solid_capstyle='round', zorder=4)
        ax.plot([x_pos[i] - cap_w, x_pos[i] + cap_w], [ci_uppers[i], ci_uppers[i]], 
                color=colors[i], linewidth=2, solid_capstyle='round', zorder=4)
        
        # Mean dot
        ax.scatter(x_pos[i], means[i], color=colors[i], s=120, zorder=5, 
                   edgecolors='white', linewidths=1.5)
        
        # Mean value label
        ax.annotate(f'M={means[i]:.2f}\n(n={n_list[i]})', 
                    xy=(x_pos[i], means[i]),
                    xytext=(12, 8), textcoords='offset points',
                    fontsize=9, color=colors[i], fontweight='bold',
                    ha='left')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel(dv, fontsize=12)
    ax.set_title(f'Mean Comparison by Group: {dv}', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='-')
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
               markersize=10, label=f'{labels[i]}')
        for i in range(len(labels))
    ]
    legend_elements.append(
        Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label='95% CI')
    )
    ax.legend(handles=legend_elements, fontsize=10, loc='best')
    
    # Margins
    y_range = max(ci_uppers) - min(ci_lowers)
    ax.set_ylim(min(ci_lowers) - y_range * 0.3, max(ci_uppers) + y_range * 0.3)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(result, output_dir, chart_filename, alpha=0.05):
    """마크다운 보고서 생성 및 저장"""
    today = datetime.datetime.now().strftime("%Y%m%d")
    test_method_map = {
        'independent': 'independent-ttest',
        'paired': 'paired-ttest',
        'anova': 'oneway-anova'
    }
    method_name = test_method_map.get(result.get('test_type', ''), result.get('test_type', 'analysis'))
    report_filename = f"{today}-{method_name}-result.md"
    report_path = os.path.join(output_dir, 'report', report_filename)
    
    os.makedirs(os.path.join(output_dir, 'report'), exist_ok=True)
    
    # 차트 파일 복사 (report 폴더로)
    chart_src = os.path.join(output_dir, chart_filename)
    chart_dst = os.path.join(output_dir, 'report', chart_filename)
    if os.path.exists(chart_src):
        shutil.copy(chart_src, chart_dst)
    
    # 보고서 내용 작성
    lines = []
    # p < .001 인 경우 각주에 정확한 값을 모으기 위한 리스트
    footnotes = []   # (참조기호, 설명)
    fn_counter = [0] # mutable for closure
    
    def _fp(p_raw):
        """보고서 p-value 표시: < .001이면 각주 참조 기호 추가"""
        if p_raw is None or p_raw == '' or p_raw == '-':
            return str(p_raw)
        p = float(p_raw)
        if p < 0.001:
            fn_counter[0] += 1
            ref_label = f"†{fn_counter[0]}"
            ref_sup = f"<sup>†{fn_counter[0]}</sup>"
            if p == 0.0 or p < 1e-6:
                footnotes.append((ref_label, "p < .000001"))
            else:
                footnotes.append((ref_label, f"p = {p:.6f}"))
            return f"< .001 {ref_sup}"
        return f"{p:.{DEC_P}f}"
    
    today_display = datetime.datetime.now().strftime("%Y-%m-%d")
    lines.append("# 평균비교 분석 보고서")
    lines.append(f'<div style="text-align: right;">작성일 : {today_display}</div>')
    lines.append("")
    lines.append(f"## 1. 분석 개요")
    if result['test_type'] == 'paired':
        lines.append(f"- **시점 1**: {result['dv']}")
        lines.append(f"- **시점 2**: {result.get('iv2', result['iv'])}")
    else:
        lines.append(f"- **종속변수**: {result['dv']}")
        lines.append(f"- **독립변수**: {result['iv']}")
    lines.append(f"- **분석 방법**: {result.get('test_result', {}).get('test_name', result['test_type'])}")
    lines.append(f"- **유의수준**: α = {alpha}")
    if result['test_type'] not in ('anova',):
        lines.append(f"- **검정 방향**: {result.get('test_result', {}).get('alternative', 'two-sided')}")
    lines.append("")
    
    # ── 정규성 검정 ──
    lines.append(f"## 2. 정규성 검정 및 분포 확인")
    norm_res = result.get('normality_test', [])
    if norm_res:
        lines.append("| 집단 | N | Shapiro-Wilk W | p-value | 정규성 판정 |")
        lines.append("|---|---|---|---|---|")
        for nr in norm_res:
            w_val = f"{nr['w_statistic']:.{DEC_STAT}f}" if nr.get('w_statistic') is not None else '-'
            p_disp = _fp(nr.get('p_value'))
            lines.append(f"| {nr['group']} | {nr.get('n', '-')} | {w_val} | {p_disp} | {nr.get('is_normal', '-')} |")
        lines.append("")
        
        if result.get('nonparametric_recommended', False):
            lines.append("> [!WARNING]")
            lines.append("> 표본 크기가 작거나(n≤20) 정규성 가정이 기각되었습니다.")
            
            tt = result['test_type']
            if tt == 'independent':
                lines.append("> **비모수 검정 권장**: Mann-Whitney U test")
            elif tt == 'paired':
                lines.append("> **비모수 검정 권장**: Wilcoxon signed-rank test")
            elif tt == 'anova':
                lines.append("> **비모수 검정 권장**: Kruskal-Wallis test")
            lines.append("")
    
    # ── 기술통계 ──
    lines.append(f"## 3. 기술통계")
    lines.append("| 집단 | N | 평균(M) | 표준편차(SD) | 표준오차(SE) | 95% CI (각 집단 평균) |")
    lines.append("|---|---|---|---|---|---|")
    for d in result.get('descriptives', []):
        m = f"{d['mean']:.{DEC_MEAN}f}"
        sd = f"{d['std']:.{DEC_SE}f}"
        se = f"{d['se']:.{DEC_SE}f}"
        ci_l = fmt_ci_val(d.get('mean_ci_lower', '-'))
        ci_u = fmt_ci_val(d.get('mean_ci_upper', '-'))
        lines.append(f"| {d['group']} | {d['n']} | {m} | {sd} | {se} | [{ci_l}, {ci_u}] |")
    lines.append("")
    
    # ── 검정 결과 ──
    lines.append(f"## 4. 검정 결과")
    
    tr = result.get('test_result', {})
    
    if result['test_type'] in ['independent', 'paired']:
        lines.append(f"### {tr.get('test_name', 't-test')}")
        t_val = f"{tr.get('statistic', 0):.{DEC_STAT}f}"
        p_disp = _fp(tr.get('p_value'))
        d_val = f"{tr.get('cohens_d', 0):.{DEC_STAT}f}"
        ci_l = fmt_ci_val(tr.get('diff_ci_lower', '-'))
        ci_u = fmt_ci_val(tr.get('diff_ci_upper', '-'))
        
        lines.append(f"| 통계량(t) | 자유도(df) | 유의확률(p) | Cohen's d | 효과크기 해석 |")
        lines.append(f"|---|---|---|---|---|")
        lines.append(f"| {t_val} | {tr.get('df')} | {p_disp} {tr.get('significance', '')} | {d_val} | {tr.get('effect_interpretation', '')} |")
        lines.append("")
        lines.append(f"- **평균 차이**: {tr.get('mean_diff', '-'):.{DEC_MEAN}f}" if isinstance(tr.get('mean_diff'), (int, float)) else f"- **평균 차이**: {tr.get('mean_diff', '-')}")
        lines.append(f"- **평균 차이의 95% CI**: [{ci_l}, {ci_u}]")
        lines.append(f"- **검정 방향**: {tr.get('alternative')}")
        lines.append("")
        
    elif result['test_type'] == 'anova':
        lines.append(f"### {tr.get('test_name', 'ANOVA')}")
        f_val = f"{tr.get('f_statistic', 0):.{DEC_STAT}f}"
        p_disp = _fp(tr.get('p_value'))
        eta_val = f"{tr.get('eta_squared', 0):.{DEC_STAT}f}"
        
        lines.append(f"- **통계량**: F = {f_val}, p = {p_disp} ({tr.get('significance')})")
        lines.append(f"- **효과크기**: η² = {eta_val} ({tr.get('effect_interpretation')})")
        
        tbl = tr.get('anova_table', {})
        if tbl:
            lines.append("")
            lines.append("**분산분석표**")
            lines.append("| 요인 | SS | df | MS | F | p |")
            lines.append("|---|---|---|---|---|---|")
            
            for r_key in ['between', 'within', 'total']:
                if r_key in tbl:
                    td = tbl[r_key]
                    f_cell = f"{td['f']:.{DEC_STAT}f}" if isinstance(td.get('f'), (int, float)) else td.get('f', '')
                    p_cell = _fp(td['p']) if isinstance(td.get('p'), (int, float)) else td.get('p', '')
                    ss_cell = f"{td['ss']:.{DEC_SE}f}" if isinstance(td.get('ss'), (int, float)) else td.get('ss', '')
                    ms_cell = f"{td['ms']:.{DEC_SE}f}" if isinstance(td.get('ms'), (int, float)) else td.get('ms', '')
                    lines.append(f"| {td.get('source')} | {ss_cell} | {td.get('df')} | {ms_cell} | {f_cell} | {p_cell} |")
            lines.append("")
        
        # Welch ANOVA 결과 추가
        if 'welch_anova' in result:
             wa = result['welch_anova']
             wa_f = f"{wa.get('f_statistic', 0):.{DEC_STAT}f}"
             wa_p = _fp(wa.get('p_value'))
             lines.append("> [!WARNING]")
             lines.append(f"> **Welch's ANOVA** (이분산 보정)")
             lines.append(f"> - F({wa.get('df1')}, {wa.get('df2')}) = {wa_f}, p = {wa_p}")
             lines.append(f"> - {wa.get('note')}")
             lines.append("")
             
    # ── 등분산 검정 결과 ──
    lt = result.get('levene_test')
    if lt:
        lt_f = f"{lt['statistic']:.{DEC_STAT}f}"
        lt_p = _fp(lt['p_value'])
        lines.append(f"### 등분산 검정 (Levene)")
        lines.append(f"- F({lt['df1']}, {lt['df2']}) = {lt_f}, p = {lt_p}")
        lines.append(f"- **해석**: {lt['interpretation']}")
        lines.append("")
        
    # ── 사후검정 (ANOVA only) ──
    if 'posthoc' in result:
        lines.append(f"## 5. 사후검정 (Post-hoc)")
        ph = result['posthoc']
        
        if 'tukey' in ph and isinstance(ph['tukey'], list):
            lines.append("### Tukey HSD")
            lines.append("| Grp1 | Grp2 | Diff | p-adj | 95% CI | Sig |")
            lines.append("|---|---|---|---|---|---|")
            for r in ph['tukey']:
                sig = "**Yes**" if r['reject'] else "No"
                diff_s = f"{r['mean_diff']:.{DEC_MEAN}f}" if isinstance(r.get('mean_diff'), (int, float)) else r['mean_diff']
                p_s = _fp(r['p_value'])
                ci_l = f"{r['ci_lower']:.{DEC_SE}f}" if isinstance(r.get('ci_lower'), (int, float)) else r['ci_lower']
                ci_u = f"{r['ci_upper']:.{DEC_SE}f}" if isinstance(r.get('ci_upper'), (int, float)) else r['ci_upper']
                lines.append(f"| {r['group1']} | {r['group2']} | {diff_s} | {p_s} | [{ci_l}, {ci_u}] | {sig} |")
            lines.append("")
            
        if 'scheffe' in ph and isinstance(ph['scheffe'], list):
            lines.append("### Scheffé")
            lines.append("| Grp1 | Grp2 | Diff | p-val | Sig |")
            lines.append("|---|---|---|---|---|")
            for r in ph['scheffe']:
                sig = "**Yes**" if r['significant'] else "No"
                diff_s = f"{r['mean_diff']:.{DEC_MEAN}f}" if isinstance(r.get('mean_diff'), (int, float)) else r['mean_diff']
                p_s = _fp(r['p_value'])
                lines.append(f"| {r['group1']} | {r['group2']} | {diff_s} | {p_s} | {sig} |")
            lines.append("")
            
        if 'games_howell_or_fallback' in ph and isinstance(ph['games_howell_or_fallback'], list):
            lines.append(f"### {ph.get('games_howell_or_fallback_note', 'Games-Howell / Welch t-test')}")
            lines.append("| Grp1 | Grp2 | Diff | p-val | Sig | Note |")
            lines.append("|---|---|---|---|---|---|")
            for r in ph['games_howell_or_fallback']:
                sig = "**Yes**" if r['significant'] else "No"
                diff_s = f"{r['mean_diff']:.{DEC_MEAN}f}" if isinstance(r.get('mean_diff'), (int, float)) else r['mean_diff']
                p_s = _fp(r['p_value'])
                note = r.get('note', '')
                lines.append(f"| {r['group1']} | {r['group2']} | {diff_s} | {p_s} | {sig} | {note} |")
            lines.append("")
            
    # ── 결론 ──
    tr = result.get('test_result', {})
    p_val = tr.get('p_value')
    p_num = float(p_val) if p_val is not None and p_val != '' else None
    is_sig = p_num is not None and p_num < alpha

    # 섹션 번호 결정: ANOVA는 사후검정(5)이 있으므로 6, 그 외는 5
    conclusion_num = 6 if result['test_type'] == 'anova' else 5
    chart_num = conclusion_num + 1

    lines.append(f"## {conclusion_num}. 결론")

    # 통계량 요약 문자열 구성
    test_name = tr.get('test_name', '')
    p_display = _fp(p_val) if p_val is not None else '-'

    if result['test_type'] == 'anova':
        f_val = tr.get('f_statistic', '')
        anova_tbl = tr.get('anova_table', {})
        df_between = anova_tbl.get('between', {}).get('df', '')
        df_within = anova_tbl.get('within', {}).get('df', '')
        eta_sq = tr.get('eta_squared', '')
        stat_str = f"F({df_between}, {df_within}) = {f_val}, p = {p_display}"
        if eta_sq:
            stat_str += f", η² = {eta_sq}"
    elif result['test_type'] in ('independent', 'paired'):
        t_val = tr.get('statistic', '')
        df_val = tr.get('df', '')
        d_val = tr.get('cohens_d', '')
        stat_str = f"t({df_val}) = {t_val}, p = {p_display}"
        if d_val:
            stat_str += f", d = {d_val}"
    else:
        stat_str = f"p = {p_display}"

    if is_sig:
        lines.append(f"> **유의수준 α = {alpha}에서 통계적으로 유의한 차이가 있습니다.**")
    else:
        lines.append(f"> 유의수준 α = {alpha}에서 통계적으로 유의한 차이가 없습니다.")

    lines.append("")
    lines.append(f"- **주요 통계량**: {stat_str}")

    # 효과크기 해석 추가
    effect_interp = tr.get('effect_interpretation', '')
    if effect_interp:
        lines.append(f"- **효과크기 해석**: {effect_interp}")

    # ANOVA 사후검정 결론 요약 (Tukey HSD 기준)
    if result['test_type'] == 'anova' and is_sig:
        ph = result.get('posthoc', {})
        tukey_results = ph.get('tukey', [])
        if isinstance(tukey_results, list) and tukey_results:
            lines.append("")
            lines.append("**사후검정 결과 요약 (Tukey HSD)**")
            sig_pairs = []
            nonsig_pairs = []
            for r in tukey_results:
                pair = f"{r['group1']} vs {r['group2']}"
                if r.get('significant') or r.get('reject'):
                    sig_pairs.append(f"{pair} (Diff = {r['mean_diff']}, p = {_fp(r['p_value'])})")
                else:
                    nonsig_pairs.append(f"{pair} (p = {_fp(r['p_value'])})")
            if sig_pairs:
                lines.append("- 유의한 차이가 있는 집단 쌍:")
                for sp in sig_pairs:
                    lines.append(f"  - {sp}")
            if nonsig_pairs:
                lines.append("- 유의한 차이가 없는 집단 쌍:")
                for np_ in nonsig_pairs:
                    lines.append(f"  - {np_}")

    lines.append("")

    # ── 시각화 ──
    lines.append(f"## {chart_num}. 시각화")
    lines.append(f"![Distribution Chart]({chart_filename})")
    lines.append("> 각 집단의 평균(M)을 점으로, 95% 신뢰구간(CI)을 점선 오차막대로 나타냅니다.")
    
    # ── 각주 (p < .001 참조) ──
    if footnotes:
        lines.append("")
        lines.append("---")
        lines.append("### 유의확률 참조")
        for ref, desc in footnotes:
            lines.append(f"- <sup>{ref}</sup>: {desc}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
        
    return report_path


def main():
    parser = argparse.ArgumentParser(description='평균비교 검정 분석')
    
    # 환경변수 기본값 적용
    sk_dir = os.environ.get('SKILL_DIR', '.')
    out_dir_def = os.environ.get('OUTPUT_DIR', '.')
    
    parser.add_argument('--data', required=True, help='데이터 파일 경로')
    parser.add_argument('--dv', required=True, help='종속변수 (또는 paired wide의 사전 변수)')
    parser.add_argument('--iv', default=None, help='독립변수 (또는 paired long의 시점 변수)')
    parser.add_argument('--iv2', default=None, help='paired wide의 사후 변수')
    parser.add_argument('--id_col', default=None, help='paired long의 ID 변수')
    parser.add_argument('--test_type', required=True, choices=['independent', 'paired', 'anova'])
    parser.add_argument('--alternative', default='two-sided', choices=['two-sided', 'greater', 'less'])
    parser.add_argument('--equal_var', default='true', choices=['true', 'false'])
    parser.add_argument('--alpha', type=float, default=0.05, help='유의수준 (기본값: 0.05)')
    parser.add_argument('--output_dir', default=out_dir_def)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        data = load_data(args.data)
    except Exception as e:
        print(json.dumps({'error': f"데이터 로드 실패: {str(e)}"}, ensure_ascii=False))
        sys.exit(1)
        
    result = {
        'test_type': args.test_type,
        'dv': args.dv,
        'iv': args.iv or '',
        'iv2': args.iv2 or '',
        'generated_at': datetime.datetime.now().isoformat()
    }

    try:
        # ── Independent t-test ──
        if args.test_type == 'independent':
            if not args.iv: raise ValueError("--iv 필요")
            
            # 결측 제거
            df_clean = data[[args.dv, args.iv]].dropna()
            
            grps = df_clean.groupby(args.iv)[args.dv]
            gn = list(grps.groups.keys())
            
            if len(gn) != 2:
                raise ValueError(f"2개 집단 필요, {len(gn)}개 발견 ({gn})")
                
            g1 = grps.get_group(gn[0]).values
            g2 = grps.get_group(gn[1]).values
            group_list = [g1, g2]
            
            # 정규성 검정
            norm_res, rec_nonparam = normality_test(group_list, gn, alpha=args.alpha)
            result['normality_test'] = norm_res
            result['nonparametric_recommended'] = rec_nonparam
            
            # 등분산 검정
            result['levene_test'] = levene_test(group_list, alpha=args.alpha)
            
            # 기술통계
            result['descriptives'] = group_desc(df_clean, args.dv, args.iv, args.alternative)
            
            # T-test 실행
            # 등분산 옵션이 false 이거나, Levene 검정에서 equal_var=False가 나오면 Welch 고려 가능
            # 하지만 사용자 입력(args.equal_var)이 우선하므로 이를 따름
            use_equal_var = (str(args.equal_var).lower() == 'true')
            
            result['test_result'] = run_ind_ttest(g1, g2, equal_var=use_equal_var, alternative=args.alternative, g1_name=str(gn[0]), g2_name=str(gn[1]))
            
            # 차트
            cp = 'distribution_chart.png'
            plot_dist(df_clean, args.dv, args.iv, os.path.join(args.output_dir, cp))
            result['chart_path'] = cp

        # ── Paired t-test ──
        elif args.test_type == 'paired':
            if args.iv2:
                # Wide format
                pair_df = data[[args.dv, args.iv2]].dropna()
                if len(pair_df) == 0: raise ValueError("유효한 데이터가 없습니다.")
                
                pre = pair_df[args.dv].values
                post = pair_df[args.iv2].values
                
                # 정규성 검정 (차이값에 대해 수행하는 것이 원칙이나, 각각 분포도 확인)
                result['normality_test'], result['nonparametric_recommended'] = normality_test([post-pre], ["Difference"], alpha=args.alpha)
                
                # 기술통계 (직접 구성)
                desc = []
                for label, vals in [(args.dv, pre), (args.iv2, post)]:
                    n = len(vals); m = np.mean(vals); s = np.std(vals, ddof=1); se = s / np.sqrt(n)
                    # Simple two-sided CI for display
                    tc = stats.t.ppf(0.975, n-1)
                    desc.append({
                        'group': label, 'n': int(n), 'mean': r_mean(m),
                        'std': r_se(s), 'se': r_se(se),
                        'mean_ci_lower': r_se(m - tc * se),
                        'mean_ci_upper': r_se(m + tc * se)
                    })
                result['descriptives'] = desc
                
                result['test_result'] = run_paired(pd.Series(pre), pd.Series(post), alternative=args.alternative, pre_name=args.dv, post_name=args.iv2)
                
                # Chart (Melt for display)
                melted = pd.DataFrame({
                    'Group': [args.dv]*len(pre) + [args.iv2]*len(post),
                    'Value': np.concatenate([pre, post])
                })
                cp = 'distribution_chart.png'
                plot_dist(melted, 'Value', 'Group', os.path.join(args.output_dir, cp))
                result['chart_path'] = cp
                result['iv'] = "Paired Groups" # for report
                
            elif args.iv:
                # Long format
                # ID 컬럼 기준 매칭
                if not args.id_col:
                     warnings.warn("Long format paired t-test에서 --id_col 미지정. 단순 정렬 순서로 매칭합니다 (위험).")
                     # 기존 로직 유지하되 경고
                     df_clean = data[[args.dv, args.iv]].dropna()
                     grps = df_clean.groupby(args.iv)[args.dv]
                     gn = sorted(grps.groups.keys())
                     pre = grps.get_group(gn[0]).values
                     post = grps.get_group(gn[1]).values
                     min_len = min(len(pre), len(post))
                     pre = pre[:min_len]
                     post = post[:min_len]
                else:
                    # ID merge
                    df_clean = data[[args.id_col, args.iv, args.dv]].dropna()
                    grps = df_clean.groupby(args.iv)
                    gn = sorted(grps.groups.keys())
                    if len(gn) != 2: raise ValueError(f"집단이 2개여야 합니다: {gn}")
                    
                    df_pre = grps.get_group(gn[0]).set_index(args.id_col)[args.dv]
                    df_post = grps.get_group(gn[1]).set_index(args.id_col)[args.dv]
                    
                    # Inner join
                    merged = pd.concat([df_pre, df_post], axis=1, join='inner').dropna()
                    pre = merged.iloc[:, 0].values
                    post = merged.iloc[:, 1].values
                
                if len(pre) == 0: raise ValueError("매칭되는 대응 쌍이 없습니다.")
                
                result['normality_test'], result['nonparametric_recommended'] = normality_test([post-pre], ["Difference"], alpha=args.alpha)
                result['descriptives'] = group_desc(data.dropna(subset=[args.dv, args.iv]), args.dv, args.iv)
                result['test_result'] = run_paired(pd.Series(pre), pd.Series(post), alternative=args.alternative, pre_name=str(gn[0]), post_name=str(gn[1]))
                
                cp = 'distribution_chart.png'
                plot_dist(data.dropna(subset=[args.dv, args.iv]), args.dv, args.iv, os.path.join(args.output_dir, cp))
                result['chart_path'] = cp
            else:
                 raise ValueError("Paired test: --iv2 (Wide) 또는 --iv (Long) 필요")

        # ── ANOVA ──
        elif args.test_type == 'anova':
            if not args.iv: raise ValueError("--iv 필요")
            df_clean = data[[args.dv, args.iv]].dropna()
            
            grps = df_clean.groupby(args.iv)[args.dv]
            gn = list(grps.groups.keys())
            groups = [grps.get_group(g).values for g in gn]
            
            # 정규성 검정
            result['normality_test'], result['nonparametric_recommended'] = normality_test(groups, gn, alpha=args.alpha)
            
            # 등분산 검정
            lt = levene_test(groups, alpha=args.alpha)
            result['levene_test'] = lt
            
            # ANOVA 실행
            # 등분산이 아닐 경우 Welch ANOVA 수행
            if not lt['equal_variance']:
                result['test_result'] = run_anova(groups, gn) # 기본 ANOVA 결과도 제공
                result['welch_anova'] = run_welch_anova(groups, gn) # Welch 추가
                # Report generation logic might need to choose one, currently JSON includes both
            else:
                result['test_result'] = run_anova(groups, gn)
                
            # 기술통계
            result['descriptives'] = group_desc(df_clean, args.dv, args.iv)
            
            # 사후검정
            result['posthoc'] = run_posthoc(df_clean, args.dv, args.iv, gn, alpha=args.alpha)
            
            # 차트
            cp = 'distribution_chart.png'
            plot_dist(df_clean, args.dv, args.iv, os.path.join(args.output_dir, cp))
            result['chart_path'] = cp

        # ── 결과 저장 ──
        oj = os.path.join(args.output_dir, 'analysis_result.json')
        with open(oj, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        # 보고서 자동 생성
        if 'chart_path' in result:
             generate_report(result, args.output_dir, result['chart_path'], alpha=args.alpha)
             
    except Exception as e:
        err = {'error': str(e), 'type': type(e).__name__}
        print(json.dumps(err, ensure_ascii=False))
        sys.exit(1)
        # raise

if __name__ == '__main__':
    main()
