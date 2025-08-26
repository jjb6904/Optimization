#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization_mat.py
전체 상품을 한꺼번에 처리하는 주문 완료량 추적 메인 코드
"""

import pandas as pd
import asso 
import opti_vrp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path 

def _build_start_end_minutes(line_schedules, dish_quantity, changeover_df): 
    """
    각 라인에 대해 (시작시간 종료시간)을 분 단위로 계산하는 함수
    
    계산방식
    - 첫 상품: start=0, end=start+조리시간(cook)
    - 이후 상품: start=이전 상품의 end + changeover, end=start + 조리시간(cook)

    입력
    - line_schedules : dict[int, list[str]]
      {라인 ID : [제품명 순서]} 형태의 스케줄 정보

    - dish_quantity : dict[str, int]
      {제품명: 총 수량} 형태의 생산 수량 정보
        
    - changeover_df : pd.DataFrame
      제품 간 전환 시간(분) 매트릭스. index=이전 제품, columns=다음 제품.


    반환: list[dict(line_id, sequence(작업순서), product(제품명)), start_min, end_min, cook_min))] 
    """

    rows = []
    for line_id, seq in line_schedules.items():
        prev = None
        current_end = 0.0
        for i, dish in enumerate(seq, start=1):
            cook = float(opti_vrp.get_cooking_time(dish, dish_quantity.get(dish, 1)))
            if prev is None:
                start_min = 0.0
            else:
                change = float(changeover_df.loc[prev, dish])
                start_min = current_end + change
            end_min = start_min + cook
            rows.append({
                "line_id": int(line_id),
                "sequence": i,
                "product": dish,
                "cook_min": round(cook, 4),
                "start_min": round(start_min, 4),
                "end_min": round(end_min, 4),
            })
            prev = dish
            current_end = end_min
    return rows

def make_timeline_df(line_schedules, dish_quantity, changeover_df,
                     code_map: dict[str, str]):
    """
    라인별 시작 기준시각을 지정하고 타임라인 DataFrame 생성.
    반환 컬럼: ['상품명','상품코드','시작시간','완료시간','라인','순서','작업시간(분)']
    """
    raw = _build_start_end_minutes(line_schedules, dish_quantity, changeover_df)
    out_rows = []
    for r in raw:
        line_id = r["line_id"]
        start_min = round(r["start_min"], 2)
        end_min   = round(r["end_min"], 2)
        prod = r["product"]
        out_rows.append({
            "상품명": prod,
            "상품코드": code_map.get(prod, ""),
            "시작시간": start_min,
            "완료시간": end_min,
            "라인": line_id + 1,
            "순서": r["sequence"],
            "작업시간(분)": round(r["cook_min"], 2),
        })
    cols = ["상품명","상품코드","시작시간","완료시간","라인","순서","작업시간(분)"]
    return pd.DataFrame(out_rows)[cols].sort_values(["라인","순서"]).reset_index(drop=True)

def save_timeline_excel(df_timeline: pd.DataFrame,
                        out_path="./outputs/상품_타임라인.xlsx",
                        only_AD=True):
    """
    only_AD=True면 A~D열(상품명,상품코드,시작시간,완료시간)만 저장.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as w:
        if only_AD:
            df_timeline[["상품명","상품코드","시작시간","완료시간"]].to_excel(
                w, index=False, sheet_name="timeline")
        else:
            df_timeline.to_excel(w, index=False, sheet_name="timeline_detail")
    return out_path


# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'Malgun Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 🍎 30분 단위로 완료되는 주문량 추적 함수
def track_order_completion_30min(line_schedules, dish_quantity, changeover_df, df_original):
    """
    30분 단위로 주문 완료량 추적 및 출력
    """
    print("\n🔍 주문 완료량 추적 시작...")
    
    # 🍎 1. 각 라인별 상품 완료 시간 계산 
    # 🍎 상품1 조리시간 + 전환시간 + 상품2 조리시간 + ..... = 라인 총 시간

    line_completion_times = {}
    
    for line_id, sequence in line_schedules.items():
        completion_times = []
        current_time = 0
        
        for i, dish in enumerate(sequence):
            # 조리시간 추가
            cooking_time = opti_vrp.get_cooking_time(dish, dish_quantity.get(dish, 1))
            current_time += cooking_time
            
            # 전환시간 추가 (다음 상품이 있는 경우)
            if i < len(sequence) - 1:
                next_dish = sequence[i + 1]
                changeover_time = changeover_df.loc[dish, next_dish]
                current_time += changeover_time
            
            completion_times.append((dish, current_time))
        
        line_completion_times[line_id] = completion_times
    
    # 🍎 2. 상품별 완료 시간 계산
    # 🍎 콩나물 (1분) + 전환(7분) + 시금치(3분) 일 때
    # 🍎 콩나물 : 1분
    # 🍎 시금치 : 11분

    dish_completion_map = {}
    for line_id, completions in line_completion_times.items():
        for dish, time in completions:
            dish_completion_map[dish] = time
    
    # 🍎 3. 주문별 완료 시간 계산
    # 🍎 각 주문 내에 max 상품 완료 시간을 해당 주문의 완료 시간으로 


    order_completion_times = []
    
    for order_id in df_original['주문번호'].unique():
        order_items = df_original[df_original['주문번호'] == order_id]['상품명'].tolist()
        # 해당 주문의 모든 상품이 완료되는 시간 = 가장 늦게 완료되는 상품의 시간
        order_complete_time = max(dish_completion_map.get(item, 0) for item in order_items)
        order_completion_times.append((order_id, order_complete_time))
    
    # 4. 30분 단위 집계 및 출력
    max_time = max(time for _, time in order_completion_times) if order_completion_times else 0
    time_intervals = np.arange(30, max_time + 30, 30)  # 30, 60, 90, ...
    
    completion_data = {'intervals': [], 'cumulative': [], 'interval_counts': []}
    
    cumulative_count = 0
    

     # 🍎 30분 단위로 완료된 누적 주문 수 뽑는 코드

    print(f"\n📊 30분 단위 주문 완료량 추적:")
    print("-" * 50)
    
    for time_point in time_intervals:
       
        interval_count = sum(1 for _, complete_time in order_completion_times 
                           if complete_time <= time_point and complete_time > time_point - 30)
        
        cumulative_count = sum(1 for _, complete_time in order_completion_times 
                             if complete_time <= time_point)
        
        completion_data['intervals'].append(time_point)
        completion_data['cumulative'].append(cumulative_count)
        completion_data['interval_counts'].append(interval_count)
        
        if interval_count > 0:
            print(f"⏰ {time_point:.0f}분 시점: 주문 {interval_count}개 완료 (누적: {cumulative_count}개)")
    
    return completion_data, order_completion_times


# 🍎 누적 주문 완료량 & 30분 구간별 주문 완료량 시각화 


def visualize_order_completion(completion_data, total_orders, title_prefix=""):
    """
    주문 완료량 시각화
    """
    print(f"\n📈 {title_prefix}시각화 생성 중...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    intervals = completion_data['intervals']
    cumulative = completion_data['cumulative']
    interval_counts = completion_data['interval_counts']
    
    # 1. 누적 주문 완료량
    ax1.plot(intervals, cumulative, marker='o', linewidth=3, markersize=8, color='#2E86AB')
    ax1.fill_between(intervals, cumulative, alpha=0.3, color='#2E86AB')
    ax1.axhline(y=total_orders, color='red', linestyle='--', alpha=0.7, 
               label=f'전체 주문수: {total_orders}개')
    
    ax1.set_title(f'📈 {title_prefix}누적 주문 완료량', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('시간 (분)', fontsize=12)
    ax1.set_ylabel('누적 완료 주문 수', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 완료율 표시
    if cumulative:
        final_completion_rate = (cumulative[-1] / total_orders) * 100
        ax1.text(0.02, 0.98, f'최종 완료율: {final_completion_rate:.1f}%', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top')
    
    # 2. 구간별 주문 완료량
    bars = ax2.bar(intervals, interval_counts, width=20, alpha=0.7, color='#A23B72')
    ax2.set_title(f'📊 {title_prefix}30분 구간별 주문 완료량', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('시간 (분)', fontsize=12)
    ax2.set_ylabel('구간별 완료 주문 수', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 막대 위에 숫자 표시
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 실행 함수"""
    try:
        print("="*60)
        print("🏭 전체 상품 통합 생산 스케줄링 및 주문 완료량 분석")
        print("="*60)
        
        # 🍎 1. 주문 데이터 불러옴
        print("📋 데이터 로딩 중...")
        df = pd.read_excel("생산전략_비교_분석데이터_전처리.xlsx")
        dish_quantity = df.groupby('상품명')['수량'].sum().to_dict()

        print(f"전체 주문 수: {len(df['주문번호'].unique())}개")
        print(f"전체 상품 수: {len(dish_quantity)}개")
        print(f"총 상품 수량: {sum(dish_quantity.values())}개")

        # 🍎 2. 전체 상품에 대한 병렬 생산 스케줄링
        print("\n" + "="*60)
        print("🚩 전체 상품 생산 라인 배정")
        print("="*60)

        # 병렬 생산 스케줄링

        line_schedules, line_time, makespan = asso.assign_parallel_by_workload(df, n_lines=8)

        # 🍎 3. 임베딩 및 전환시간 생성
        print("🔧 임베딩 및 전환시간 매트릭스 생성 중...")
        embedding_result = opti_vrp.create_dish_embeddings(df)
        changeover_df = opti_vrp.calculate_changeover_matrix(embedding_result, base_time=5, max_additional_time=20)

        # 🍎 4. 조리시간 + 전환시간을 모두 합산해서 계산
        line_total_time, makespan = asso.calc_line_times_with_changeover(line_schedules, dish_quantity, changeover_df)

        # 🍎 5. 각 생상라인별로 10개까지만 결과 출력
        print("\n📊 전체 상품 생산 라인 결과:")
        for i, seq in line_schedules.items():
            print(f"🚩라인 {i+1} (총 작업시간: {line_total_time[i]:.2f}분) (제품 수: {len(seq)}개)")
            if len(seq) > 0:
                print(f"   상품: {', '.join(seq[:10])}{'...' if len(seq) > 10 else ''}")
        print(f"\n최종 makespan(전체 완료시간): {makespan:.2f}분")


        # 5-1) 상품코드 매핑
        code_map = dict(zip(df['상품명'].astype(str), df['상품코드'].astype(str)))

        # 5-2) 타임라인 DF 생성
        timeline_df = make_timeline_df(
            line_schedules=line_schedules,
            dish_quantity=dish_quantity,
            changeover_df=changeover_df,
            code_map=code_map
        )

        # 5-3) 엑셀 저장
        out_path = save_timeline_excel(
            timeline_df,
            out_path="/Users/cmnss/25-sum/urop/0721민서/output.xlsx",
            only_AD=False
        )
        print(f"\n📝 상품 타임라인 저장 완료: {out_path}")

        # 🍎 6. 30분 단위로 주문 완료량 print
        print("\n" + "="*60)
        print("📊 전체 주문 완료량 추적 (30분 단위)")
        print("="*60)

        completion_data, order_times = track_order_completion_30min(
            line_schedules, dish_quantity, changeover_df, df
        )

        total_orders = len(df['주문번호'].unique())
        print(f"\n📈 전체 주문 수: {total_orders}개")

        # 7. 시각화
        visualize_order_completion(completion_data, total_orders, "전체 상품 ")

        # 8. 전체 요약 및 분석
        print("\n" + "="*60)
        print("🎯 전체 생산 분석 결과")
        print("="*60)

        print(f"📊 생산 현황:")
        print(f"  - 전체 주문 수: {total_orders}개")
        print(f"  - 전체 상품 종류: {len(dish_quantity)}개")
        print(f"  - 총 생산 수량: {sum(dish_quantity.values())}개")
        print(f"  - 사용 라인 수: {len(line_schedules)}개")
        print(f"  - 전체 완료시간: {makespan:.1f}분 ({makespan/60:.1f}시간)")
        print(f"  - 시간당 주문 처리량: {total_orders/(makespan/60):.1f}개/시간")
        print(f"  - 시간당 상품 생산량: {sum(dish_quantity.values())/(makespan/60):.1f}개/시간")

        # 라인별 효율성 분석
        print(f"\n🏭 라인별 효율성:")
        max_time = max(line_total_time.values())
        min_time = min(line_total_time.values())
        avg_time = sum(line_total_time.values()) / len(line_total_time)
        
        print(f"  - 최대 작업시간: {max_time:.1f}분 (라인 {max(line_total_time, key=line_total_time.get)+1})")
        print(f"  - 최소 작업시간: {min_time:.1f}분 (라인 {min(line_total_time, key=line_total_time.get)+1})")
        print(f"  - 평균 작업시간: {avg_time:.1f}분")
        print(f"  - 작업시간 편차: {max_time - min_time:.1f}분")
        print(f"  - 라인 효율성: {(avg_time/max_time)*100:.1f}%")

        # 완료율 분석
        if completion_data['cumulative']:
            final_completion = completion_data['cumulative'][-1]
            completion_rate = (final_completion / total_orders) * 100
            
            print(f"\n📈 완료율 분석:")
            print(f"  - 최종 완료 주문: {final_completion}개")
            print(f"  - 전체 완료율: {completion_rate:.1f}%")
            
            # 시간대별 완료율 마일스톤
            milestones = [0.25, 0.5, 0.75, 0.9]
            print(f"  - 완료 마일스톤:")
            for milestone in milestones:
                target = total_orders * milestone
                for i, cum_count in enumerate(completion_data['cumulative']):
                    if cum_count >= target:
                        time_point = completion_data['intervals'][i]
                        print(f"    * {milestone*100:.0f}% 완료: {time_point:.0f}분")
                        break

        print("\n✅ 전체 분석 완료!")
        
    except FileNotFoundError:
        print("❌ 파일을 찾을 수 없습니다. '생산전략_비교_분석데이터_전처리.xlsx' 파일이 있는지 확인해주세요.")
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {str(e)}")
        print("상세 오류 정보:")
        import traceback
        traceback.print_exc()

# 스크립트가 직접 실행될 때만 main 함수 실행
if __name__ == "__main__":
    main()