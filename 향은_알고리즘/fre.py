import pandas as pd
import math
from collections import defaultdict

# 데이터 로딩
def load_data(order_file, cooking_times_file, changeover_matrix_file):
    orders = pd.read_excel(order_file)
    cooking_times = pd.read_csv(cooking_times_file)
    changeover_matrix = pd.read_csv(changeover_matrix_file, index_col=0)
    
    # 상품별 총 수량 계산
    product_quantities = orders.groupby('상품명')['수량'].sum().to_dict()
    
    # 조리시간 매핑
    cooking_time_map = dict(zip(cooking_times['상품명'], cooking_times['조리시간(분)']))
    
    # 주문별 필요 상품
    order_requirements = defaultdict(list)
    for _, row in orders.iterrows():
        order_requirements[row['주문번호']].append(row['상품명'])
    
    # 상품별 주문 빈도 계산 (몇 개의 주문에 포함되는지)
    product_order_frequency = defaultdict(int)
    for order_products in order_requirements.values():
        for product in set(order_products):  # 같은 주문에서 중복 제거
            product_order_frequency[product] += 1
    
    return product_quantities, cooking_time_map, changeover_matrix, order_requirements, product_order_frequency

# 조리시간 계산
def get_cooking_time(product, quantity, cooking_time_map):
    base_time = cooking_time_map.get(product, 3)
    return base_time + (quantity * 0.01)

# 전환시간 계산
def get_changeover_time(from_product, to_product, changeover_matrix):
    if from_product == to_product:
        return 0
    try:
        return changeover_matrix.loc[from_product, to_product]
    except:
        return 0

# 병렬 생산 최적화
def optimize_parallel_production(product_quantities, cooking_time_map, changeover_matrix, order_requirements, product_order_frequency):
    # 1. 주문 빈도 기준으로 상품 우선순위 설정
    products_by_frequency = sorted(product_quantities.keys(), 
                                 key=lambda x: product_order_frequency[x], 
                                 reverse=True)
    
    # 8개 라인 초기화
    lines = [[] for _ in range(8)]
    line_times = [0] * 8
    product_end_times = {}
    
    # 각 상품이 어느 주문에 포함되는지 역매핑
    product_to_orders = defaultdict(list)
    for order_num, products in order_requirements.items():
        for product in products:
            product_to_orders[product].append(order_num)
    
    # 각 주문의 상품들이 어느 라인에 배치되었는지 추적
    order_assigned_lines = defaultdict(set)
    
    for product in products_by_frequency:
        quantity = product_quantities[product]
        cooking_time = get_cooking_time(product, quantity, cooking_time_map)
        
        # 이 상품을 포함한 주문들
        related_orders = product_to_orders[product]
        
        # 같은 주문의 다른 상품들이 이미 배치된 라인들 찾기
        used_lines = set()
        for order_num in related_orders:
            used_lines.update(order_assigned_lines[order_num])
        
        # 사용되지 않은 라인 중에서 가장 빠른 라인 선택
        available_lines = [i for i in range(8) if i not in used_lines]
        
        if available_lines:
            # 사용 가능한 라인 중 가장 빠른 라인
            line_idx = min(available_lines, key=lambda x: line_times[x])
        else:
            # 모든 라인이 사용 중이면 가장 빠른 라인 선택
            line_idx = line_times.index(min(line_times))
        
        # 전환시간 계산
        changeover_time = 0
        if lines[line_idx]:
            last_product = lines[line_idx][-1][0]
            changeover_time = get_changeover_time(last_product, product, changeover_matrix)
        
        # 시간 계산
        start_time = line_times[line_idx] + changeover_time
        end_time = start_time + cooking_time
        
        # 라인에 추가
        lines[line_idx].append((product, quantity, cooking_time, changeover_time, start_time, end_time))
        line_times[line_idx] = end_time
        product_end_times[product] = end_time
        
        # 이 상품의 관련 주문들에 라인 정보 업데이트
        for order_num in related_orders:
            order_assigned_lines[order_num].add(line_idx)
    
    return lines, product_end_times, max(line_times)

# 주문 완료시간 계산
def calculate_order_completion(order_requirements, product_end_times):
    order_completion_times = {}
    for order_num, products in order_requirements.items():
        completion_times = [product_end_times.get(product, 0) for product in products]
        order_completion_times[order_num] = max(completion_times)
    return order_completion_times

# 결과 출력
def print_results(lines, order_completion_times, total_time):
    print("=== 8개 생산라인 최적화 결과 ===\n")
    
    # 라인별 생산순서
    print("📋 8개 라인별 생산순서:")
    for i, line in enumerate(lines):
        print(f"\n🏭 라인 {i+1} (총 {len(line)}개 작업)")
        for j, (product, quantity, cook_time, change_time, _, _) in enumerate(line[:5]):
            print(f"   {j+1}. {product} ({quantity}개) - 조리: {cook_time:.1f}분, 전환: {change_time:.1f}분")
        if len(line) > 5:
            print(f"   ... 외 {len(line) - 5}개 작업")
    
    # 각 라인 완료시간
    print(f"\n⏰ 각 라인별 완료시간:")
    for i, line in enumerate(lines):
        line_total = line[-1][5] if line else 0  # 마지막 작업의 end_time
        print(f"   라인 {i+1}: {line_total:.0f}분 ({line_total/60:.1f}시간)")
    
    # 전체 완료시간
    print(f"\n🎯 전체 주문완료시간: {total_time:.0f}분 ({total_time/60:.1f}시간)")
    
    # 30분 단위 누적완전커버주문량
    print(f"\n📦 30분 단위 누적완전커버주문량:")
    cumulative_orders = 0
    for time_point in range(30, int(total_time) + 30, 30):
        completed_this_interval = sum(
            1 for completion_time in order_completion_times.values()
            if time_point - 30 < completion_time <= time_point
        )
        cumulative_orders += completed_this_interval
        print(f"⏰ {time_point}분 시점: 주문 {completed_this_interval}개 완료 (누적: {cumulative_orders}개)")

# 메인 실행 함수
def run_parallel_optimization(order_file, cooking_times_file, changeover_matrix_file):
    # 데이터 로딩
    product_quantities, cooking_time_map, changeover_matrix, order_requirements, product_order_frequency = load_data(
        order_file, cooking_times_file, changeover_matrix_file
    )
    
    # 병렬 생산 최적화 실행
    lines, product_end_times, total_time = optimize_parallel_production(
        product_quantities, cooking_time_map, changeover_matrix, order_requirements, product_order_frequency
    )
    
    # 주문 완료시간 계산
    order_completion_times = calculate_order_completion(order_requirements, product_end_times)
    
    # 결과 출력
    print_results(lines, order_completion_times, total_time)
    
    return lines, order_completion_times

# 실행
if __name__ == "__main__":
    order_file = "생산전략_비교_분석데이터_0401_전처리.xlsx"
    cooking_times_file = "cooking_times.csv"
    changeover_matrix_file = "changeover_matrix.csv"
    
    lines, completion_times = run_parallel_optimization(order_file, cooking_times_file, changeover_matrix_file)
