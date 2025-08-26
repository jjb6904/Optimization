import pandas as pd
import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
import opti_vrp


# 1. 주문별 basket 생성 함수
def make_baskets_from_orders(df):
    """
    주문 데이터에서 주문번호별 상품 리스트 생성
    """
    baskets = df.groupby('주문번호')['상품명'].apply(list).tolist()
    return baskets

# 2. 동시주문 행렬 생성 함수 = 동시주문 몇번 들어갔는지
def make_cooccurrence_matrix(baskets, dish_list):
    """
    상품별 동시주문(연관성) 행렬 생성
    (dish_list 순서대로 square matrix)
    """
    idx = {dish: i for i, dish in enumerate(dish_list)}
    mat = np.zeros((len(dish_list), len(dish_list)), dtype=int)
    for basket in baskets:
        for i in range(len(basket)):
            for j in range(i+1, len(basket)):
                a, b = basket[i], basket[j]
                if a in idx and b in idx:
                    mat[idx[a], idx[b]] += 1
                    mat[idx[b], idx[a]] += 1
    return pd.DataFrame(mat, index=dish_list, columns=dish_list)

# 3. 클러스터링 함수
def cluster_dishes(co_mat, n_clusters=8):
    """
    동시주문 행렬을 기반으로 상품을 n_clusters개로 군집화
    """
    # 행렬을 유사도(거리)로 변환
    dist = 1 - (co_mat / co_mat.max().max())  # 간단화: 동시주문 횟수 정규화(0~1), 유사도→거리
    # AgglomerativeClustering (hierarchical)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(dist)
    groups = {i: [] for i in range(n_clusters)}
    for dish, label in zip(co_mat.index, labels):
        groups[label].append(dish)
    return groups

# 4. 생산계획표 출력 함수
def print_production_clusters(groups):
    """
    8개 라인별 생산 그룹 출력
    """
    print("=== 8개 생산라인(그룹)별 생산 리스트 ===")
    for line, dishes in groups.items():
        print(f"라인 {line+1} : {', '.join(dishes)}")

# 5. 전체 워크플로우 함수
def production_line_clustering(df, n_lines):
    """
    주문 데이터에서 8개 라인별 생산계획(상품 그룹) 도출
    """
    baskets = make_baskets_from_orders(df)
    dish_list = sorted(df['상품명'].unique())
    co_mat = make_cooccurrence_matrix(baskets, dish_list)
    groups = cluster_dishes(co_mat, n_clusters=n_lines)
    #print_production_clusters(groups)
    return groups



def assign_parallel_slots_balanced(df, n_lines, return_makespan=False):

    # 상품 목록, 연관성 행렬 생성
    dish_list = sorted(df['상품명'].unique())
    # 주문
    baskets = make_baskets_from_orders(df)
    # 상품별 동시주문(연관성) 행ㅕㄹ 생성
    co_mat = make_cooccurrence_matrix(baskets, dish_list)
    # 각 상품별 총 수량
    dish_quantity = df.groupby('상품명')['수량'].sum().to_dict()
    assigned = set()
    slots = []
    # 라인당 들어갈 상품 수
    num_slots = int(np.ceil(len(dish_list) / n_lines))
    for _ in range(num_slots):
        remain = [d for d in dish_list if d not in assigned]
        if not remain:
            break
        # 연관성 높은 상품 뽑아서 seed로 선정
        score = co_mat.loc[remain, remain].sum(axis=1)
        seed = score.idxmax()
        # seed와 동시주문 많은 순서대로 후보 리스트
        candidates = co_mat.loc[seed, remain].sort_values(ascending=False).index.tolist()
        slot = []
        slot_times = []
        # candidate에서 미배정 상품을 line 수(8) 만큼 병렬 추가
        for cand in candidates:
            if cand not in assigned:
                q = dish_quantity.get(cand, 1)
                time = opti_vrp.get_cooking_time(cand, q)
                slot.append(cand)
                slot_times.append(time)
            if len(slot) >= n_lines:
                break
        # 예시: slot_times 합이 너무 큰 경우 조정 등 응용 가능
        for dish in slot:
            assigned.add(dish)
        slots.append(slot)

    # 라인별로 재구성
    line_schedules = {i: [] for i in range(n_lines)}
    for t, slot in enumerate(slots):
        for i, dish in enumerate(slot):
            line_schedules[i].append(dish)


    # ⬇️ 라인별 총 작업시간(누적) 및 makespan 추가 계산
    dish_quantity = df.groupby('상품명')['수량'].sum().to_dict()
    line_total_time = {}
    for i, seq in line_schedules.items():
        total_time = sum(opti_vrp.get_cooking_time(d, dish_quantity.get(d,1)) for d in seq)
        line_total_time[i] = total_time
    makespan = max(line_total_time.values())  # 가장 늦게 끝나는 라인의 시간
    if return_makespan:
        return line_schedules, line_total_time, makespan
    else:
        return line_schedules



def assign_parallel_by_workload(df, n_lines):
    """
    동시주문 연관성 기반 우선순위 → 작업량(조리시간) 균등하게 동적 배정
    """
    import opti_vrp

    # (1) data load
    dish_list = sorted(df['상품명'].unique()) # 생산해야할 모든 반찬 리스트
    baskets = make_baskets_from_orders(df)  # 각 주문별 반찬 정보 (장바구니 리스트)
    co_mat = make_cooccurrence_matrix(baskets, dish_list) # 상품별 동시주문 행렬 (두 상품이 한 주문에 같이 들어온 횟수)
    dish_quantity = df.groupby('상품명')['수량'].sum().to_dict() # 각 반찬별 수량

    # (2) 연관성 seed부터 우선순위 리스트(order) 생성
    remain = set(dish_list) # 남은 반찬
    order = [] # 순서 저장할 빈 리스트
    while remain:
        if not order:
            # 첫 seed: 동시주문 합이 최대인 상품
            score = co_mat.sum(axis=1)
            seed = score.idxmax()
        else:
            # 배정된 반찬과 연관성이 가장 높은 남은 반찬 셀렉
            score = co_mat.loc[list(remain), order].sum(axis=1)
            seed = score.idxmax()
        order.append(seed)
        remain.remove(seed)

    # (3) 각 상품을 "가장 작업량이 적은 라인"에 동적으로 배정
    line_schedules = {i: [] for i in range(n_lines)} # 각 라인별 반찬 리스트
    line_time = {i: 0 for i in range(n_lines)} # 각 라인별 배정된 작업의 총 시간
    last_dish = {i: None for i in range(n_lines)}  # 각 라인에 마지막으로 넣은 반찬


    # order에서 반찬 하나씩 꺼내서
    for dish in order:
        # 해당 dish의 수량 q 계산
        q = dish_quantity.get(dish, 1)
        # 해당 dish 완성하는데 걸리는 총 조리시간 t 계산
        t = opti_vrp.get_cooking_time(dish, q)
        # 지금까지 작업 시간이 가장 작은 라인 idx 구함
        idx = min(line_time, key=line_time.get)
        # 해당 idx에 dish 배정
        line_schedules[idx].append(dish)
        # 해당 idx의 time, dish 정보 업데이트
        line_time[idx] += t
        last_dish[idx] = dish

    # 결과 반환 (각 라인별 할당된 반찬 리스트 / 각 라인 총 작업시간 / 그 중 max)
    return line_schedules, line_time, max(line_time.values())


def calc_line_times_with_changeover(line_schedules, dish_quantity, changeover_df):
    """
    라인별 생산순서, 수량, 전환시간 기준으로
    [조리시간 + 전환시간] 누적 작업량 계산 (각 라인별)
    """
    import opti_vrp
    line_total_time = {}
    for i, seq in line_schedules.items():
        total = 0
        prev = None
        for dish in seq:
            # 1) 조리시간(수량 반영)
            total += opti_vrp.get_cooking_time(dish, dish_quantity.get(dish,1))
            # 2) 전환시간 (이전 반찬→현재 반찬)
            if prev is not None:
                # changeover_df.loc[이전, 현재]
                total += changeover_df.loc[prev, dish]
            prev = dish
        line_total_time[i] = total
    makespan = max(line_total_time.values())
    return line_total_time, makespan
