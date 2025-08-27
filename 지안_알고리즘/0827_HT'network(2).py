import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
import copy

plt.rcParams['font.family'] = 'Malgun Gothic'    # Windows
# plt.rcParams['font.family'] = 'AppleGothic'    # macOS
plt.rcParams['axes.unicode_minus'] = False

# ==================== 데이터 전처리 ====================

### 반찬별 조리시간 ########################################
def get_dish_cooking_times():
    cooking_times = {
        # 무침류 (1-3분)
        '콩나물무침': 1, '미나리무침': 2, '무생채': 2, '시금치나물 - 90g': 3,
        '새콤달콤 유채나물무침': 3, '새콤달콤 방풍나물무침': 3, '닭가슴살 두부무침': 3,
        '새콤달콤 돌나물무침': 2, '새콤달콤 오징어무침': 3, '새콤달콤 오이달래무침': 2,
        '브로콜리 두부무침 - 100g': 3, '매콤 콩나물무침': 3, '오이부추무침': 2,
        '참깨소스 시금치무침': 3, '(gs재등록) 닭가슴살 참깨무침': 3, '무말랭이무침': 3,
        '오징어무말랭이무침 - 130g': 3, '참나물무침 - 80g': 2, '연근참깨무침': 3,
        '참깨소스 버섯무침 - 100g': 3, '톳두부무침': 3, '가지무침': 3,
        '숙주나물무침 - 90g': 3, '달래김무침': 2, '새콤 꼬시래기무침': 3,
        '오이부추무침 - 100g': 2, '참깨두부무침 - 200g': 3, '새콤 오이무생채': 3,
        '새콤달콤 오징어무침 - 110g': 3, '새콤달콤 도라지무침': 3, '콩나물무침 - 90g': 2,
        '무생채 - 100g': 2, '파래김무침': 2, '무나물 - 100g': 2,
        
        # 김치/절임류 (1-3분)
        '물김치 - 350g': 2, '백김치 - 350g': 2, '양파고추 장아찌 - 150g': 2,
        '유자향 오이무피클 - 240g': 2, '깻잎 장아찌': 2, '셀러리 장아찌': 2,
        '깍두기': 3, '나박김치': 3, '총각김치': 3, '곰취 장아찌': 2, '볶음김치': 3,
        '볶음김치_대용량': 3,
        
        # 국물류 (3-5분)
        '아이들 된장국': 4, '감자국': 5, '계란국(냉동)': 3, '순한 오징어무국': 5,
        '시래기 된장국(냉동)': 5, '달래 된장찌개': 4, '근대 된장국(냉동)': 5,
        '된장찌개': 5, '동태알탕': 5, '맑은 콩나물국(냉동)': 4, '오징어 무국(냉동)': 5,
        '냉이 된장국(냉동)': 4, '한우 소고기 감자국': 5, '우리콩 강된장찌개': 5,
        '맑은 순두부찌개': 4, '계란 황태국(냉동)': 4, '오징어찌개': 5,
        '시금치 된장국(냉동)': 4, '김치콩나물국(냉동)': 5, '한우사골곰탕(냉동) - 600g': 5,
        '한우 소고기 무국(냉동) - 650g': 5, '한우 소고기 미역국(냉동) - 650g': 5,
        '맑은 동태국': 5, '콩나물 황태국(냉동)': 4, '배추 된장국(냉동)': 5,
        
        # 찌개류 (5-8분)
        '한돈 돼지김치찌개': 7, '한돈 청국장찌개': 6, '동태찌개': 6,
        '한돈 돼지돼지 김치찌개_쿠킹박스': 7, '한돈 돼지고추장찌개': 7, '알탕': 8,
        
        # 볶음류 (3-5분)
        '한우 무볶음': 4, '고추장 멸치볶음': 3, '야채 어묵볶음': 4, '느타리버섯볶음 - 90g': 3,
        '풋마늘 어묵볶음': 4, '애호박볶음': 3, '새우 애호박볶음 - 110g': 4,
        '한돈 가지볶음': 4, '들깨머위나물볶음': 3, '도라지볶음 - 80g': 3,
        '감자햄볶음': 4, '느타리버섯볶음': 3, '토마토 계란볶음': 3, '미역줄기볶음': 3,
        '건곤드레볶음': 4, '건고사리볶음 - 80g': 3, '호두 멸치볶음_대용량': 4,
        '미역줄기볶음_대용량': 4, '감자채볶음': 3, '건취나물볶음 - 80g': 3,
        '호두 멸치볶음': 4, '꼴뚜기 간장볶음': 5, '새우오이볶음': 3,
        '소고기 야채볶음_반조리': 5, '들깨시래기볶음 - 90g': 4, '보리새우 간장볶음': 4,
        '소고기 우엉볶음': 5, '한우오이볶음': 4, '건가지볶음': 3,
        '들깨고구마 줄기볶음 - 80g': 3, '한우오이볶음 - 100g': 4,
        '야채 어묵볶음 - 80g': 4, '감자채볶음 - 80g': 3, '매콤 어묵볶음': 4,
        '건피마자볶음': 3, '한우 무볶음 - 110g': 4, '감자햄볶음 - 80g': 4,
        '소고기 우엉볶음 - 80g': 5, '꽈리멸치볶음 - 60g': 3, '호두 멸치볶음 - 60g': 4,
        '미역줄기볶음 - 60g': 3, '꽈리멸치볶음_대용량': 4, '소고기 가지볶음': 5,
        '간장소스 어묵볶음': 4, '건호박볶음': 3, '고추장 멸치볶음_대용량': 4,
        '한돈 냉이 버섯볶음밥 재료': 5, '상하농원 케찹 소세지 야채볶음': 4,
        '상하농원 햄 어묵볶음': 4,
        
        # 제육/고기볶음류 (3-5분)
        '한돈 매콤 제육볶음_반조리 - 500g': 5, '주꾸미 한돈 제육볶음_반조리': 5,
        '한돈 김치두루치기_반조리': 5, '한돈 미나리 고추장불고기_반조리': 5,
        '한돈 대파 제육볶음_반조리': 5, '주꾸미 야채볶음_반조리': 5,
        '오징어 야채볶음_반조리': 4, '간장 오리 주물럭_반조리': 5,
        '한돈 콩나물불고기_반조리': 5, '한돈 간장 콩나물불고기_반조리': 5,
        '한돈 간장불고기_반조리': 4, '오리 주물럭_반조리': 5, '한돈 된장불고기_반조리': 5,
        '한돈 간장불고기_쿠킹박스': 4, '한돈 매콤 제육볶음_쿠킹박스': 5,
        '한돈 풋마늘 두루치기_반조리': 5,
        
        # 조림류 (3-5분)
        '메추리알 간장조림': 5, '소고기 장조림 - 180g': 5, '두부조림': 4,
        '알감자조림': 4, '케찹두부조림': 4, '매콤 닭가슴살 장조림': 5,
        '메추리알 간장조림_대용량': 5, '깻잎조림_대용량': 3, '소고기 장조림_대용량': 5,
        '한입 두부간장조림': 4, '검은콩조림': 5, '한입 두부간장조림 - 110g': 4,
        '표고버섯조림': 5, '케찹두부조림 - 120g': 4, '계란 간장조림': 4,
        '명란 장조림': 3, '국내산 땅콩조림': 5, '깻잎조림': 3, '간장 감자조림': 5,
        '마늘쫑 간장조림': 3, '메추리알 간장조림 - 110g': 5, '한우 장조림': 5,
        '우엉조림 - 100g': 5, '유자견과류조림': 4, '한돈 매콤 안심장조림': 5,
        '촉촉 간장무조림': 5, '미니새송이버섯조림': 4, '간장 코다리조림': 5,
        '매콤 코다리조림': 5, '고등어무조림': 5,
        
        # 찜류 (5-8분)
        '꽈리고추찜': 5, '야채 계란찜': 5, '계란찜': 5, '매운돼지갈비찜': 8,
        '순두부 계란찜': 5, '안동찜닭_반조리': 8,
        
        # 전류 (3-5분)
        '소고기육전과 파채': 5, '참치깻잎전': 5, '냉이전 - 140g': 4, '매생이전': 4,
        '동태전': 5, '달콤 옥수수전 - 140g': 4, '반달 계란전': 4, '매콤김치전': 5,
        
        # 구이류 (3-5분)
        '간편화덕 고등어 순살구이': 4, '간편화덕 삼치 순살구이': 4,
        '간편화덕 연어 순살구이': 5, '한돈 너비아니(냉동)': 4, '오븐치킨_반조리(냉동)': 5,
        '한돈등심 치즈가스_반조리(냉동)': 4, '통등심 수제돈가스_반조리(냉동)': 4,
        
        # 밥/주먹밥류 (1-3분)
        '한돈 주먹밥': 3, '계란 두부소보로 주먹밥': 3, '멸치 주먹밥': 3,
        '참치마요 주먹밥': 3, '한우 주먹밥': 3, '햇반 발아현미밥': 2, '햇반 백미': 2,
        
        # 덮밥류 (1-3분)
        '한돈 토마토 덮밥': 3, '아이들 두부덮밥': 3, '사색 소보로 덮밥': 3,
        
        # 볶음밥 재료 (3-5분)
        '새우 볶음밥 재료': 4, '닭갈비 볶음밥 재료': 4, '냉이 새우볶음밥 재료': 4,
        '상하농원 소세지 볶음밥 재료': 4, '감자볶음밥 재료': 4, '한돈 불고기볶음밥 재료': 4,
        
        # 비빔밥류 (1-3분)
        '꼬막비빔밥': 3,
        
        # 떡볶이류 (3-5분)
        '궁중 떡볶이_반조리 - 520g': 5, '우리쌀로 만든 기름떡볶이_반조리': 4,
        
        # 불고기/전골류 (5-8분)
        '뚝배기 불고기_반조리': 7, '서울식 불고기버섯전골_반조리': 8,
        '한우 파육개장(냉동)': 8, '소불고기_반조리 - 400g': 7,
        '한우 소불고기_반조리': 8, '모둠버섯 불고기_반조리': 6,
        
        # 계란말이 (3-5분)
        '계란말이': 3, '야채계란말이': 3,
        
        # 장류/소스 (1분)
        '달래장': 1, '맛쌈장': 1, '양배추와 맛쌈장': 1, '사랑담은 돈가스소스': 1,
        
        # 기타 특수 요리 (3분)
        '옥수수 버무리': 3, '상하농원 햄 메추리알 케찹볶음': 3, '무나물': 3,
        '수제비_요리놀이터': 3, '봄나물 샐러드': 3, '황태 보푸리': 3,
        '가지강정_대용량': 3, '가지강정': 3, '낙지젓': 3, '영양과채사라다': 3,
        '시래기 된장지짐': 3, '잡채 - 450g': 3, '해물잡채': 3,
        '바른 간장참치 - 130g': 3, '골뱅이무침_반조리': 3, '참깨소스 버섯무침': 3,
        '한우 계란소보로': 3, '꼬마김밥_요리놀이터': 3, '요리놀이터 꼬꼬마 김발': 3,
        '오징어젓': 3, '황기 닭곰탕(냉동)': 3, '불고기 잡채': 3,
        '우엉잡채 - 80g': 3, '만두속재료_요리놀이터': 3,
    }
    return cooking_times

# 총 조리시간 계산
def get_cooking_time(dish_name, quantity=1):
    """
    특정 반찬의 총 조리시간 계산 (기본시간 + 수량비례시간)
    
    Parameters:
    -----------
    dish_name : str - 반찬명
    quantity : int - 수량
    
    Returns:
    --------
    float : 총 조리시간 (분)
    """
    cooking_times = get_dish_cooking_times()
    
    if dish_name in cooking_times:
        base_time = cooking_times[dish_name]
    else:
        base_time = 3  # 기본값 : 조리시간을 못찾을 시 3분으로 지정
        print(f"⚠️ '{dish_name}' 조리시간을 찾을 수 없어 기본값 {base_time}분 사용")
    
    # 수량 비례 시간 추가 (개당 0.01분)
    unit_time = 0.01
    total_time = base_time + (quantity * unit_time)
    
    return total_time


### 주문 & 상품: 각각 딕셔너리 생성 ######################################
def process_orders_data(orders_df):
    orders = {}    #{'주문번호': ~~}
    products_info = {}  #{'상품명': {'code':상품코드, 'order_ids':[주문번호,..], 'quantity':하루 주문량}}
    
    # 주문별로 그룹핑
    grouped = orders_df.groupby('주문번호')
    
    for order_id, group in grouped:
        order_products = []
        quantities = {}
        product_codes = {}
        
        for _, row in group.iterrows():
            product_code = row['상품코드']
            product_name = row['상품명']
            quantity = row['수량']
            
            order_products.append(product_name)
            product_codes[product_name] = product_code
            quantities[product_name] = quantity
            
            if product_name not in products_info:
                products_info[product_name] = {'code': None,'order_ids': [], 'total_quantity': 0}
            products_info[product_name]['code'] = product_code
            products_info[product_name]['order_ids'].append(str(order_id))
            products_info[product_name]['total_quantity'] += quantity
        
        orders[str(order_id)] = {
            'products': order_products,
            'quantities': quantities,
            'order_date': group.iloc[0]['주문일자'],
            'product_codes': product_codes
        }
    
    return orders, products_info


### 상품 간 네트워크 >>> 상품 특성 파악 #####################################
def build_product_connections(orders, products_info):
    product_connections = defaultdict(int)         #{'상품A/상품B': weight}
    product_max_connections = {}                   #{'상품A': weight_max}
    product_total_connections = defaultdict(int)   #{'상품A': weight_sum}
    
    # 상품 간 연결 관계 계산
    for order_id, order_data in orders.items():
        product_names = order_data['products']      
        
        for i in range(len(product_names)):
            for j in range(i + 1, len(product_names)):
                product1, product2 = sorted([product_names[i], product_names[j]])
                key = f"{product1}|{product2}"
                product_connections[key] += 1
    
    # 각 상품의 최대 연결 횟수 및 총 연결 수 계산    
    all_products = list(products_info.keys())
    for product in all_products:
        max_count = 0
        total_count = 0
        
        for connection_key, count in product_connections.items():
            product1, product2 = connection_key.split('|')
            if product1 == product or product2 == product:
                max_count = max(max_count, count)
                total_count += count
        
        product_max_connections[product] = max_count
        product_total_connections[product] = total_count
    
    return dict(product_connections), product_max_connections, dict(product_total_connections)

def calculate_connection_ratio(product_name, product_total_connections, products_info):
    """연결수/주문수 비율 계산"""
    total_connections = product_total_connections.get(product_name, 0)
    order_count = len(products_info[product_name]['order_ids'])
    return total_connections / order_count

def classify_products_by_connection_strength(all_products, product_max_connections):
    """연결 강도에 따라 상품 분류"""
    group_1 = [p for p in all_products if product_max_connections.get(p, 0) == 1]
    group_2 = [p for p in all_products if product_max_connections.get(p, 0) == 2]
    group_3_plus = [p for p in all_products if product_max_connections.get(p, 0) >= 3]
    
    return group_1, group_2, group_3_plus


def preprocess_all_data(orders_df):
    """모든 전처리 작업을 수행하는 통합 함수"""
    
    # 1. 기본 데이터 처리
    orders, products_info = process_orders_data(orders_df)
    all_products = list(products_info.keys())
    
    # 2. 상품 간 연결 관계 구축
    product_connections, product_max_connections, product_total_connections = build_product_connections(orders, products_info)
    
    # 3. 상품별 총 조리시간 계산
    cooking_times = {}
    for product_name, info in products_info.items():
        total_quantity = info['total_quantity']
        cooking_times[product_name] = get_cooking_time(product_name, total_quantity)
    
    # 4. 전환시간 매트릭스 읽어오기
    changeover_matrix = pd.read_csv('changeover_matrix.csv', index_col=0)
    
    # 5. 상품 분류
    group_1, group_2, group_3_plus = classify_products_by_connection_strength(all_products, product_max_connections)
    
    # 전처리된 데이터 반환
    return {
        'orders': orders,
        'products_info': products_info,
        'product_connections': product_connections,
        'product_max_connections': product_max_connections,
        'product_total_connections': product_total_connections,
        'cooking_times': cooking_times,
        'changeover_matrix': changeover_matrix,
        'all_products': all_products,
        'group_1': group_1,
        'group_2': group_2,
        'group_3_plus': group_3_plus
    }

# ==================== 최적화 엔진 ====================
# 기본 라인 수 설정
DEFAULT_NUM_LINES = 8

def get_changeover_time(product1, product2, changeover_matrix):
    """두 상품 간 전환시간 계산"""
    if (product1 in changeover_matrix.index and 
        product2 in changeover_matrix.columns):
        return changeover_matrix.loc[product1, product2]
    else:
        return 4  # 기본 전환시간

def assign_low_connection_products(g_products, target_lines, product_total_connections, products_info): #, cooking_times, changeover_matrix, time_limit_minutes):
    """낮은/중간 연결성 상품들, 비율 최소인 상품 지그재그 배치""" 
    line_assignments = {line: [] for line in target_lines}
    
    # calculate_connection_ratio 기준으로 오름차순 정렬
    sorted_products = sorted(g_products, 
                           key=lambda p: calculate_connection_ratio(p, product_total_connections, products_info))
    
    # 지그재그 배치를 위한 변수
    line_index = 0
    direction = 1  # 1: 순방향, -1: 역방향
    
    for product in sorted_products:
        current_line = target_lines[line_index]
        
        # 상품을 현재 라인에 배치
        line_assignments[current_line].append(product)
        
        # 다음 라인 인덱스 계산 (지그재그)
        if line_index == (len(target_lines)-1):
            direction = -1
        elif line_index == 0:
            direction = 1
        line_index += direction
            
    return line_assignments

def assign_high_connection_products(g_products, target_lines, product_connections, product_total_connections, products_info):#, cooking_times, changeover_matrix, time_limit_minutes, num_lines=DEFAULT_NUM_LINES):
    """높은 연결성 상품들, 네트워크 기반, 비율 최소인 상품 지그재그 배치"""   
    line_assignments = {line: [] for line in target_lines}
    
    # 예외 처리; 상품이 없거나 라인이 없으면 빈 배치 반환
    if not g_products or not target_lines:
        return line_assignments
    
    remaining_products = g_products.copy()
    
    # 이웃 노드들(상품)
    def get_product_neighbors(product, product_connections):
        neighbors = []
        for connection_key in product_connections.keys():
            product1, product2 = connection_key.split('|')
            if product1 == product:
                neighbors.append(product2)
            elif product2 == product:
                neighbors.append(product1)
        return neighbors
    
    # 상품 배치에 따른 업데이트
    def assign_product_to_line(product, line):
        line_assignments[line].append(product)
        remaining_products.remove(product)
    
    line_index = 0
    direction = 1
    current_product = None
    while remaining_products and line_index < len(target_lines):
        current_line = target_lines[line_index]
        
        if current_product is None: # 가장 처음 생산하거나 이웃이 없으면 ~~
            # remaining_products에서 가장 비율이 낮은 상품 선택
            current_product = remaining_products[0]
        
        # 현재 상품을 현재 라인에 배치
        assign_product_to_line(current_product, current_line)
        
        # 다음 라인에 배치할 상품 찾기; 현재 상품의 이웃 중 가장 비율이 낮은 상품
        neighbors = get_product_neighbors(current_product, product_connections)
        neighbors_in_remaining = [p for p in neighbors if p in remaining_products]
        if neighbors_in_remaining:
            # 이웃 중 가장 비율이 낮은 상품
            current_product = min(neighbors_in_remaining, 
                                key=lambda p: calculate_connection_ratio(p, product_total_connections, products_info))
        else:
            # 이웃이 없으면 다시 remaining_products[0]에서 시작
            current_product = None
        
        # 다음 라인 인덱스 계산 (순환)
        if line_index == (len(target_lines)-1):
            direction = -1
        elif line_index == 0:
            direction = 1
        line_index += direction

    return line_assignments

def create_initial_solution(preprocessed_data, num_lines=DEFAULT_NUM_LINES):
    """개선된 초기해 생성 (라인 수 파라미터 적용)"""
    
    # 전처리된 데이터 추출
    product_connections = preprocessed_data['product_connections']
    product_total_connections = preprocessed_data['product_total_connections']
    products_info = preprocessed_data['products_info']
    #cooking_times = preprocessed_data['cooking_times']
    #changeover_matrix = preprocessed_data['changeover_matrix']
    
    group_1 = preprocessed_data['group_1']
    group_2 = preprocessed_data['group_2']
    group_3_plus = preprocessed_data['group_3_plus']
    
    # 모든 라인 초기화
    solution = {f'line{i}': [] for i in range(1, num_lines + 1)}
    
    # 그룹 1을 라인 1, 2에 배치
    #if sorted_group_1 and num_lines >= 2:
    group1_assignment = assign_low_connection_products(group_1, ['line1','line2'], product_total_connections, products_info)
    solution.update(group1_assignment)
    
    # 그룹 2를 라인 3, 4에 배치
    #if sorted_group_2 and num_lines >= 4:
    group2_assignment = assign_low_connection_products(group_2, ['line3','line4'], product_total_connections, products_info)
    solution.update(group2_assignment)
    
    # 그룹 3+를 라인 5 ~ 8에 배치
    #if sorted_group_3_plus and num_lines >= 5:
    group3_assignment = assign_high_connection_products(
        group_3_plus, ['line5','line6','line7','line8'], product_connections, 
        product_total_connections, products_info
    )
    solution.update(group3_assignment)
    
    #print(f"🔍 solution 내용: {solution}")
    return solution


### 시간 계산 ##################################################
def calculate_line_schedule(l_products, cooking_times, changeover_matrix):
    """라인의 상품별 시작시간과 완료시간 계산"""
    
    if not l_products:
        return {}
    
    schedule = {}
    current_time = 0
    
    for i, product in enumerate(l_products):
        start_time = current_time
        
        # 전환시간 추가 (첫 번째 상품은 제외)
        if i > 0:
            prev_product = l_products[i-1]
            changeover_time = get_changeover_time(prev_product, product, changeover_matrix)
            start_time += changeover_time
        
        # 조리시간
        cooking_time = cooking_times.get(product, 3)
        completion_time = start_time + cooking_time
        
        schedule[product] = {
            'start_time': start_time,
            'completion_time': completion_time,
            'cooking_time': cooking_time
        }
        
        current_time = completion_time
    
    return schedule

def calculate_order_and_line_completion_times(orders, solution, cooking_times, changeover_matrix):
    """각 주문의 완료시간과 각 라인의 최종 완료시간을 함께 계산"""
    # 각 라인의 스케줄 계산
    line_schedules = {}
    line_completion_times = {}
    for line_id, l_products in solution.items():
        line_schedules[line_id] = calculate_line_schedule(l_products, cooking_times, changeover_matrix)
        
        # 라인 완료시간 = 해당 라인의 마지막 제품 완료시간
        last_product = l_products[-1]
        line_completion_times[line_id] = line_schedules[line_id][last_product]['completion_time']
        
        
    # 각 주문의 완료시간 계산
    order_completion_times = {}
    
    for order_id, order_data in orders.items():
        order_products = order_data['products']
        product_completion_times = []
        
        for product in order_products:
            # 상품이 어느 라인에 배치되었는지 찾기
            product_line = None
            for line_id, products in solution.items():
                if product in products:
                    product_line = line_id
                    break
            
            if product_line and product in line_schedules[product_line]:
                completion_time = line_schedules[product_line][product]['completion_time']
                product_completion_times.append(completion_time)
        
        # 주문 완료시간 = 해당 주문의 모든 상품 중 가장 늦게 완료되는 시간
        if product_completion_times:
            order_completion_times[order_id] = max(product_completion_times)
        else:
            order_completion_times[order_id] = 0
    
    return order_completion_times, line_completion_times

def calculate_completion_interval_variance(completion_times_list):
    """주문 완료시간 간격의 분산 계산 (균등 간격을 위한 밸런싱)"""
    
    if len(completion_times_list) <= 1:
        return 0
    
    # 완료시간들을 정렬
    sorted_times = sorted(completion_times_list)
    
    # 연속된 완료시간들 간의 간격 계산
    intervals = []
    for i in range(1, len(sorted_times)):
        interval = sorted_times[i] - sorted_times[i-1]
        intervals.append(interval)
    
    # 간격들의 분산 계산
    interval_variance = np.var(intervals) if len(intervals) > 1 else 0
    
    return interval_variance

def calculate_line_balance_variance(line_completion_times):
    """라인별 완료시간의 분산 계산 (라인 밸런싱 지표)"""
    if not line_completion_times:
        return 0
    
    # 실제로 사용된 라인들만 고려 (완료시간이 0보다 큰 라인들)
    active_line_times = [time for time in line_completion_times.values() if time > 0]
    
    if len(active_line_times) <= 1:
        return 0  # 라인이 1개 이하면 분산 없음
    
    # 표준편차 계산 (분산의 제곱근)
    std_dev = np.std(active_line_times)
    return std_dev



def calculate_objective_function(orders, solution, cooking_times, changeover_matrix, 
                               order_priorities=None):
    """다중 목표 목적함수 계산"""
    
    # 주문 완료시간들 계산
    order_completion_times, line_completion_times = calculate_order_and_line_completion_times(orders, solution, cooking_times, changeover_matrix)
    completion_times_list = list(order_completion_times.values())
    
    if not completion_times_list:
        return float('inf')
    
    # 1. 최대 주문 완료시간 (마지막 주문 출고시각)
    max_order_completion = max(completion_times_list)
    
    # 2. 주문 완료시간 간격의 분산 (균등 간격 밸런싱)
    order_completion_interval_variance = calculate_completion_interval_variance(completion_times_list)
    
    '''
    # 3. 총 전환시간
    total_changeover_time = 0
    for line_id, l_products in solution.items():
        if len(l_products) > 1:
            for i in range(1, len(l_products)):
                prev_product = l_products[i-1]
                current_product = l_products[i]
                changeover_time = get_changeover_time(prev_product, current_product, changeover_matrix)
                total_changeover_time += changeover_time
    
    # 4. 라인별 완료시간 밸런싱
    line_balance_variance = calculate_line_balance_variance(line_completion_times)
    
    # 5. 제약조건 위반 페널티
    time_limit_penalty = 0
    line_schedules = {}
    for line_id, products in solution.items():
        line_schedules[line_id] = calculate_line_schedule(products, cooking_times, changeover_matrix)
        if products:
            last_product = products[-1]
            line_completion_time = line_schedules[line_id][last_product]['completion_time']
            if line_completion_time > time_limit_minutes:
                time_limit_penalty += (line_completion_time - time_limit_minutes) * 10  # 큰 페널티
    '''
    
    # 가중합 목적함수
    objective = (
        0.4 * max_order_completion +                # 전체 완료시간 최소화
        0.6 * order_completion_interval_variance    # 주문 완료시간 간격 균등화  
        #0.1 * total_changeover_time                # 전환시간 최소화; 라인 내 순서 최적화에서만 고려!
        #0.3 * line_balance_variance +              # 라인별 완료시간 밸런싱
        #0.1 * time_limit_penalty                   # 제약조건 준수
    )
    
    return objective

def local_optimization(preprocessed_data, solution, num_lines=DEFAULT_NUM_LINES):
    """지역 최적화를 통한 해 개선 - 스마트 라인 밸런싱 중심"""
    
    orders = preprocessed_data['orders']
    cooking_times = preprocessed_data['cooking_times']
    changeover_matrix = preprocessed_data['changeover_matrix']
    #order_priorities = preprocessed_data['order_priorities']
    
    current_solution = copy.deepcopy(solution)
    current_objective = calculate_objective_function(orders, current_solution, cooking_times, changeover_matrix)#, order_priorities)
    
    # 라인 내, 전환시간만 고려, 최대 5번 swap
    def optimize_line_order_by_changeover(line_id, products, max_iterations=5):
        if len(products) <= 1:
            return products
        current_order = products.copy()
        best_order = current_order.copy()
        
        def calculate_total_changeover(order):
            total_changeover_time = 0
            for i in range(1, len(order)):
                prev_product = order[i-1]
                current_product = order[i]
                total_changeover_time += get_changeover_time(prev_product, current_product, changeover_matrix)
            return total_changeover_time
        best_changeover_time = calculate_total_changeover(best_order)
        
        for iteration in range(max_iterations):
            improved = False
            for i in range(len(current_order)):
                for j in range(i + 2, len(current_order)):
                    # i와 j 위치의 상품들을 교환
                    new_order = current_order.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    
                    new_changeover_time = calculate_total_changeover(new_order)
                    
                    if new_changeover_time < best_changeover_time:
                        best_order = new_order
                        best_changeover_time = new_changeover_time
                        improved = True
            
            current_order = best_order.copy()
            if not improved:
                break
        
        return best_order
    
    # *** 상품 이동 함수 ***
    def try_product_move(source_line, target_line, product_idx, target_position=None):
        """상품을 다른 라인의 지정된 위치로 이동"""
        temp_solution = copy.deepcopy(current_solution)
        
        source_products = temp_solution[source_line]
        if product_idx >= len(source_products):
            return None, float('inf')
        
        # 상품 제거
        product = source_products.pop(product_idx)
        
        # 상품 삽입 (위치 지정되지 않으면 끝에 추가)
        if target_position is None:
            temp_solution[target_line].append(product)
        else:
            target_position = min(target_position, len(temp_solution[target_line]))
            temp_solution[target_line].insert(target_position, product)
        
        # 목적함수 계산
        temp_objective = calculate_objective_function(orders, temp_solution, cooking_times, changeover_matrix) #order_priorities
        
        return temp_solution, temp_objective
    
    # *** 영향받은 라인들의 순서 재최적화 (전환시간 기준) ***
    def reoptimize_affected_lines(affected_lines):
        """상품 이동 후 영향받은 라인들의 순서를 전환시간 기준으로 재최적화"""
        for line_id in affected_lines:
            if line_id in current_solution and len(current_solution[line_id]) > 1:
                # 해당 라인의 순서 최적화 (전환시간 기준)
                optimized_order = optimize_line_order_by_changeover(line_id, current_solution[line_id])
                
                # 순서가 실제로 바뀌었는지 확인
                if optimized_order != current_solution[line_id]:
                    current_solution[line_id] = optimized_order
                    print(f"🔄 {line_id} 라인 순서 재최적화 완료 (전환시간 기준)")
    
    # *** 라인별 완료시간 계산 ***
    def get_line_completion_times(solution):
        """각 라인의 완료시간 계산"""
        _, line_completion_times = calculate_order_and_line_completion_times(
            orders, solution, cooking_times, changeover_matrix
        )
        return line_completion_times
    
    # *** 가장 부담되는 상품 선택 ***
    def select_most_burden_product(line_products):
        """라인에서 전환시간+조리시간+전환시간이 가장 긴 상품의 인덱스 반환"""
        if not line_products:
            return None
        
        burden_scores = {}
        
        for i, product in enumerate(line_products):
            burden = cooking_times.get(product, 3)  # 조리시간
            
            # 이전 전환시간
            if i > 0:
                burden += get_changeover_time(line_products[i-1], product, changeover_matrix)
            
            # 다음 전환시간  
            if i < len(line_products) - 1:
                burden += get_changeover_time(product, line_products[i+1], changeover_matrix)
                
            burden_scores[i] = burden
        
        # 부담도가 가장 높은 상품의 인덱스 반환
        return max(burden_scores.keys(), key=lambda x: burden_scores[x])
    
    # *** 스마트 라인 밸런싱 ***
    def balance_lines_smartly():
        """가장 긴 라인에서 가장 부담되는 상품을 가장 짧은 라인으로 이동"""
        line_completion_times = get_line_completion_times(current_solution)
        
        # 실제 사용된 라인들만 고려
        active_lines = {line_id: time for line_id, time in line_completion_times.items() 
                       if time > 0 and current_solution[line_id]}
        
        if len(active_lines) < 2:
            return None, float('inf'), None, None
        
        # 가장 긴 라인과 가장 짧은 라인 찾기
        longest_line = max(active_lines.keys(), key=lambda x: active_lines[x])
        shortest_line = min(active_lines.keys(), key=lambda x: active_lines[x])
        
        if longest_line == shortest_line:
            return None, float('inf'), None, None
        
        # 가장 긴 라인에서 가장 부담되는 상품 선택
        long_products = current_solution[longest_line]
        burden_product_idx = select_most_burden_product(long_products)
        
        if burden_product_idx is None:
            return None, float('inf'), None, None
        
        # 가장 짧은 라인의 (처음/중간/끝)에 삽입 시도
        short_products = current_solution[shortest_line]
        insert_positions = [0, len(short_products)//2, len(short_products)]
        if len(short_products) > 3:
            insert_positions.extend([1, len(short_products)-1])
        
        best_solution = None
        best_objective = float('inf')
        
        for insert_pos in insert_positions:
            temp_solution, temp_objective = try_product_move(
                longest_line, shortest_line, burden_product_idx, insert_pos
            )
            
            if temp_solution and temp_objective < best_objective:
                best_solution = temp_solution
                best_objective = temp_objective
        
        return best_solution, best_objective, longest_line, shortest_line
    
    print(f"🚀 지역 최적화 시작 (초기 목적함수: {current_objective:.2f})")
    
    # 1. 초기 라인 내 순서 최적화 (전환시간 기준)
    print("📋 1단계: 초기 라인 내 순서 최적화 (전환시간 기준)")
    for line_id, products in current_solution.items():
        if len(products) > 1:
            original_order = products.copy()
            optimized_order = optimize_line_order_by_changeover(line_id, products)
            
            if optimized_order != original_order:
                current_solution[line_id] = optimized_order
                print(f"    ✅ {line_id} 라인 순서 최적화 완료")
    
    # 목적함수 재계산
    current_objective = calculate_objective_function(orders, current_solution, cooking_times, changeover_matrix)#, order_priorities)
    print(f"📊 1단계 완료 후 목적함수: {current_objective:.2f}")
    
    # 2. 스마트 라인 밸런싱과 라인 내 재최적화 반복
    print("\n⚖️ 2단계: 스마트 라인 밸런싱 + 라인 내 재최적화 반복")
    
    iteration = 0
    max_iterations = 20
    
    while iteration < max_iterations:
        iteration += 1
        
        # 스마트 라인 밸런싱 시도
        temp_solution, temp_objective, source_line, target_line = balance_lines_smartly()
        
        if temp_solution and temp_objective < current_objective:
            # 이동된 상품 정보 찾기
            moved_product = None
            if source_line and target_line:
                original_products = set(current_solution[source_line])
                new_products = set(temp_solution[source_line])
                moved_products = original_products - new_products
                if moved_products:
                    moved_product = list(moved_products)[0]
            
            # 솔루션 업데이트
            current_solution = temp_solution
            old_objective = current_objective
            current_objective = temp_objective
            
            print(f"  🔄 반복 {iteration}: 라인 밸런싱 개선 {old_objective:.2f} → {current_objective:.2f}")
            if moved_product and source_line and target_line:
                print(f"    📦 '{moved_product}' {source_line} → {target_line}")
            
            # 영향받은 라인들의 순서 재최적화 (전환시간 기준)
            if source_line and target_line:
                affected_lines = [source_line, target_line]
                reoptimize_affected_lines(affected_lines)
                
                # 재최적화 후 목적함수 재계산
                current_objective = calculate_objective_function(orders, current_solution, cooking_times, changeover_matrix)#, order_priorities)
                print(f"    📊 재최적화 후 목적함수: {current_objective:.2f}")
        else:
            # 개선이 없으면 종료
            print(f"  ⭐ 반복 {iteration}에서 개선 없음 - 최적화 완료")
            break
    
    print(f"\n🎯 지역 최적화 완료: 최종 목적함수 {current_objective:.2f}")
    return current_solution

def analyze_solution(preprocessed_data, solution, num_lines=DEFAULT_NUM_LINES):
    """최적화 결과 종합 분석"""
    
    orders = preprocessed_data['orders']
    cooking_times = preprocessed_data['cooking_times']
    changeover_matrix = preprocessed_data['changeover_matrix']
    
    # 기본 통계
    total_products = sum(len(products) for products in solution.values())
    total_orders = len(orders)
    
    # 라인별 분석
    line_schedules = {}
    for line_id, products in solution.items():
        if products:
            line_schedule = calculate_line_schedule(products, cooking_times, changeover_matrix)
            line_schedules[line_id] = line_schedule
    
    # 주문 완료시간 분석
    order_completion_times, _ = calculate_order_and_line_completion_times(orders, solution, cooking_times, changeover_matrix)
    completion_times_list = list(order_completion_times.values())
    
    interval_variance = 0
    if completion_times_list:
        interval_variance = calculate_completion_interval_variance(completion_times_list)
    
    # 목적함수 값
    objective_value = calculate_objective_function(orders, solution, cooking_times, changeover_matrix)#, order_priorities)
    
    
    return {
        'objective_value': objective_value,
        'order_completion_times': order_completion_times,
        'line_schedules': line_schedules,
        'interval_variance': interval_variance,
        'total_products': total_products,
        'total_orders': total_orders
    }

def optimize_production_schedule(preprocessed_data, num_lines=DEFAULT_NUM_LINES):
    """
    통합 생산 스케줄링 최적화 실행
    
    Parameters:
    -----------
    preprocessed_data : dict
        전처리된 데이터
    time_limit_hours : int
        제한 시간 (시간 단위)
    num_lines : int
        사용할 라인 수
    
    Returns:
    --------
    dict : 최적화 결과
    """
    
    # 1. 초기해 생성
    initial_solution = create_initial_solution(preprocessed_data, num_lines)
    
    initial_objective = calculate_objective_function(
        preprocessed_data['orders'], 
        initial_solution, 
        preprocessed_data['cooking_times'], 
        preprocessed_data['changeover_matrix'] 
    )
    
    # 2. 지역 최적화
    optimized_solution = local_optimization(preprocessed_data, initial_solution, num_lines)
    
    # 3. 결과 분석
    analysis_result = analyze_solution(preprocessed_data, optimized_solution, num_lines)
    
    # 개선율 계산
    improvement = ((initial_objective - analysis_result['objective_value']) / initial_objective * 100)
    
    return {
        'initial_solution': initial_solution,
        'optimized_solution': optimized_solution,
        'analysis': analysis_result,
        'initial_objective': initial_objective,
        'final_objective': analysis_result['objective_value'],
        'improvement_rate': improvement,
        'num_lines': num_lines
    }


def get_final_completion_time(solution, preprocessed_data):
    """전체 주문 완료시점 계산"""
    orders = preprocessed_data['orders']
    cooking_times = preprocessed_data['cooking_times']
    changeover_matrix = preprocessed_data['changeover_matrix']
    
    order_completion_times, _ = calculate_order_and_line_completion_times(orders, solution, cooking_times, changeover_matrix)
    return max(order_completion_times.values()) if order_completion_times else 0


def export_solution_to_excel_by_lines(solution, preprocessed_data, filename="production_schedule_by_lines.xlsx"):
    """라인별로 별도 시트에 생산 스케줄 저장"""
    products_info = preprocessed_data['products_info']
    cooking_times = preprocessed_data['cooking_times']
    changeover_matrix = preprocessed_data['changeover_matrix']
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for line_id, products in solution.items():
            if not products:
                continue
                
            line_schedule = calculate_line_schedule(products, cooking_times, changeover_matrix)
            
            # 라인별 스케줄 데이터 생성
            line_data = []
            for i, product_name in enumerate(products):
                if product_name in line_schedule:
                    schedule_info = line_schedule[product_name]
                    
                    # 상품코드 찾기
                    product_code = products_info.get(product_name, {}).get('code', 'Unknown')
                    
                    line_data.append({
                        '순서': i + 1,
                        '상품코드': product_code,
                        '상품명': product_name,
                        '시작시간': round(schedule_info['start_time'], 1),
                        '완료시간': round(schedule_info['completion_time'], 1),
                        '조리시간': schedule_info['cooking_time']
                    })
            
            # DataFrame 생성 및 시트에 저장
            df_line = pd.DataFrame(line_data)
            sheet_name = line_id.upper()  # 'LINE1', 'LINE2', ...
            df_line.to_excel(writer, sheet_name=sheet_name, index=False)
            
        print(f"📁 라인별 스케줄이 {filename}에 저장되었습니다.")
        
def print_final_results(solution, preprocessed_data):
    """최종 결과 출력 및 시각화"""
    
    # 1. 전체 주문 완료시점 
    final_time = get_final_completion_time(solution, preprocessed_data)
    print(f"\n🎯 전체 주문 완료시점: {final_time:.1f}분 ({final_time/60:.1f}시간)")
    
    # 2. 솔루션 분석 결과 출력 추가
    print("\n📊 솔루션 분석 결과:")
    analysis = analyze_solution(preprocessed_data, solution)
    
    print(f"   • 목적함수값: {analysis['objective_value']:.2f}")
    print(f"   • 총 상품 수: {analysis['total_products']}개")
    print(f"   • 총 주문 수: {analysis['total_orders']}개")
    print(f"   • 주문 완료시간 간격 분산: {analysis['interval_variance']:.2f}")
    
    # 주문 완료시간 통계
    completion_times = list(analysis['order_completion_times'].values())
    if completion_times:
        #print(f"   • 주문 완료시간 - 최소: {min(completion_times):.1f}분, 최대: {max(completion_times):.1f}분, 평균: {np.mean(completion_times):.1f}분")
        
        sorted_times = sorted(completion_times)
        intervals = [sorted_times[i] - sorted_times[i-1] for i in range(1, len(sorted_times))]
        print(f"   • 주문 완료시간 간격 분산: {analysis['interval_variance']:.2f}")
        print(f"   • 주문 완료시간 간격 - 최소: {min(intervals):.1f}분, 최대: {max(intervals):.1f}분, 평균: {np.mean(intervals):.1f}분")
    
    # 라인별 완료시간 출력
    print("\n🏭 라인별 완료시간:")
    line_completion_times = {}
    for line_id, products in solution.items():
        if products:
            line_schedule = analysis['line_schedules'].get(line_id, {})
            if products[-1] in line_schedule:
                completion_time = line_schedule[products[-1]]['completion_time']
                line_completion_times[line_id] = completion_time
                print(f"   • {line_id.upper()}: {completion_time:.1f}분 ({len(products)}개 상품)")
    
    # 3. 라인별 엑셀 파일 생성
    print("\n📄 라인별 스케줄 엑셀 파일 생성 중...")
    export_solution_to_excel_by_lines(solution, preprocessed_data)
    
    return {
        'final_completion_time': final_time,
        'analysis_result': analysis  # 분석 결과도 함께 반환
    }

# ==================== 사용 예시 ====================
if __name__ == "__main__":
    # 1. 데이터 로드
    orders_df = pd.read_excel('zipbanchan_220401.xlsx')
    
    # 2. 전처리
    preprocessed_data = preprocess_all_data(orders_df)
    
    # 3. 최적화
    result = optimize_production_schedule(preprocessed_data, num_lines=8)
    
    # 4. 결과 확인
    print_final_results(result['optimized_solution'], preprocessed_data)





#%% 위의 결과 파일을 아래의 함수로 변환한 후 평가함수에 넣기
def load_data(file_a_path, output_path):
    """
    엑셀 파일들을 로드하고 통합 데이터를 저장하는 함수
    """
    import pandas as pd

    # 파일 A: 모든 시트(LINE1~8) 읽기
    production_data = {}
    for i in range(1, 9):
        sheet_name = f'LINE{i}'
        try:
            df = pd.read_excel(file_a_path, sheet_name=sheet_name)
            df['라인'] = i
            production_data[sheet_name] = df
        except:
            print(f"{sheet_name} 시트를 찾을 수 없습니다.")
    
    # 모든 생산 데이터를 하나로 합치기
    production_df = pd.concat(production_data.values(), ignore_index=True)
    
    # 저장
    production_df.to_excel(output_path, index=False)

    print(f"통합 데이터가 '{output_path}' 파일로 저장되었습니다.")

# 사용 예시
load_data(
    file_a_path="production_schedule_by_lines.xlsx",  # 생산스케줄.xlsx 경로, 
    output_path="production_schedule_combined.xlsx"
)