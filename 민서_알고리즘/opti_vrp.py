# 벡터 임베딩 수행(함수 생성 -> 실행)
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer # 한국어 지원 임베딩모델
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
import seaborn as sns


def create_dish_embeddings(df, dish_column='상품명'):
    """
    DataFrame의 반찬 이름을 Sentence Transformers로 벡터 임베딩
    
    Parameters:
    -----------
    df : pandas.DataFrame
        반찬 주문 데이터가 담긴 DataFrame
    dish_column : str
        반찬 이름이 들어있는 컬럼명 (기본값: '\상품명')
    
    Returns:
    --------
    dict : {
        'dish_names': list,           # 고유한 반찬 이름들
        'embeddings': numpy.ndarray,  # 각 반찬의 벡터 임베딩 (n_dishes x embedding_dim)
        'embedding_dim': int,         # 임베딩 차원 수
        'model': SentenceTransformer  # 사용된 모델
    }
    """
    
    print("Sentence Transformers 모델 로딩 중...")
    # 한국어 지원 모델 로드
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    # 고유한 반찬 이름들 추출
    unique_dishes = df[dish_column].unique().tolist()
    print(f"총 {len(unique_dishes)}개의 고유한 반찬 발견")
    
    # 반찬 이름들을 벡터로 임베딩
    print("벡터 임베딩 생성 중...")
    embeddings = model.encode(unique_dishes, show_progress_bar=True)
    
    print(f"임베딩 완료! 차원: {embeddings.shape}")
    
    return {
    'dish_names': unique_dishes,
    'embeddings': embeddings,
    'embedding_dim': embeddings.shape[1],
    'model': model}


# 벡터간 거리 계산하여 반찬간 전환에 걸리는 시간으로 변환해줌 : 기본시간 5분 및 스케일값(코사인값에 곱해줄) 20 임의지정 -> 추후 변환도 가능함
def calculate_changeover_matrix(embedding_result, base_time, max_additional_time):
    """
    벡터 임베딩을 기반으로 전환 시간 매트릭스 계산
    
    Parameters:
    -----------
    embedding_result : dict
        create_dish_embeddings 함수의 결과
    base_time : int
        최소 전환 시간 (분)
    max_additional_time : int
        최대 추가 전환 시간 (분)
    
    Returns:
    --------
    pandas.DataFrame : 전환 시간 매트릭스
    """
    
    dish_names = embedding_result['dish_names']
    embeddings = embedding_result['embeddings']
    
    print("전환 시간 매트릭스 계산 중...")
    
    # 코사인 거리 계산 (0~2 범위)
    cosine_dist_matrix = cosine_distances(embeddings)
    
    # 거리를 전환 시간으로 변환 (base_time ~ base_time + max_additional_time)
    changeover_matrix = base_time + (cosine_dist_matrix * max_additional_time)
    
    # 대각선 (같은 반찬) 요소는 0으로 설정
    np.fill_diagonal(changeover_matrix, 0)
    
    # DataFrame으로 변환
    changeover_df = pd.DataFrame(
        changeover_matrix, 
        index=dish_names, 
        columns=dish_names
    )
    
    print(f"전환 시간 범위: {changeover_matrix.min():.1f}분 ~ {changeover_matrix.max():.1f}분")
    
    return changeover_df

# 각 반찬 별 조리시간 정의 : 4월 1일자 주문 248개의 반찬종류 별 조리시간 임의설정
import pandas as pd
# 반찬종류 별 조리시간 정의 함수
def get_dish_cooking_times():
    """
    반찬별 조리시간 데이터 (주석의 범위에 맞게 수정됨)
    각 카테고리의 주석 범위를 준수하도록 조리시간을 조정
    """
    cooking_times = {
        # 무침류 (1-3분)
        '콩나물무침': 1,
        '미나리무침': 2,
        '무생채': 2,
        '시금치나물 - 90g': 3,
        '새콤달콤 유채나물무침': 3,
        '새콤달콤 방풍나물무침': 3,
        '닭가슴살 두부무침': 3,
        '새콤달콤 돌나물무침': 2,
        '새콤달콤 오징어무침': 3,
        '새콤달콤 오이달래무침': 2,
        '브로콜리 두부무침 - 100g': 3,
        '매콤 콩나물무침': 3,
        '오이부추무침': 2,
        '참깨소스 시금치무침': 3,
        '(gs재등록) 닭가슴살 참깨무침': 3,
        '무말랭이무침': 3,
        '오징어무말랭이무침 - 130g': 3,
        '참나물무침 - 80g': 2,
        '연근참깨무침': 3,
        '참깨소스 버섯무침 - 100g': 3,
        '톳두부무침': 3,
        '가지무침': 3,
        '숙주나물무침 - 90g': 3,
        '달래김무침': 2,
        '새콤 꼬시래기무침': 3,
        '오이부추무침 - 100g': 2,
        '참깨두부무침 - 200g': 3,
        '새콤 오이무생채': 3,
        '새콤달콤 오징어무침 - 110g': 3,
        '새콤달콤 도라지무침': 3,
        '콩나물무침 - 90g': 2,
        '무생채 - 100g': 2,
        '파래김무침': 2,
        '무나물 - 100g' : 2,
        
        # 김치/절임류 (1-3분)
        '물김치 - 350g': 2,
        '백김치 - 350g': 2,
        '양파고추 장아찌 - 150g': 2,
        '유자향 오이무피클 - 240g': 2,
        '깻잎 장아찌': 2,
        '셀러리 장아찌': 2,
        '깍두기': 3,
        '나박김치': 3,
        '총각김치': 3,
        '곰취 장아찌': 2,
        '볶음김치': 3,
        '볶음김치_대용량': 3,
        
        # 국물류 (3-5분)
        '아이들 된장국': 4,
        '감자국': 5,
        '계란국(냉동)': 3,
        '순한 오징어무국': 5,
        '시래기 된장국(냉동)': 5,
        '달래 된장찌개': 4,
        '근대 된장국(냉동)': 5,
        '된장찌개': 5,
        '동태알탕': 5,
        '맑은 콩나물국(냉동)': 4,
        '오징어 무국(냉동)': 5,
        '냉이 된장국(냉동)': 4,
        '한우 소고기 감자국': 5,
        '우리콩 강된장찌개': 5,
        '맑은 순두부찌개': 4,
        '계란 황태국(냉동)': 4,
        '오징어찌개': 5,
        '시금치 된장국(냉동)': 4,
        '김치콩나물국(냉동)': 5,
        '한우사골곰탕(냉동) - 600g': 5,
        '한우 소고기 무국(냉동) - 650g': 5,
        '한우 소고기 미역국(냉동) - 650g': 5,
        '맑은 동태국': 5,
        '콩나물 황태국(냉동)': 4,
        '배추 된장국(냉동)': 5,
        
        # 찌개류 (5-8분)
        '한돈 돼지김치찌개': 7,
        '한돈 청국장찌개': 6,
        '동태찌개': 6,
        '한돈 돼지돼지 김치찌개_쿠킹박스': 7,
        '한돈 돼지고추장찌개': 7,
        '알탕': 8,
        
        # 볶음류 (3-5분)
        '한우 무볶음': 4,
        '고추장 멸치볶음': 3,
        '야채 어묵볶음': 4,
        '느타리버섯볶음 - 90g': 3,
        '풋마늘 어묵볶음': 4,
        '애호박볶음': 3,
        '새우 애호박볶음 - 110g': 4,
        '한돈 가지볶음': 4,
        '들깨머위나물볶음': 3,
        '도라지볶음 - 80g': 3,
        '감자햄볶음': 4,
        '느타리버섯볶음': 3,
        '토마토 계란볶음': 3,
        '미역줄기볶음': 3,
        '건곤드레볶음': 4,
        '건고사리볶음 - 80g': 3,
        '호두 멸치볶음_대용량': 4,
        '미역줄기볶음_대용량': 4,
        '감자채볶음': 3,
        '건취나물볶음 - 80g': 3,
        '호두 멸치볶음': 4,
        '꼴뚜기 간장볶음': 5,
        '새우오이볶음': 3,
        '소고기 야채볶음_반조리': 5,
        '들깨시래기볶음 - 90g': 4,
        '보리새우 간장볶음': 4,
        '소고기 우엉볶음': 5,
        '한우오이볶음': 4,
        '건가지볶음': 3,
        '들깨고구마 줄기볶음 - 80g': 3,
        '한우오이볶음 - 100g': 4,
        '야채 어묵볶음 - 80g': 4,
        '감자채볶음 - 80g': 3,
        '매콤 어묵볶음': 4,
        '건피마자볶음': 3,
        '한우 무볶음 - 110g': 4,
        '감자햄볶음 - 80g': 4,
        '소고기 우엉볶음 - 80g': 5,
        '꽈리멸치볶음 - 60g': 3,
        '호두 멸치볶음 - 60g': 4,
        '미역줄기볶음 - 60g': 3,
        '꽈리멸치볶음_대용량': 4,
        '소고기 가지볶음': 5,
        '간장소스 어묵볶음': 4,
        '건호박볶음': 3,
        '고추장 멸치볶음_대용량': 4,
        '한돈 냉이 버섯볶음밥 재료': 5,
        '상하농원 케찹 소세지 야채볶음': 4,
        '상하농원 햄 어묵볶음' : 4,
        
        # 제육/고기볶음류 (3-5분)
        '한돈 매콤 제육볶음_반조리 - 500g': 5,
        '주꾸미 한돈 제육볶음_반조리': 5,
        '한돈 김치두루치기_반조리': 5,
        '한돈 미나리 고추장불고기_반조리': 5,
        '한돈 대파 제육볶음_반조리': 5,
        '주꾸미 야채볶음_반조리': 5,
        '오징어 야채볶음_반조리': 4,
        '간장 오리 주물럭_반조리': 5,
        '한돈 콩나물불고기_반조리': 5,
        '한돈 간장 콩나물불고기_반조리': 5,
        '한돈 간장불고기_반조리': 4,
        '오리 주물럭_반조리': 5,
        '한돈 된장불고기_반조리': 5,
        '한돈 간장불고기_쿠킹박스': 4,
        '한돈 매콤 제육볶음_쿠킹박스': 5,
        '한돈 풋마늘 두루치기_반조리': 5,
        
        # 조림류 (3-5분)
        '메추리알 간장조림': 5,
        '소고기 장조림 - 180g': 5,
        '두부조림': 4,
        '알감자조림': 4,
        '케찹두부조림': 4,
        '매콤 닭가슴살 장조림': 5,
        '메추리알 간장조림_대용량': 5,
        '깻잎조림_대용량': 3,
        '소고기 장조림_대용량': 5,
        '한입 두부간장조림': 4,
        '검은콩조림': 5,
        '한입 두부간장조림 - 110g': 4,
        '표고버섯조림': 5,
        '케찹두부조림 - 120g': 4,
        '계란 간장조림': 4,
        '명란 장조림': 3,
        '국내산 땅콩조림': 5,
        '깻잎조림': 3,
        '간장 감자조림': 5,
        '마늘쫑 간장조림': 3,
        '메추리알 간장조림 - 110g': 5,
        '한우 장조림': 5,
        '우엉조림 - 100g': 5,
        '유자견과류조림': 4,
        '한돈 매콤 안심장조림': 5,
        '촉촉 간장무조림': 5,
        '미니새송이버섯조림': 4,
        '간장 코다리조림': 5,
        '매콤 코다리조림': 5,
        '고등어무조림': 5,
        
        # 찜류 (5-8분)
        '꽈리고추찜': 5,
        '야채 계란찜': 5,
        '계란찜': 5,
        '매운돼지갈비찜': 8,
        '순두부 계란찜': 5,
        '안동찜닭_반조리': 8,
        
        # 전류 (3-5분)
        '소고기육전과 파채': 5,
        '참치깻잎전': 5,
        '냉이전 - 140g': 4,
        '매생이전': 4,
        '동태전': 5,
        '달콤 옥수수전 - 140g': 4,
        '반달 계란전': 4,
        '매콤김치전': 5,
        
        # 구이류 (3-5분)
        '간편화덕 고등어 순살구이': 4,
        '간편화덕 삼치 순살구이': 4,
        '간편화덕 연어 순살구이': 5,
        '한돈 너비아니(냉동)': 4,
        '오븐치킨_반조리(냉동)': 5,
        '한돈등심 치즈가스_반조리(냉동)': 4,
        '통등심 수제돈가스_반조리(냉동)': 4,
        
        # 밥/주먹밥류 (1-3분)
        '한돈 주먹밥': 3,
        '계란 두부소보로 주먹밥': 3,
        '멸치 주먹밥': 3,
        '참치마요 주먹밥': 3,
        '한우 주먹밥': 3,
        '햇반 발아현미밥': 2,
        '햇반 백미': 2,
        
        # 덮밥류 (1-3분)
        '한돈 토마토 덮밥': 3,
        '아이들 두부덮밥': 3,
        '사색 소보로 덮밥': 3,
        
        # 볶음밥 재료 (3-5분)
        '새우 볶음밥 재료': 4,
        '닭갈비 볶음밥 재료': 4,
        '냉이 새우볶음밥 재료': 4,
        '상하농원 소세지 볶음밥 재료': 4,
        '감자볶음밥 재료': 4,
        '한돈 불고기볶음밥 재료': 4,
        
        # 비빔밥류 (1-3분)
        '꼬막비빔밥': 3,
        
        # 떡볶이류 (3-5분)
        '궁중 떡볶이_반조리 - 520g': 5,
        '우리쌀로 만든 기름떡볶이_반조리': 4,
        
        # 불고기/전골류 (5-8분)
        '뚝배기 불고기_반조리': 7,
        '서울식 불고기버섯전골_반조리': 8,
        '한우 파육개장(냉동)': 8,
        '소불고기_반조리 - 400g': 7,
        '한우 소불고기_반조리': 8,
        '모둠버섯 불고기_반조리': 6,
        
        # 계란말이 (3-5분)
        '계란말이': 3,
        '야채계란말이': 3,
        
        # 장류/소스 (1분)
        '달래장': 1,
        '맛쌈장': 1,
        '양배추와 맛쌈장': 1,
        '사랑담은 돈가스소스': 1,
        
        # 기타 특수 요리 (3분)
        '옥수수 버무리': 3,
        '상하농원 햄 메추리알 케찹볶음': 3,
        '무나물': 3,
        '수제비_요리놀이터': 3,
        '봄나물 샐러드': 3,
        '황태 보푸리': 3,
        '가지강정_대용량': 3,
        '가지강정': 3,
        '낙지젓': 3,
        '영양과채사라다': 3,
        '시래기 된장지짐': 3,
        '잡채 - 450g': 3,
        '해물잡채': 3,
        '바른 간장참치 - 130g': 3,
        '골뱅이무침_반조리': 3,
        '참깨소스 버섯무침': 3,
        '한우 계란소보로': 3,
        '꼬마김밥_요리놀이터': 3,
        '요리놀이터 꼬꼬마 김발': 3,
        '오징어젓': 3,
        '황기 닭곰탕(냉동)': 3,
        '불고기 잡채': 3,
        '우엉잡채 - 80g': 3,
        '만두속재료_요리놀이터': 3,
    }
    
    return cooking_times


# 정의한 조리시간 데이터를 dataframe으로 변환하는 함수
def create_cooking_time_dataframe():
    """조리시간 데이터를 DataFrame으로 변환"""
    cooking_times = get_dish_cooking_times()
    
    df = pd.DataFrame([
        {'반찬명': dish, '기본조리시간(분)': time}
        for dish, time in cooking_times.items()
    ])
    
    return df.sort_values('기본조리시간(분)')


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



# 실제 vrp최적화 해보기
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_dish_production_vrp(embedding_result, changeover_matrix, orders_df, 
                             num_lines=4, max_time=240):
    """
    Multiple Depot VRP로 반찬 생산 최적화
    
    Parameters:
    -----------
    embedding_result : dict - 벡터 임베딩 결과
    changeover_matrix : pd.DataFrame - 전환 시간 매트릭스  
    orders_df : pd.DataFrame - 주문 데이터
    num_lines : int - 생산라인 수 (기본 4개)
    max_time : int - 최대 조리 시간 (기본 240분)
    """
    
    # 1. 데이터 준비
    print("=== 데이터 준비 중 ===")
    
    # 주문된 반찬별 총 수량 계산
    dish_demands = orders_df.groupby('상품명')['수량'].sum().to_dict() # 딕셔너리 형태로 변환해야함
    ordered_dishes = list(dish_demands.keys()) # dish_demands의 키(반찬이름)를 ordered_dish로 저장
    num_dishes = len(ordered_dishes)           # 총 반찬개수 구하기
    
    print(f"주문된 반찬: {num_dishes}개")
    print(f"총 생산량: {sum(dish_demands.values())}개")
    
    # 2. 조리 시간 계산 (기존 함수 사용)
    # get_cooking_time() 함수를 import해서 사용
    
    # 각 반찬의 조리 시간 계산
    cooking_times = {}
    for dish in ordered_dishes:
        quantity = dish_demands[dish]
        cooking_times[dish] = get_cooking_time(dish, quantity) # 위에서 만들어 놓은 cookingtime계산하는 함수를 가져와서 사용
    
    print(f"조리 시간 범위: {min(cooking_times.values()):.1f}분 ~ {max(cooking_times.values()):.1f}분") # 최소 ~ 최대 조리시간을 미리 출력
    
    # 3. 거리 매트릭스 생성
    print("\n=== 거리 매트릭스 생성 중 ===")
    
    # VRP 노드 구성: [depot0, depot1, depot2, depot3, dish1, dish2, ..., dishN]
    num_depots = num_lines # vrp의 총 depot개수 지정(여기서는 생산라인 4개가 depot 4개)
    num_nodes = num_depots + num_dishes # 모든 노드의 개수 : depot + 총 반찬수
    
    # 거리 매트릭스 초기화
    distance_matrix = np.zeros((num_nodes, num_nodes), dtype=int) # 거리 메트릭스(num_nodes x num_nodeds)에 0값을 채움
    
    # depot에서 반찬으로 가는 거리 = 0 (시작 비용 없음 : 여기서는 실제 차량이 움직이는것이 아닌 생산라인이기 때문에, 첫번째 재료를 만드는 과정에서는 전환시간이 안든다는 가정)
    for depot in range(num_depots): # 총 생산라인만큼 반복(4회, 0 ~ 3)
        for dish_idx in range(num_dishes): # 총 반찬개수만큼 반복(248개, 0 ~ 247)
            node_idx = num_depots + dish_idx # vrp의 노드 인덱스를 구함
            distance_matrix[depot][node_idx] = 0 # 첫번째로 생산하는 반찬까지 가는 거리행렬값을 0으로 지정
    
    # 반찬에서 depot으로 돌아가는 거리 = 0 (종료 비용 없음)
    for dish_idx in range(num_dishes):
        node_idx = num_depots + dish_idx
        for depot in range(num_depots):
            distance_matrix[node_idx][depot] = 0 # 마지막으로 생산하는 반찬에서 depot까지 가는 거리행렬값을 0으로 지정
    
    # 반찬 간 전환 시간 설정
    for i in range(num_dishes):
        for j in range(num_dishes):
            dish_i = ordered_dishes[i]
            dish_j = ordered_dishes[j]
            
            if dish_i in changeover_matrix.index and dish_j in changeover_matrix.columns:
                changeover_time = int(changeover_matrix.loc[dish_i, dish_j])
            else:
                changeover_time = 3  # 기본 전환 시간 3분 지정
                
            node_i = num_depots + i
            node_j = num_depots + j
            distance_matrix[node_i][node_j] = changeover_time # 거리 행렬에 전환시간을 저장
    
    
    # 4. VRP 모델 생성
    print("\n=== VRP 모델 생성 중 ===")
    
    depot_starts = [0, 1, 2, 3]  # 각 라인이 시작하는 depot
    depot_ends = [0, 1, 2, 3]    # 각 라인이 끝나는 depot (같은 곳)

    
    # 라우팅 매니저 생성
    manager = pywrapcp.RoutingIndexManager(
        num_nodes,           # 총 노드 수
        num_lines,           # 차량(생산라인) 수  
        depot_starts,        # 각 차량의 시작 depot
        depot_ends           # 각 차량의 종료 depot(동일)
    )
    
    # 라우팅 모델 생성
    routing = pywrapcp.RoutingModel(manager)
    
    # 5. 거리 콜백 함수
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # 6. 시간 제약 (Capacity 제약)
    print("\n=== 제약 조건 설정 중 ===")
    
    def time_callback(from_index):
        """각 노드에서의 시간 소모량"""
        from_node = manager.IndexToNode(from_index)
        
        # depot이면 시간 소모 없음
        if from_node < num_depots:
            return 0
            
        # 반찬이면 조리 시간 소모
        dish_idx = from_node - num_depots
        dish_name = ordered_dishes[dish_idx]
        return int(cooking_times[dish_name]) # 정수변환
    
    time_callback_index = routing.RegisterUnaryTransitCallback(time_callback)
    
    # 각 라인별 시간 제약 추가
    routing.AddDimensionWithVehicleCapacity(
        time_callback_index,
        0,              # slack (여유시간)
        [max_time] * num_lines,  # 각 차량(라인)의 최대 시간 : [240, 240, 240, 240]
        True,           # start cumul을 0으로 고정 : 출발시간 기준(0)으로 누적시간 계산
        'Time'          # 시간제약의 dimension이름
    )
    
    # 7. 모든 반찬이 정확히 한 번씩 방문되도록 제약
    for dish_idx in range(num_dishes):
        node_idx = num_depots + dish_idx
        routing.AddDisjunction([manager.NodeToIndex(node_idx)], 1000000)  # 큰 페널티
    
    '''
    # 8. 목적함수 설정 (Makespan 최소화)
    time_dimension = routing.GetDimensionOrDie('Time')

    
    # 각 라인의 완료 시간을 최소화
    for line in range(num_lines):
        routing.AddVariableMinimizedByFinalizer(
            time_dimension.CumulVar(routing.End(line))
        )
    '''

    # 8. 목적함수 설정 (Makespan 최소화)
    time_dimension = routing.GetDimensionOrDie('Time')

    # 각 생산라인(차량)의 종료 시간 CumulVar 변수 수집
    end_time_vars = [time_dimension.CumulVar(routing.End(line)) for line in range(num_lines)]

    # 종료 시간 중 최댓값을 나타내는 변수 정의
    max_end_time = routing.AddVariableMinimizedByFinalizer(
        routing.solver().Max(end_time_vars)
    )

    # 종료 시간 변수들도 모두 최적화 대상에 포함 (Makespan 정밀도 향상)
    for var in end_time_vars:
        routing.AddVariableMinimizedByFinalizer(var)


    # 9. 솔버 설정
    print("\n=== 최적화 시작 ===")
    
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(30)  # 30초 제한
    
    # 10. 최적화 실행
    solution = routing.SolveWithParameters(search_parameters)
    
    # 11. 결과 출력(아래의 print_solution이라는 함수를 실행함)
    if solution:
        print_solution(manager, routing, solution, ordered_dishes, cooking_times, num_depots)
        return manager, routing, solution
    else:
        print("❌ 해를 찾을 수 없습니다!")
        return None, None, None



# 결과를 print해주는 함수(윗쪽 함수에 포함됨)
def print_solution(manager, routing, solution, ordered_dishes, cooking_times, num_depots):
    """최적화 결과 출력"""
    
    print("\n" + "="*50)
    print("🎯 최적화 결과")
    print("="*50)
    
    total_time = 0
    max_line_time = 0
    
    for line_id in range(routing.vehicles()):
        index = routing.Start(line_id)
        plan_output = f'생산라인 {line_id + 1}: '
        route_time = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            
            # 반찬 노드인 경우
            if node >= num_depots:
                dish_idx = node - num_depots
                dish_name = ordered_dishes[dish_idx]
                cooking_time = cooking_times[dish_name]
                
                plan_output += f'{dish_name}({cooking_time:.1f}분) -> '
                route_time += cooking_time
                
                # 다음 노드로의 전환 시간 추가
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                if not routing.IsEnd(index):
                    route_time += routing.GetArcCostForVehicle(previous_index, index, line_id)
            else:
                index = solution.Value(routing.NextVar(index))
        
        plan_output += '완료'
        print(f'{plan_output}')
        print(f'⏱️  총 소요시간: {route_time:.1f}분')
        print('-' * 50)
        
        max_line_time = max(max_line_time, route_time)
    
    print(f"\n🏆 전체 완료 시간 (Makespan): {max_line_time:.1f}분")
    print(f"⏰ 제한 시간 대비: {max_line_time/240*100:.1f}%")
    
    if max_line_time <= 240:
        print("✅ 시간 제약 만족!")
    else:
        print("⚠️  시간 제약 초과!")



# 사용해주는 함수
def run_vrp_optimization(embedding_result, changeover_matrix, orders_df):
    """VRP 최적화 실행"""
    
    print("🚀 반찬 생산 최적화를 시작합니다!")
    
    manager, routing, solution = solve_dish_production_vrp(
        embedding_result=embedding_result,
        changeover_matrix=changeover_matrix,
        orders_df=orders_df
    )
    
    return manager, routing, solution


def run_full_optimization(file_path):
    """전체 최적화 프로세스 실행"""
    import pandas as pd
    df = pd.read_excel(file_path)
    embedding_result = create_dish_embeddings(df, dish_column='상품명')
    changeover_df = calculate_changeover_matrix(embedding_result, base_time=2, max_additional_time=2)
    return run_vrp_optimization(embedding_result, changeover_df, df)

if __name__ == "__main__":
    test_file ="/Users/cmnss/25-sum/urop/아카이브/생산전략_비교_분석데이터_전처리.xlsx"
    run_full_optimization(test_file)