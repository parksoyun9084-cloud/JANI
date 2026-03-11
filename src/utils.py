import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import auc, roc_curve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def roc_graph(model, y_train, y_train_pred, y_test, y_test_pred):
    fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_pred)
    fpr_te, tpr_te, _ = roc_curve(y_test, y_test_pred)

    auc_tr = auc(fpr_tr, tpr_tr)
    auc_te = auc(fpr_te, tpr_te)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr_tr, tpr_tr, label=f"Train ROC (AUC={auc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, label=f"Test ROC (AUC={auc_te:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.title(f'{model} ROC Curve (Train vs Test)')
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.legend()
    plt.grid(True)

    plt.show()




def elbow_graph(data_df):
    inertia = []

    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_df)

        inertia.append(kmeans.inertia_)

    plt.plot(range(2, 7), inertia)

    return plt.show()




def visualize_silhouette(n_clusters_list, X_features):
    """
    다양한 클러스터 개수에 대해 KMeans 클러스터링을 수행하고,
    각 클러스터링 결과의 실루엣 계수를 시각화하는 함수.

    Parameters:
    - n_clusters_list : 클러스터 개수 리스트 (예: [2, 3, 4])
    - X_features : 클러스터링에 사용할 특성 데이터 (numpy 배열 또는 DataFrame)
    """
    n_cols = len(n_clusters_list)  # 서브플롯 열 개수

    # 서브플롯 생성 (1행, n_cols열)
    fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(4 * n_cols, 4))     # 가로가 넓어지게 figsize 설정.

    # axs가 1개일 경우 리스트 형태로 변환 (for문에서 인덱싱을 위함)
    if n_cols == 1:
        axs = [axs]     # for문 돌 때 indexing을 하는 부분이 있어서 경우의 수가 한 개여도 리스트로 변환해줌.

    for idx, n_clusters in enumerate(n_clusters_list):

        # KMeans 클러스터링 수행
        kmeans = KMeans(n_clusters=n_clusters, max_iter=500, random_state=0)
        cluster_labels = kmeans.fit_predict(X_features)

        # 전체 평균 실루엣 점수와 각 샘플의 실루엣 계수 계산
        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        # 시각화 초기 설정
        ax = axs[idx]
        ax.set_title(f'Number of Clusters: {n_clusters}\nSilhouette Score: {round(sil_avg, 3)}')
        ax.set_xlim([-0.1, 1])
        ax.set_ylim([0, len(X_features) + (n_clusters + 1) * 10]) # y축(데이터포인트 개수) + margin을 위한 패딩 설정
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.set_yticks([])  # Y축 눈금 제거
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        y_lower = 10  # 막대 시작 위치 초기화

        for i in range(n_clusters):

            # 현재 클러스터에 속한 샘플들의 실루엣 계수 추출 및 정렬
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()

            # 막대 그릴 y 위치 계산
            size_cluster_i = ith_cluster_sil_values.shape[0] # 데이터샘플수.
            y_upper = y_lower + size_cluster_i

            # 색상 설정 및 막대 시각화
            color = cm.nipy_spectral(float(i) / n_clusters) # 클러스터마다 다른 색을 자동으로 지정
            # fill_betweenx(y, x1, x2) : y축 위에 있는 두 개의 선(x1, x2) 사이에 색칠된 영역을 만듦
            # - y와 x1, x2가 모두 같은 길이여야 한다. (scala인 경우 broadcasting 가능)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values,
                             facecolor=color, edgecolor=color, alpha=0.7)       # 0부터 실루엣 스코어까지의 길이만큼 색칠하라는 뜻.

            # 클러스터 번호 표시
            # - x좌표: 0보다 왼쪽
            # - y좌표: 높이의 중간
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # 다음 막대 시작 위치 업데이트
            y_lower = y_upper + 10

        # 전체 실루엣 평균값 수직선 표시
        ax.axvline(x=sil_avg, color="red", linestyle="--")      # 각 경우의 실루엣 계수를 세로로 그려줌.

    plt.tight_layout()
    return plt.show()


def visualize_kmeans_cluster(n_clusters_list, X_features):
    """
    다양한 클러스터 수에 대해 KMeans 클러스터링을 수행하고,
    PCA로 2차원 변환한 결과를 시각화하는 함수.

    Parameters:
    - n_clusters_list : 클러스터 개수 리스트 (예: [2, 3, 4])
    - X_features : 클러스터링에 사용할 입력 데이터 (2차원 배열 또는 DataFrame)
    """
    n_cols = len(n_clusters_list)  # subplot 열 수
    fig, axs = plt.subplots(nrows=1, ncols=n_cols, figsize=(4 * n_cols, 4))

    # axs가 하나일 경우 리스트 형태로 변환 (인덱싱 편의 위해)
    if n_cols == 1:
        axs = [axs]

    # PCA로 입력 데이터를 2차원으로 변환
    pca = PCA(n_components=2)       # 원활한 시각화를 위해 2차원으로 줄임.
    pca_transformed = pca.fit_transform(X_features)
    base_df = pd.DataFrame(pca_transformed, columns=['PCA1', 'PCA2'])

    # 클러스터 수를 바꿔가며 클러스터링과 시각화 반복
    for idx, n_clusters in enumerate(n_clusters_list):
        # KMeans 클러스터링 수행
        kmeans = KMeans(n_clusters=n_clusters, max_iter=500, random_state=0)
        cluster_labels = kmeans.fit_predict(pca_transformed)

        # 클러스터 결과를 복사본에 저장 (base_df를 매번 복사)
        df = base_df.copy()
        df['cluster'] = cluster_labels

        # 클러스터 라벨별로 시각화
        ax = axs[idx]
        unique_labels = np.unique(cluster_labels)
        markers = ['o', 's', '^', 'd', '*', 'D', 'v']  # 7개까지 지원

        for label in unique_labels:     # 클러스터의 종류만큼 돌면서 scatter 출력
            cluster_data = df[df['cluster'] == label]
            marker = markers[label % len(markers)]  # 마커 수 초과 대비. 클러스터의 수가 마커수보다 많을 경우 다시 순회하도록 함.
            cluster_name = f"Cluster {label}"

            ax.scatter(cluster_data['PCA1'], cluster_data['PCA2'],
                       s=70, edgecolor='k', marker=marker, label=cluster_name)

        ax.set_title(f'Number of Clusters: {n_clusters}')
        ax.legend(loc='upper right')

    plt.tight_layout()
    return plt.show()