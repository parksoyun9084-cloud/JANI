import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


# train data용 preprocess 함수
# LabelEncoder와 StandardScaler를 이용해서 fit_transform해주는 함수
def fit_preprocessing(data):

    '''
    LabelEncoder와 StandardScaler를 이용해서 fit_transform해주는 함수
    :return: data_scaled, encoders, scaler
    '''

    data = data.copy()

    features = ['location', 'subscription_type', 'payment_plan', 'payment_method', 'customer_service_inquiries']

    # encoding
    encoders = {}

    for feature in features:
        encoder = LabelEncoder()
        data[feature] = encoder.fit_transform(data[feature])
        encoders[feature] = encoder

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    data_scaled = pd.DataFrame(
        data_scaled,
        columns=data.columns,
        index=data.index
    )

    return data_scaled, encoders, scaler


# val data / test data용 preprocess 함수
# train data에서 fit한 LabelEncoder와 StandardScaler를 이용해 transform해주는 함수
def transform_preprocessor(data, encoders, scaler):

    '''
    train data에서 fit한 LabelEncoder와 StandardScaler를 이용해 transform해주는 함수
    :return: data_scaled
    '''

    data = data.copy()

    features = ['location', 'subscription_type', 'payment_plan', 'payment_method', 'customer_service_inquiries']

    for feature in features:
        encoder = encoders[feature]
        data[feature] = encoder.transform(data[feature])

    data_scaled = scaler.transform(data)

    data_scaled = pd.DataFrame(
        data_scaled,
        columns=data.columns,
        index=data.index
    )

    return data_scaled




def rfm_df_preprocessing(csv_root):
    model_df = pd.read_csv(csv_root)

    rfm_df = model_df.assign(
        activity_score = model_df["notifications_clicked"],
        adjusted_frequency = model_df["weekly_songs_played"] * (1 - model_df["song_skip_rate"]),
        Monetary = model_df["weekly_hours"],
        Engagement = (
            model_df["num_playlists_created"] +
            model_df["num_platform_friends"] +
            model_df["num_shared_playlists"]
        ),
        subscription_risk = model_df["num_subscription_pauses"],
        support_pressure = model_df["customer_service_inquiries"]
    )[[
        "activity_score",
        "adjusted_frequency",
        "Monetary",
        "Engagement",
        "subscription_risk",
        "support_pressure"
    ]]

    mapping = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
    }

    rfm_df['support_pressure'] = rfm_df['support_pressure'].map(mapping)


    scaler = StandardScaler()

    rfm_scaled_df = scaler.fit_transform(rfm_df)
    rfm_scaled_df = pd.DataFrame(rfm_scaled_df, columns=rfm_df.columns, index=rfm_df.index)

    return rfm_df, rfm_scaled_df