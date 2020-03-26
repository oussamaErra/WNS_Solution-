# this is the model that scored 75.28
import pandas as pd
import lightgbm as lgbm
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

warnings.simplefilter("ignore")
from datetime import datetime
from sklearn.metrics import roc_auc_score
import os

##  dASHBOARD ##########

kaggle = False
save_test_prediction = True
use_target_encoding = True
BEST_ROUNDS = 400
try_one_day = False
time_cols = ["impression_time"]
cols_to_labelencode = []
cols_to_target_encode = []
cols_to_mean_price_encode = []
file_name = "test_the_test_2"
categorical_features = ["category_1", "category_2", "category_3", "product_type"]

if kaggle:
    input_path = "../input/wizardanalyticsdata/train_data/train_data/"
else:
    input_path = "./"
train = pd.read_csv(os.path.join(input_path, "train.csv"))
test = pd.read_csv(os.path.join(input_path, "test.csv"))
view_logs = pd.read_csv(os.path.join(input_path, "view_log.csv"))
items_data = pd.read_csv(os.path.join(input_path, "item_data.csv"))

########### prepare categorical data #############


def encode_categories_items(category, items_data):
    gb = items_data.groupby(category).agg(
        {
            "item_price": {category + "_pr_mn": "mean", category + "_pr_var": "var",},
            "item_id": {category + "_itm_cnt": "nunique"},
            category: {category + "_size": "size"},
        }
    )
    gb.columns = [k[1] for k in gb.columns]
    return gb.reset_index()


for col in categorical_features:
    items_data = pd.merge(
        items_data, encode_categories_items(col, items_data), on=col, how="left"
    )


############"" Converting times cols to dates ###############

view_logs["server_time"] = pd.to_datetime(view_logs["server_time"])
train["impression_time"] = pd.to_datetime(train["impression_time"])
test["impression_time"] = pd.to_datetime(test["impression_time"])


def app_code_features(train, test, view_logs, items_data):
    all_data = pd.concat((train, test))
    all_data = pd.merge(all_data, view_logs, on="user_id", how="left")
    all_data = pd.merge(all_data, items_data, on="item_id", how="left")
    all_data_gb = all_data.groupby("app_code").agg(
        {
            "app_code": ["size"],
            # "is_4G":['nunique'],
            # "os_version":["nunique"],
            "session_id": ["nunique"],
            "item_id": ["nunique"],
            "item_price": ["mean", "median", "sum", "max", "nunique"],
            "user_id": ["nunique"],
            "category_1_size": ["mean"],
            "category_2_size": ["sum", "mean"],
            "category_3_size": ["sum", "mean"],
            "category_1_pr_mn": ["mean"],
            "category_2_pr_mn": ["sum", "mean"],
            "category_3_pr_mn": ["sum", "mean"],
            "category_1_itm_cnt": ["sum", "mean"],
            "category_2_itm_cnt": ["sum", "mean"],
            "category_3_itm_cnt": ["sum", "mean"],
            "product_type_pr_mn": ["sum", "mean"],
            "product_type_itm_cnt": ["sum", "mean"],
            "product_type_size": ["mean"],
            "device_type": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_1": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_2": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_3": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "product_type": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            #'impression_time':{'life':lambda x: np.max(x) -np.min(x)}
        }
    )
    all_data_gb.columns = ["_".join(k) + "_app" for k in all_data_gb.columns]
    # all_data_gb.impression_time_life_app = all_data_gb.impression_time_life_app / np.timedelta64(1, 'D')

    return all_data_gb.reset_index()


def session_id_features(view_logs, items_data):
    all_data = pd.merge(view_logs, items_data, on="item_id", how="left")
    all_data["server_hour"] = all_data.server_time.dt.hour
    all_data["server_day"] = all_data.server_time.dt.dayofyear
    all_data_gb = all_data.groupby("session_id").agg(
        {
            "user_id": ["nunique"],
            "server_time": {
                "life": lambda x: (np.max(x) - np.min(x)) / np.timedelta64(1, "h")
            },
            "item_id": ["nunique"],
        }
    )  # ,'server_hour':['nunique'],'server_day':['nunique'] })
    all_data_gb.columns = ["_".join(k) + "_session" for k in all_data_gb.columns]
    return all_data_gb.reset_index()


try:
    all_data_session = pd.read_csv("all_data_session.csv")
except:
    print("all_data session not found, generating one now ...")
    all_data_session = session_id_features(view_logs, items_data)


def user_features(view_logs, items_data, train, test):
    # train_test_all = pd.concat((train,test))
    all_data = pd.merge(view_logs, items_data, on="item_id", how="left")
    all_data = pd.merge(all_data, all_data_session, on="session_id", how="left")
    # all_data = pd.merge(all_data,train_test_all,on='user_id',how='left')
    all_data["server_hour"] = all_data.server_time.dt.hour
    all_data["server_day"] = all_data.server_time.dt.dayofyear
    all_data_gb = all_data.groupby("user_id").agg(
        {
            "user_id": ["size"],
            # "app_code": {"m_c": lambda x : Counter(x).most_common(1)[0][0]},#,'n_unique':lambda x : len(np.unique(x))},
            "session_id": ["nunique"],
            "item_id": ["nunique"],
            "item_price": ["mean", "median", "sum", "max", "nunique", "var"],
            "device_type": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_1": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_2": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_3": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "product_type_size": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "product_type_itm_cnt": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "product_type_pr_mn": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "product_type": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
            },
            "category_1_size": ["mean"],
            "category_2_size": ["mean"],
            "category_3_size": ["mean"],
            "category_1_pr_mn": ["mean"],
            "category_2_pr_mn": ["mean"],
            "category_3_pr_mn": ["mean"],
            "category_1_itm_cnt": ["mean"],
            "category_2_itm_cnt": ["sum", "mean"],
            "category_3_itm_cnt": ["sum", "mean"],
            "product_type_pr_mn": ["sum", "mean"],
            "product_type_itm_cnt": ["sum", "mean"],
            "product_type_size": ["sum", "mean"],
            "server_time_life_session": ["max", "mean", "sum"],
            "item_id_nunique_session": ["max", "mean", "nunique"],
            "server_hour": {
                "m_c": lambda x: Counter(x).most_common(1)[0][0],
                "nq": lambda x: len(np.unique(x)),
                "mean": lambda x: np.mean(x),
            },
            #'server_day': {
            #'m_c': lambda x: Counter(x).most_common(1)[0][0], \
            #'nq': lambda x: len(np.unique(x)), \
            #'mean': lambda x: np.mean(x)
            # },
            "server_time": {
                "life": lambda x: (np.max(x) - np.min(x)) / np.timedelta64(1, "h")
            }
            #'life_avg':lambda x: ((np.max(x) -np.min(x))/np.timedelta64(1,'D'))/x.shape[0]}
            #'std':lambda x: np.std(np.max(x) - x /np.timedelta64(1,'D'))},
            # ,'life_per_itm':lambda x: (1/x.shape[0])*(np.max(x) -np.min(x))/np.timedelta64(1,'D')}
        }
    )
    all_data_gb.columns = ["_".join(k) + "_user" for k in all_data_gb.columns]
    # all_data_gb['user_per_session_user'] = all_data_gb['user_id_size_user']/all_data_gb['session_id_nunique_user']
    # all_data_gb['life_per_session_user'] = all_data_gb['server_time_life_user'] / all_data_gb['session_id_nunique_user']
    # all_data_gb = all_data_gb.impression_time_life / np.timedelta64(1, 'D')
    return all_data_gb.reset_index()


def encode_feat_by_feat(
    d_train, public_set, private_set, test, to_encode, to_use_to_encode, function="mean"
):
    dict_encode = (
        d_train.groupby(to_encode)
        .agg({to_use_to_encode: function})
        .to_dict()[to_use_to_encode]
    )
    d_train[to_encode + "_" + to_use_to_encode] = d_train[to_encode].apply(
        lambda x: dict_encode.get(x, 0)
    )
    test[to_encode + "_" + to_use_to_encode] = test[to_encode].apply(
        lambda x: dict_encode.get(x, 0)
    )
    if public_set is not None:
        public_set[to_encode + "_" + to_use_to_encode] = public_set[to_encode].apply(
            lambda x: dict_encode.get(x, 0)
        )
        private_set[to_encode + "_" + to_use_to_encode] = private_set[to_encode].apply(
            lambda x: dict_encode.get(x, 0)
        )

    return d_train, public_set, private_set, test


all_data_user = user_features(view_logs, items_data, train, test)


all_data_app_code = app_code_features(train, test, view_logs, items_data)


def process_impressions_data(impress_data):
    impress_data["impression_time_hour"] = pd.to_datetime(
        impress_data.impression_time
    ).dt.hour

    return impress_data


def return_publc_private(train, view_logs):
    train.impression_time = pd.to_datetime(train.impression_time)
    view_logs["server_time"] = pd.to_datetime(view_logs["server_time"])
    start_public = datetime.strptime("2018-12-07", "%Y-%m-%d")
    limit_view_log = datetime.strptime("2018-12-07", "%Y-%m-%d")
    end_public = datetime.strptime("2018-12-10", "%Y-%m-%d")
    d_train = train.loc[train.impression_time < start_public]
    public_set = train.loc[
        (train.impression_time > start_public) & (train.impression_time <= end_public)
    ]
    private_set = train.loc[train.impression_time > end_public]
    return (
        d_train,
        public_set,
        private_set,
        view_logs[view_logs.server_time < limit_view_log],
    )


def encode_target_features(d_train, public_set, private_set, test, col, target):
    skf = StratifiedKFold(n_splits=5, random_state=1991, shuffle=True)
    train_list = []
    for t, v in skf.split(d_train, d_train[col]):
        t_train = d_train.iloc[t, :]
        v_train = d_train.iloc[v, :]
        encod_dict = t_train.groupby(col).agg({target: "mean"}).to_dict()[target]
        v_train[col + "_encode_target"] = v_train[col].apply(
            lambda x: encod_dict.get(x, float("nan"))
        )
        train_list.append(v_train)
    encod_dict_train = d_train.groupby(col).agg({target: "mean"}).to_dict()[target]
    test[col + "_encode_target"] = test[col].apply(
        lambda x: encod_dict_train.get(x, float("nan"))
    )
    if public_set is not None:
        public_set[col + "_encode_target"] = public_set[col].apply(
            lambda x: encod_dict_train.get(x, float("nan"))
        )
        private_set[col + "_encode_target"] = private_set[col].apply(
            lambda x: encod_dict_train.get(x, float("nan"))
        )
    return pd.concat(train_list), public_set, private_set, test


def train_test_features(train, test):
    n_train = train.shape[0]
    # train.sort_values(by =['impression_time'],inplace=True,ascending=True)
    # test.sort_values(by=['impression_time'], inplace=True, ascending=True)
    data_all = pd.concat((train, test))
    data_all["day_impression"] = data_all.impression_time.dt.dayofyear
    data_all["hour_impression"] = data_all.impression_time.dt.hour

    group_pass = (
        data_all.groupby(["hour_impression", "day_impression", "user_id"])
        .size()
        .to_frame("size_userd_per_day")
        .reset_index()
    )
    data_all = pd.merge(
        data_all,
        group_pass,
        on=["user_id", "day_impression", "hour_impression"],
        how="left",
    )

    group_pass = (
        data_all.groupby(["hour_impression", "day_impression", "app_code", "user_id"])
        .size()
        .to_frame("size_user_per_app_code_per_hour")
        .reset_index()
    )
    data_all = pd.merge(
        data_all,
        group_pass,
        on=["app_code", "user_id", "hour_impression", "day_impression"],
        how="left",
    )

    group_pass = (
        data_all.groupby(["hour_impression", "day_impression", "user_id"])
        .agg(
            {
                "app_code": {
                    "n_q_app_per_user_per_hour": "nunique",
                    "app_code_sz_by_nq_per_h": lambda x: x.shape[0] / len(np.unique(x)),
                }
            }
        )
        .reset_index()
    )
    data_all = pd.merge(
        data_all,
        group_pass,
        on=["user_id", "hour_impression", "day_impression"],
        how="left",
    )

    group_pass = (
        data_all.groupby(["day_impression", "user_id"])
        .agg(
            {
                "app_code": {
                    "n_q_app_per_user_per_day": lambda x: len(np.unique(x)),
                    "app_code_sz_by_nq_per_day": lambda x: x.shape[0]
                    / len(np.unique(x)),
                }
            }
        )
        .reset_index()
    )
    data_all = pd.merge(
        data_all, group_pass, on=["user_id", "day_impression"], how="left"
    )
    print(data_all.columns)

    # group_pass = data_all.groupby(['hour_impression', 'day_impression', 'app_code']). \
    #    agg({'user_id': {'n_q_user_per_app_per_hour': lambda x : len(np.unique(x))}}).reset_index()
    # data_all = pd.merge(data_all, group_pass, on=['app_code', 'hour_impression', 'day_impression'], how='left')

    # group_pass = data_all.groupby(['day_impression', 'app_code']).agg({'user_id': \
    #                          { 'n_q_user_per_user_per_day': lambda x: x.shape[0], }}).reset_index()
    # data_all = pd.merge(data_all, group_pass, on=['app_code', 'day_impression'], how='left')

    # group_pass = data_all.groupby(['hour_impression','day_impression','app_code']).size()\
    #                        .to_frame('size_app_code_per_day').reset_index()
    # data_all = pd.merge(data_all,group_pass,on=['app_code','day_impression','hour_impression'],how='left')

    # data_all.drop('day_impression',axis= 1,inplace=True)
    data_all.reset_index(inplace=True, drop=True)
    # data_all['app_is_one'] =
    data_all["n_show_user"] = data_all.groupby("user_id")["user_id"].transform("size")
    # data_all['n_show_user_per_plat'] = data_all.groupby('user_id')['app_code'].transform('nunique')
    data_all["n_show_plalt"] = data_all.groupby("app_code")["app_code"].transform(
        "size"
    )
    data_all["ratio_type_1"] = data_all["n_show_plalt"] / data_all["n_show_user"]
    return (
        data_all.iloc[:n_train, :],
        data_all.drop("is_click", axis=1).iloc[n_train:, :].reset_index(drop=True),
    )


############## generating time features #############"
train = process_impressions_data(train)
test = process_impressions_data(test)

train, test = train_test_features(train, test)
######### split data to train, public and private data sets
if save_test_prediction:
    d_train, public_set, private_set = train, None, None
else:
    d_train, public_set, private_set, view_logs = return_publc_private(train, view_logs)


train = pd.merge(train, all_data_user, on="user_id", how="left")
d_train = pd.merge(d_train, all_data_user, on="user_id", how="left")
test = pd.merge(test, all_data_user, on="user_id", how="left")
if public_set is not None:
    public_set = pd.merge(public_set, all_data_user, on="user_id", how="left")
    private_set = pd.merge(private_set, all_data_user, on="user_id", how="left")

train = pd.merge(train, all_data_app_code, on="app_code", how="left")
d_train = pd.merge(d_train, all_data_app_code, on="app_code", how="left")
test = pd.merge(test, all_data_app_code, on="app_code", how="left")
if public_set is not None:
    public_set = pd.merge(public_set, all_data_app_code, on="app_code", how="left")
    private_set = pd.merge(private_set, all_data_app_code, on="app_code", how="left")

#########################################################################
cols_to_labelencode.extend(
    ["os_version", "device_type_m_c_user", "device_type_m_c_app"]
)
encoders = {}
for col in set(cols_to_labelencode).intersection(set(d_train.columns)):
    encoders[col] = LabelEncoder()
    encoders[col].fit(
        train[col].map(str).values.tolist() + test[col].map(str).values.tolist()
    )
    d_train[col] = encoders[col].transform(d_train[col].map(str))
    test[col] = encoders[col].transform(test[col].map(str))
    if public_set is not None:
        public_set[col] = encoders[col].transform(public_set[col].map(str))
        private_set[col] = encoders[col].transform(private_set[col].map(str))


cols_to_mean_price_encode.extend(["app_code"])
cols_to_mean_price_encode.extend([k for k in d_train.columns if "m_c" in k])
# cols_to_mean_price_encode.extend(['app_code'])

for col in set(cols_to_mean_price_encode).intersection(set(d_train.columns)):
    d_train, public_set, private_set, test = encode_feat_by_feat(
        d_train,
        public_set,
        private_set,
        test,
        col,
        "item_price_mean_user",
        function="mean",
    )


###########################################################
id_cols = ["impression_id", "user_id", "day_impression", "hour_impression"]
time_cols = ["impression_time"]  # ,'server_time_min','server_time_max']

cols_to_drop = []

cols_to_drop += id_cols
cols_to_drop += time_cols
# cols_to_drop+=['is_4G','os_version']

target = ["is_click"]

### generate time features


if use_target_encoding:
    d_train, public_set, private_set, test = encode_target_features(
        d_train, public_set, private_set, test, "app_code", "is_click"
    )

d_train["show_encoding_mult"] = d_train.app_code_encode_target * d_train.n_show_plalt
d_train["show_encoding_dvid"] = d_train.app_code_encode_target / d_train.n_show_plalt


test["show_encoding_mult"] = test.app_code_encode_target * test.n_show_plalt
test["show_encoding_dvid"] = test.app_code_encode_target / test.n_show_plalt

#################

params = {
    "objective": "binary",
    "boosting": "gbdt",
    "num_leaves": 250,
    "metric": "auc",
    "learning_rate": 0.01,
    "bagging_freq": 5,
    "bagging_fraction": 0.9,
    "feature_fraction": 0.5,
    "n_jobs": -1,
    "lambda_l1": 1.7,
    "lambda_l1": 1.7,
    "min_gain_to_split": 0.1,
}

# d_train.sort_values(by=['impression_time'],ascending =True,inplace=True)

skf = StratifiedKFold(n_splits=5, random_state=1991)
d_train.reset_index(drop=True, inplace=True)
preds_public = []
preds_private = []
scores_private = []
scores_public = []
validation_score = []
preds_test = []
oof = np.zeros(len(d_train))
for i, (t_index, val_index) in enumerate(skf.split(d_train, d_train[target[0]])):
    df_train = lgbm.Dataset(
        d_train.drop(cols_to_drop + target, axis=1).iloc[t_index, :],
        d_train[target[0]].values[t_index],
    )
    df_val = lgbm.Dataset(
        d_train.drop(cols_to_drop + target, axis=1).iloc[val_index, :],
        d_train[target[0]].values[val_index],
    )

    model = lgbm.train(
        params,
        df_train,
        num_boost_round=10000,
        valid_sets=[df_train, df_val],
        valid_names=["train", "val"],
        early_stopping_rounds=150,
    )
    oof[val_index] = model.predict(
        d_train.drop(cols_to_drop + target, axis=1).iloc[val_index, :],
        num_iteration=model.best_iteration,
    )
    validation_score.append(model.best_score["val"]["auc"])
    if public_set is not None:
        preds_private.append(
            model.predict(private_set.drop(cols_to_drop + target, axis=1))
        )
        preds_public.append(
            model.predict(public_set.drop(cols_to_drop + target, axis=1))
        )
        scores_private.append(
            roc_auc_score(private_set[target[0]].values, preds_private[i])
        )
        scores_public.append(
            roc_auc_score(public_set[target[0]].values, preds_public[i])
        )
        print("the auc on the private set is {}".format(scores_private[i]))
        print("the auc on the public set is {}".format(scores_public[i]))
    else:
        preds_test.append(
            model.predict(
                test.drop(cols_to_drop, axis=1), num_iteration=model.best_iteration
            )
        )


if public_set is not None:
    print(
        "-------------------Validation -----------------------------------------------"
    )
    print("the validation scores are {}".format(validation_score))
    print(
        "-------------------Public-----------------------------------------------------------"
    )
    print("the public scores are {}".format(scores_public))
    print(
        "the cross validation score of public is {}".format(
            roc_auc_score(public_set[target[0]].values, np.mean(preds_public, axis=0))
        )
    )
    print(
        "--------------------Private----------------------------------------------------------"
    )
    print("the private scores are {}".format(scores_private))
    print(
        "the cross validation score of private is {}".format(
            roc_auc_score(private_set[target[0]].values, np.mean(preds_private, axis=0))
        )
    )

else:
    print("the mean of the cross validation is {}".format(np.mean(validation_score)))
    test["is_click"] = np.mean(preds_test, axis=0)
    try:
        import os

        os.mkdir(f"{file_name}")
    except:
        pass
    test[["impression_id", "is_click"]].to_csv(
        f"{file_name}/{file_name}_test.csv", index=None
    )
    a = pd.DataFrame()
    a["impression_id"] = d_train["impression_id"].values
    a["is_click"] = oof.reshape(-1, 1)
    a.to_csv(f"{file_name}/{file_name}_oof.csv", index=False)
