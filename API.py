import json

import flask
import joblib
import numpy as np
import pandas as pd
from flask import request

api = flask.Flask(__name__)
api.config["DEBUG"] = False

# Chargement du tableau et du modèle
df = pd.read_pickle("data.gz")

df.set_index("SK_ID_CURR", inplace=True)
feats = np.genfromtxt('features.csv', dtype='unicode', delimiter=',')

model = joblib.load("pipeline_housing.joblib")

explainer = joblib.load("shap_explainer.joblib")


def get_prediction(client_id):
    return model.predict_proba([df[feats].loc[client_id]])[0, 1]


def explain(client_id):
    return explainer.shap_values(df[feats].loc[client_id].to_numpy().reshape(1, -1))[1][0][:]


@api.route('/', methods=['GET'])
def index():
    return {'message': 'Bonjour, Balakissa !'}


@api.route('/predict', methods=["GET"])
def proba():
    print(request.args["client_id"])
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = get_prediction(client_id)
        return {"proba": pred}
    else:
        return "Error"


@api.route('/age_max/', methods=['GET'])
def age_max():
    age = round((df["DAYS_BIRTH"] / 365), 0)
    age = -age
    age_Max = age.max()
    return {"age_p_max": age_Max}


@api.route('/age_min/', methods=['GET'])
def age_min():
    age = round((df["DAYS_BIRTH"] / 365), 0)
    age = -age
    age_Min = age.min()
    return {"age_p_min": age_Min}


@api.route('/score_min/', methods=['GET'])
def score_min():
    return {"score_min": 0.55}


@api.route('/predictions', methods=["GET"])
def prediction():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = get_prediction(client_id)
        return {"prediction": pred}
    else:
        return "Error"


@api.route('/age_population', methods=["GET"])
def age_population():
    ages = round((df["DAYS_BIRTH"] / -365), 0).values.tolist()
    return json.dumps(ages)


def age_pop_id(client_id):
    return round((df["DAYS_BIRTH"] / -365), 0).loc[client_id]


@api.route('/age_population_id', methods=["GET"])
def age_du_client():
    print(request.args["client_id"])
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        pred = get_prediction(client_id)
        age = age_pop_id(client_id)
        return {"age_du_client": age, "prediction": pred}
    else:
        return "Error"


def infos(data):
    lst_infos = [data.shape[0], round(data["AMT_INCOME_TOTAL"].mean(), 2), round(data["AMT_CREDIT"].mean(), 2)]
    return lst_infos


@api.route('/infos', methods=["GET"])
def infos_clients():
    lst_infos = infos(df)
    nb_credits = lst_infos[0]
    rev_moy = lst_infos[1]
    credits_moy = lst_infos[2]
    targets = df['TARGET'].value_counts()
    targets = json.dumps(targets.tolist())
    print(targets)
    return {
        "nb_credits": nb_credits,
        "revenu_moy": rev_moy,
        "credit_moyen": credits_moy,
        "targets": targets
    }


def data_client(client_id):
    return df[feats].loc[client_id]


@api.route('/identite_client', methods=["GET"])
def identite_client():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        client = data_client(client_id)
        gender = client["CNT_CHILDREN"]
        print(gender)
        return {
            "gender": client["CODE_GENDER"],
            "age": client["DAYS_BIRTH"],
            "family_status": "",  # client["NAME_FAMILY_STATUS"],
            "number_of_children": client["CNT_CHILDREN"],
            "income_total": client["AMT_INCOME_TOTAL"],
            "credit_amount": client["AMT_CREDIT"],
            "credit_annuities": client["AMT_ANNUITY"],
            "mount_of_property_for_credit": client["AMT_GOODS_PRICE"]
        }
    else:
        "Error"


@api.route('/income_population', methods=["GET"])
def income_population():
    df_income = df.loc[df['AMT_INCOME_TOTAL'] < 200000, :]
    df_income = df_income['AMT_INCOME_TOTAL'].values.tolist()
    print(df_income)
    return json.dumps(df_income)


@api.route('/feats/', methods=["GET"])
def feats_ret():
    return json.dumps(feats.tolist())


@api.route('/features_importances', methods=["GET"])
def importances():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        shap_vals = explain(client_id).tolist()
        return json.dumps(shap_vals)
    else:
        return "Error"


@api.route('/bar', methods=["GET"])
def bar():
    if 'client_id' in request.args:
        client_id = int(request.args["client_id"])
        feat = str(request.args["feature"])

        dff = df[feat]
        retour = [float(dff.loc[client_id]), np.mean(dff)]
        del dff

        return json.dumps(retour)
    else:
        return "Error"


@api.route('/boxplot', methods=["GET"])
def boxplot():
    if 'feature' in request.args:
        feat = str(request.args["feature"])

        dff = df[feat]

        return json.dumps(dff.tolist())
    else:
        return "Error"


if __name__ == "__main__":
    api.run(port=8083)
