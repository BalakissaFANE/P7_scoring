import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import plotly.express as px
from sklearn.cluster import KMeans
import requests
import joblib

plt.style.use('fivethirtyeight')


# sns.set_style('darkgrid')


def main():
    # global infos_client, data
    # Chargement du tableau et du modèle
    data = pd.read_pickle("data.gz")

    model = joblib.load("pipeline_housing.joblib")
    explainer = joblib.load("shap_explainer.joblib")

    feats = requests.get("http://127.0.0.1:8083/feats/").json()
    # features = np.genfromtxt('features_import.csv', dtype='unicode', delimiter=',')

    #######################################
    # SIDEBAR
    #######################################

    # Title display
    html_temp = """
    <div style="background-color: tomato; padding:10px; border-radius:10px">
    <h1 style="color: white; text-align:center">Dashboard Scoring Credit</h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center">Credit decision support…</p>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Customer ID selection
    st.sidebar.header("**General Info**")

    # Loading selectbox
    # id_client = np.genfromtxt('client_id.csv', dtype='unicode', delimiter=',')
    id_client = np.genfromtxt('custumers.csv', dtype='unicode', delimiter=',')
    chk_id = st.sidebar.selectbox("Client ID", id_client)

    # Loading general info
    infos = requests.get("http://127.0.0.1:8083/infos")
    nb_credits = infos.json()["nb_credits"]
    rev_moy = infos.json()["revenu_moy"]
    credits_moy = infos.json()["credit_moyen"]
    # targets = infos.json()["targets"]
    targets = data.TARGET.value_counts()

    # Display of information in the sidebar ###
    # Number of loans in the sample
    st.sidebar.markdown("<u>Number of loans in the sample :</u>", unsafe_allow_html=True)
    st.sidebar.text(nb_credits)

    # Average income
    st.sidebar.markdown("<u>Average income (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(rev_moy)

    # AMT CREDIT
    st.sidebar.markdown("<u>Average loan amount (USD) :</u>", unsafe_allow_html=True)
    st.sidebar.text(credits_moy)

    # """ Pieplot de la solvabilité clients, prend un paramètre (dataframe)"""
    # % de valeurs 'TARGET' différentes
    # target_values = data['TARGET'].value_counts() / len(data) * 100
    # Pieplot du % de chaque valeurs différentes de 'TARGET'
    fig1, ax = plt.subplots(figsize=(5, 5))
    plt.title(" Solvabilité des clients", fontsize=20)
    plt.pie(
        targets,
        explode=[0, 0.1],
        labels=['Remboursé', 'Défaut de payement'],
        autopct='%1.1f%%', startangle=90,
        colors=['#2ecc71', '#e74c3c']
    )
    st.sidebar.pyplot(fig1)

    ax, fig2 = plt.subplots(figsize=(5, 5))
    ax = sns.countplot(y='FLAG_OWN_CAR', data=data, order=data['FLAG_OWN_CAR'].value_counts(ascending=False).index)
    ax.set_title('Type de revenu du client')

    for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width() / len(data['FLAG_OWN_CAR']))
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height() / 2
        ax.annotate(percentage, (x, y), fontsize=20, fontweight='bold')
    # st.sidebar.pyplot(fig2)
    st.sidebar.pyplot(fig2.figure)

    #######################################
    # HOME PAGE - MAIN CONTENT
    #######################################
    # Display Customer ID from Sidebar
    st.write("Customer ID selection :", chk_id)

    # Customer information display : Customer Gender, Age, Family status, Children, …
    st.header("**Customer information display**")

    if st.checkbox("Show customer information ?"):

        # infos_client = identite_client(data, chk_id)
        infos_client = requests.get("http://127.0.0.1:8083/identite_client", params={"client_id": chk_id})
        infos_client = infos_client.json()
        st.write("**Gender : **", infos_client["gender"])
        #print('infos_client')
        st.write("**Age : **{:.0f} ans".format(int(infos_client["age"] / -365)))
        #st.write("**Family status : **", infos_client["family_status"])
        #st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS_SEPARATED"])
       # st.write("**Family status : **", infos_client["NAME_FAMILY_STATUS_Singlenotmarried"])
        st.write("**Number of children : **{:.0f}".format(infos_client["number_of_children"]))

        # Age distribution plot
        # data_age = load_age_population(data)
        data_age = requests.get("http://127.0.0.1:8083/age_population").json()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_age, edgecolor='k', color="goldenrod", bins=20)
        ax.axvline(int(infos_client["age"] / -365), color="green", linestyle='--')
        ax.set(title='Customer age', xlabel='Age(Year)', ylabel='')
        st.pyplot(fig)

        st.subheader("*Income (USD)*")
        st.write("**Income total : **{:.0f}".format(infos_client["income_total"]))
        st.write("**Credit amount : **{:.0f}".format(infos_client["credit_amount"]))
        st.write("**Credit annuities : **{:.0f}".format(infos_client["credit_annuities"]))
        st.write("**Amount of property for credit : **{:.0f}".format(infos_client["mount_of_property_for_credit"]))

        # Income distribution plot
        # data_income = load_income_population(data)
        data_income = requests.get("http://127.0.0.1:8083/income_population").json()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_income, edgecolor='k', color="goldenrod", bins=10)
        ax.axvline(int(infos_client["income_total"]), color="green", linestyle='--')
        ax.set(title='Customer income', xlabel='Income (USD)', ylabel='')
        st.pyplot(fig)

        # Relationship Age / Income Total interactive plot
        data_sk = data.reset_index(drop=False)
        data_sk.DAYS_BIRTH = (data_sk['DAYS_BIRTH'] / -365).round(1)
        plt.subplots(figsize=(10, 10))
        fig = px.scatter(data_sk, x='DAYS_BIRTH', y="AMT_INCOME_TOTAL",
                         size="AMT_INCOME_TOTAL", color='CODE_GENDER',
                         # hover_data=['NAME_FAMILY_STATUS', 'CNT_CHILDREN', 'NAME_CONTRACT_TYPE', 'SK_ID_CURR'])
                         hover_data=['CNT_CHILDREN', 'SK_ID_CURR'])

        fig.update_layout({'plot_bgcolor': '#f0f0f0'},
                          title={'text': "Relationship Age / Income Total", 'x': 0.5, 'xanchor': 'center'},
                          title_font=dict(size=20, family='Verdana'), legend=dict(y=1.1, orientation='h'))

        fig.update_traces(marker=dict(line=dict(width=0.5, color='#3a352a')), selector=dict(mode='markers'))
        fig.update_xaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Age", title_font=dict(size=18, family='Verdana'))
        fig.update_yaxes(showline=True, linewidth=2, linecolor='#f0f0f0', gridcolor='#cbcbcb',
                         title="Income Total", title_font=dict(size=18, family='Verdana'))

        st.plotly_chart(fig)

    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    #if st.checkbox("Show importance of global features information ?"):
        # fig = plt.figure(figsize=(10, 10))
        # plt.barh(int(features["features"][:20]), features["importances"][:20])
        #st.bar_chart(data=None, width=0, height=0)
        #plt.title("Feature importances", fontsize=15)
        #plt.show()
    #else:
     #   st.markdown("<i>…</i>", unsafe_allow_html=True)

    if st.checkbox("Show customer importances information ?"):
        shap_vals = requests.get("http://127.0.0.1:8083/features_importances", params={"client_id": chk_id}).json()
        df_feats = pd.DataFrame(shap_vals, columns=["importances"])
        df_feats["feats"] = feats
        df_feats["abs"] = abs(df_feats["importances"])
        df_feats["Influence"] = np.where(df_feats["importances"] < 0, "Negative", "Positive")
        df_feats.sort_values(by="abs", ascending=False, inplace=True)
        df_feats.drop(columns=["abs"], inplace=True)

        fig3 = px.bar(df_feats.iloc[:10],
                      x="importances",
                      y="feats",
                      color="Influence",
                      orientation="h",
                      title="Principales données influant sur le résultat")
        fig3.update_xaxes(title="Impact sur le résultat")
        fig3.update_yaxes(title="Variable étudiée")
        # st.sidebar.write(fig3)
        st.write(fig3)
    else:
        st.markdown("<i>…</i>", unsafe_allow_html=True)

    # Customer solvability display
    st.header("**Customer file analysis**")
    # prediction = load_prediction(sample, chk_id, clf)
    prediction = requests.get("http://127.0.0.1:8083/predictions", params={"client_id": chk_id}).json()["prediction"]
    prediction = round(float(prediction) * 100, 2)
    st.write("**Success probability : **{:.0f} %".format(prediction))

    # Compute decision according to the best threshold
    score_min = requests.get("http://127.0.0.1:8083/score_min/").json()["score_min"] * 100
    if prediction >= score_min:
        decision = "<font color='green'>**LOAN GRANTED**</font>"
    else:
        decision = "<font color='red'>**LOAN REJECTED**</font>"

    st.write("**Decision** *(with threshold " + format(prediction) + " %)* **: **", decision, unsafe_allow_html=True)

    st.markdown("<u>Customer Data :</u>", unsafe_allow_html=True)
    # st.write(identite_client(data, chk_id))
    data_client = data[data['SK_ID_CURR'] == int(chk_id)]
    st.write(data_client)

    # Feature importance / description
    # if st.checkbox("Customer ID {:.0f} feature importance ?".format(chk_id)):
    #if st.checkbox("Customer ID {:.0f} feature importance ?".format(float(chk_id))):
     #   shap.initjs()
        # X = data.iloc[:, :-1]
        # X = X[X['SK_ID_CURR'] == int(chk_id)]
      #  number = st.slider("Pick a number of features…", 0, 20, 5)
       # fig, ax = plt.subplots(figsize=(10, 10))

        #try:
            # shap_values = explainer.shap_values(X)
            # shap.summary_plot(shap_values, X, plot_type="bar", max_display=number, color_bar=False, plot_size=(5, 5))
         #   shap_values = explainer.shap_values(feats)
          #  shap.summary_plot(shap_values, feats, plot_type="bar", max_display=number, color_bar=False,
           #                   plot_size=(5, 5))
        #except ValueError:
           # "#### data is empty."
        #st.pyplot(fig)

        #if st.checkbox("Need help about feature description ?"):
         #   # list_features = description.index.to_list()
           # list_features = model.description.index.to_list()
            #feature = st.selectbox('Feature checklist…', list_features)
            #st.table(model.description.loc[model.description['SK_ID_CURR'] == feature][:1])

   # else:
    #    st.markdown("<i>…</i>", unsafe_allow_html=True)

    # Similar customer files display
    #chk_voisins = st.checkbox("Show similar customer files ?")

    #def load_kmeans(sample):
     #   client = pd.DataFrame(sample.loc[sample.index, :])
      #  df_neighbors = pd.DataFrame(knn.fit_predict(client), index=client.index)
       # df_neighbors = pd.concat([df_neighbors, data], axis=1)
        #return df_neighbors.iloc[:, 1:].sample(10)

    #if chk_voisins:
     #   knn = KMeans(n_clusters=2).fit(data)
      #  st.markdown("<u>List of the 10 files closest to this Customer :</u>", unsafe_allow_html=True)
       # st.dataframe(load_kmeans(data))
        #st.markdown("<i>Target 1 = Customer with default</i>", unsafe_allow_html=True)
    #else:
     #   st.markdown("<i>…</i>", unsafe_allow_html=True)

    #st.markdown('***')
    #st.markdown(
     #   "Thanks for going through this Web App with me! I'd love feedback on this, so if you want to reach out you "
      #  "can find me on [linkedin](https://www.linkedin.com/in/balakissa-diarra-404840230/) or my [website]("
       # "https://localhost:8501.com/). *Code from [Github](https://github.com/BalakissaFANE/P7_scoring)* ❤️")


if __name__ == '__main__':
    main()
