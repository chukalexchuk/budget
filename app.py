import pathlib
import pickle

# from PIL import Image

import plotly.express as px
import spacy_sentence_bert
import pandas as pd
import streamlit as st

# logo_page = Image.open("logo_page.png")
# image = Image.open("logo.png")
st.set_page_config(
    page_title="Your budget",
    # page_icon=logo_page,
    layout="wide",
    menu_items={
        # 'Get Help': '', # links can be added to Readme file
        # 'Report a bug': "",
        'About': "Your budget"
    }
)
#
# sidebar_image = Image.open("logo.png")
# st.sidebar.image(
#     sidebar_image
# )

# loading the models listed at https://github.com/MartinoMensio/spacy-sentence-bert/
nlp = spacy_sentence_bert.load_model('en_stsb_distilbert_base')
# loading the trained model
pickled_model = pickle.load(open('model.pkl', 'rb'))


def csv_setup():
    with st.expander("Upload CSV file", expanded=False):
        csv_file_in = st.file_uploader("", type='csv')

        if csv_file_in is not None:
            csv_file = pathlib.Path(f'./data/{csv_file_in.name}')
            csv_file.parent.mkdir(parents=True, exist_ok=True)  # create folder if not exist
            csv_file.write_bytes(csv_file_in.read())
            df = pd.read_csv(f'./data/{csv_file.name}')  # added file path

            return df


def app():
    ###########
    # Sidebar panel
    ###########
    with st.sidebar:
        st.write("Hi there! Have you ever wondered how much you spend monthly on groceries, restaurants or transport? "
                 "Does your bank have these details? If not, you can use this ML app to predict the categories and "
                 "check your spending (currently working only with ANZ type of transaction files).")
        df_test = csv_setup()
        use_test = st.checkbox("Use pre-uploaded data")
        if use_test:
            df_test = pd.read_csv("data/data_test.csv")

    ###########
    # Main page
    ###########
    preview_data = st.checkbox("Preview uploaded ORIGINAL data")
    if preview_data:
        st.dataframe(df_test)

    col1, col2 = st.columns(2)
    # making predictions using trained model
    with col1:
        try:
            st.subheader("Predicted categories:")
            categories = []
            for code in df_test["Code"].map(str):
                categories.append(pickled_model.predict(nlp(code).vector.reshape(1, -1))[0])
            df_pred = df_test
            df_pred["Category"] = categories
            st.dataframe(df_pred)
        except TypeError:
            st.warning("Please upload your data or use pre-uploaded csv file")
    # summing up spending by category
    with col2:
        try:
            st.subheader("Amount by category:")
            st.dataframe(df_pred.groupby("Category")["Amount"].sum())
        except UnboundLocalError:
            pass
    # plotting pie chart with spending
    try:
        st.subheader("Your spending by category")
        fig_pie = px.pie(df_pred, names="Category")
        st.plotly_chart(fig_pie, use_container_width=True)
    except UnboundLocalError:
        pass


if __name__ == '__main__':
    app()
