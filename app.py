import streamlit as st
from src.frontend.pages import home, folder_upload


def main():
    st.set_page_config(
        page_title="Cockpit Gaze Analysis", page_icon=":eyes:", layout="wide"
    )

    home.show()

    # st.sidebar.title("Navigation")
    # page = st.sidebar.radio("Go to", ["Home", "Folder Upload"])

    # if page == "Home":
    #     home.show()
    # elif page == "Folder Upload":
    #     folder_upload.show()


if __name__ == "__main__":
    main()
