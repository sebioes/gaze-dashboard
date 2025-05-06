import streamlit as st
from src.frontend.pages import home


def main():
    st.set_page_config(
        page_title="Cockpit Gaze Analysis", page_icon=":eyes:", layout="wide"
    )

    home.show()


if __name__ == "__main__":
    main()
