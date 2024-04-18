import streamlit as st

def inject_scroll_to_bottom():
    # Inject JavaScript to scroll to the bottom of the Streamlit app page.
    js = """
    <script>
        window.scrollTo(0,document.body.scrollHeight);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)
