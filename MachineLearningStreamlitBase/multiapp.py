"""Frameworks for running multiple Streamlit applications as a single app.
"""
import streamlit as st

class MultiApp:
    """Framework for combining multiple streamlit applications.
    Usage:
        def foo():
            st.title("Hello Foo")
        def bar():
            st.title("Hello Bar")
        app = MultiApp()
        app.add_app("Foo", foo)
        app.add_app("Bar", bar)
        app.run()
    It is also possible keep each application in a separate file.
        import foo
        import bar
        app = MultiApp()
        app.add_app("Foo", foo.app)
        app.add_app("Bar", bar.app)
        app.run()
    """
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # st.markdown("""<style>.big-font {font-size:100px !important;}</style>""", unsafe_allow_html=True)
        # st.markdown('<p class="big-font">Hello World !!</p>', unsafe_allow_html=True)
        # app = st.sidebar.radio(
        
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
        st.write("""<style>font-size:100px !important;</style>""", unsafe_allow_html=True)
        st.markdown(
        """<style>
        .boxBorder1 {
            outline-offset: 5px;
            font-size:20px;
        }</style>
        """, unsafe_allow_html=True) 
        st.markdown('<div class="boxBorder1"><font color="black">Click the button.</font></div>', unsafe_allow_html=True)
        app = st.radio(
            '',
            self.apps,
            format_func=lambda app: '{}'.format(app['title']))
            # format_func=lambda app: '<p class="big-font">{} !!</p>'.format(app['title']))

        app['function']()