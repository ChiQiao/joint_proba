import streamlit as st
import session_state

st.sidebar.title("Pages")
radio = st.sidebar.radio(label="", options=["Set A", "Set B", "Add them"])

ss = session_state.get(a=0., b=0.)

if radio == "Set A":
    ss.a = float(st.text_input(label="What is a?", value=ss.a))
    st.write(f"You set a to {ss.a}")
elif radio == "Set B":
    ss.b = float(st.text_input(label="What is b?", value=ss.b))
    st.write(f"You set b to {ss.b}")
elif radio == "Add them":
    st.write(f"a={ss.a} and b={ss.b}")
    button = st.button("Add a and b")
    if button:
        st.write(f"a+b={ss.a+ss.b}")

print([ss.a, ss.b])